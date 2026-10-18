"""Unprofiled import A/B, allocator lifetimes, and real arrival-replay experiments."""
from dataclasses import asdict
import argparse
import gc
import gzip
import json
import math
from pathlib import Path
import random
from statistics import median
import time
from types import SimpleNamespace
import torch
from c2c_decode.reference import OfficialC2C
from c2c_decode.nano_benchmark import Nano
from c2c_decode.evaluate import token_prompt
from c2c_decode.scheduler import Request, Scheduler
from profiling_utils import capture, Hooks

OUT = Path("results/profiling")


def save(name, value):
    (OUT/name).write_text(json.dumps(value, indent=2)+"\n")


def percentile(values, p):
    values = sorted(values)
    position = (len(values)-1)*p/100
    lo, hi = math.floor(position), math.ceil(position)
    return values[lo]*(hi-position)+values[hi]*(position-lo) if lo != hi else values[lo]


def import_ablation(reference, backend):
    rows, summaries = [], []
    rng = random.Random(20261017)
    for length in (1024, 2048):
        handoffs = [reference.prefill(token_prompt(reference, length-i*17)) for i in range(8)]
        for batch in (1, 2, 4, 8):
            selected = handoffs[:batch]
            for mode in ("loop", "scatter"):
                backend.decoder.import_mode = mode
                backend.run(selected, [[]]*batch, 32)
            pairs = []
            for repeat in range(5):
                order = ["loop", "scatter"]
                rng.shuffle(order)
                pair = {}
                for mode in order:
                    backend.decoder.import_mode = mode
                    row, _ = backend.run(selected, [[]]*batch, 32)
                    row.update(mode=mode, length=length, batch=batch, repeat=repeat, order=order)
                    payload = sum(x.numel()*x.element_size() for h in selected for kv in h.cache for x in kv)
                    row["payload_bytes"] = payload
                    row["effective_payload_GB_s"] = payload/(row["import_ms"]*1e6)
                    rows.append(row)
                    pair[mode] = row
                if pair["loop"]["tokens"] != pair["scatter"]["tokens"]:
                    raise AssertionError("Import optimization changed generated tokens")
                pairs.append(pair["loop"]["import_ms"]/pair["scatter"]["import_ms"])
            summaries.append(dict(length=length, batch=batch, paired_import_speedup=median(pairs),
                min_speedup=min(pairs), max_speedup=max(pairs), faster_pairs=sum(s > 1 for s in pairs)))
            save("import-ablation.json", dict(rows=rows, summaries=summaries, token_equivalence=True))
            print("IMPORT_AB", length, batch, flush=True)
        if length == 2048:
            for mode in ("loop", "scatter"):
                backend.decoder.import_mode = mode
                def imports():
                    seqs = []
                    try:
                        for h in handoffs:
                            seqs.append(backend.decoder.import_cache(h))
                        torch.cuda.synchronize()
                    finally:
                        for seq in seqs:
                            backend.decoder.release(seq)
                capture(imports, OUT/f"import-{mode}.json")
        del selected, handoffs
        gc.collect()
    backend.decoder.import_mode = "scatter"


def memory_lifetimes(reference, backend):
    d = backend.decoder
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    records = []
    epoch = time.perf_counter()

    def record(label, **extra):
        records.append(dict(label=label, elapsed_ms=(time.perf_counter()-epoch)*1000,
            allocated=torch.cuda.memory_allocated(), reserved=torch.cuda.memory_reserved(),
            peak_allocated=torch.cuda.max_memory_allocated(),
            inactive_split_bytes=torch.cuda.memory_stats()["inactive_split_bytes.all.current"], **extra))

    originals = []
    for idx, model in enumerate(reference.model.model_list):
        original = model.forward
        originals.append((model, original))
        def model_done(*args, _fn=original, index=idx, **kwargs):
            output = _fn(*args, **kwargs)
            record(f"model_{index}_forward_end", logits_shape=list(output.logits.shape),
                logits_bytes=output.logits.numel()*output.logits.element_size(),
                logits_dtype=str(output.logits.dtype))
            return output
        model.forward = model_done
    # Rosetta calls .forward directly, so nn.Module forward hooks miss these calls.
    for idx, projector in enumerate(reference.model.projector_list):
        original = projector.forward
        originals.append((projector, original))
        def fuser_done(*args, _fn=original, index=idx, **kwargs):
            record(f"fuser_{index}_start")
            output = _fn(*args, **kwargs)
            record(f"fuser_{index}_end")
            return output
        projector.forward = fuser_done
    torch.cuda.memory._record_memory_history(enabled="all", context="all", stacks="python", max_entries=60000)
    try:
        record("resident_models_and_pool")
        for length in (1024, 2048, 2048):
            record("before_prefill", length=length)
            h = reference.prefill(token_prompt(reference, length))
            torch.cuda.synchronize()
            record("after_prefill_clean_cache_cleared", length=length,
                official_cache_dict_entries=len(reference.model.kv_cache_dict))
            seq = d.import_cache(h)
            torch.cuda.synchronize()
            record("after_import_both_caches_live", length=length)
            del h
            gc.collect()
            record("after_drop_hf_handoff", length=length)
            for _ in range(8):
                logits = d.step([seq])
                seq.append_token(int(logits.argmax(-1).item()))
            d.release(seq)
            del seq, logits
            gc.collect()
            torch.cuda.synchronize()
            record("after_decode_release", length=length)
        snapshot = OUT/"memory-snapshot.pickle"
        torch.cuda.memory._dump_snapshot(str(snapshot))
        with snapshot.open("rb") as source, gzip.open(str(snapshot)+".gz", "wb") as dest:
            import shutil
            shutil.copyfileobj(source, dest)
        snapshot.unlink()
    finally:
        torch.cuda.memory._record_memory_history(enabled=None)
        for module, original in originals:
            module.forward = original
    torch.cuda.empty_cache()
    record("after_empty_cache")
    save("memory.json", dict(records=records, pool_bytes=d.pool_bytes,
        note="PyTorch allocator in one process; graph pools remain resident; does not cover all driver memory."))
    print("MEMORY_COMPLETE", flush=True)
    # Separate timeline capture; shape recording is disabled to avoid held tensor references.
    timeline = Hooks()
    timeline.wrap(reference.receiver, "forward", "receiver_forward")
    timeline.wrap(reference.model.model_list[1], "forward", "sender_forward")
    for i, projector in enumerate(reference.model.projector_list):
        timeline.wrap(projector, "forward", f"fuser_{i}")
    try:
        capture(lambda: reference.prefill(token_prompt(reference, 2048)), OUT/"official-prefill.json", timeline)
    finally:
        timeline.close()


def scheduling(reference, backend):
    rows = []
    ids = {n: token_prompt(reference, n) for n in (128, 512, 1024, 2048)}
    # Small separate warmup, not included in any response-time distribution.
    Scheduler(reference, backend.decoder).run([Request("warm", ids[512], 16, True)])
    for gap in (0, 100, 350):
        for repeat in range(3):
            # Alternate policy order to reduce temperature/order bias.
            for quantum in ([None, 1] if repeat % 2 == 0 else [1, None]):
                requests = [Request(str(i), ids[(128, 512, 1024, 2048)[i % 4]],
                    (16, 32, 64, 128)[(i*3) % 4], True, i*gap) for i in range(24)]
                scheduler = Scheduler(reference, backend.decoder, prefill_limit_per_round=quantum)
                result = scheduler.run(requests)
                rows.append(dict(gap_ms=gap, repeat=repeat, prefill_quantum=quantum,
                    stats=scheduler.last_stats, requests=[asdict(r) for r in result]))
                save("scheduling.json", dict(runs=rows))
                print("SCHEDULE", gap, repeat, quantum, flush=True)
    summaries = []
    for gap in (0, 100, 350):
        for quantum in (None, 1):
            runs = [r for r in rows if r["gap_ms"] == gap and r["prefill_quantum"] == quantum]
            timings = [r["timings"] for run in runs for r in run["requests"]]
            intervals = [b-a for t in timings for a, b in zip(t["token_times_ms"], t["token_times_ms"][1:])]
            request_tpot = [(t["token_times_ms"][-1]-t["token_times_ms"][0])/(len(t["token_times_ms"])-1)
                            for t in timings]
            summary = dict(gap_ms=gap, prefill_quantum=quantum, requests=len(timings),
                mean_request_tpot_ms=sum(request_tpot)/len(request_tpot),
                median_throughput=median(r["stats"]["e2e_tokens_per_s"] for r in runs))
            for name, values in dict(ttft=[t["ttft_ms"] for t in timings],
                    queue=[t["queue_ms"] for t in timings], e2e=[t["e2e_ms"] for t in timings],
                    inter_token_gap=intervals, request_tpot=request_tpot).items():
                summary[name+"_ms"] = {f"p{p}": percentile(values, p) for p in (50, 95, 99)}
            summaries.append(summary)
    save("scheduling.json", dict(runs=rows, summaries=summaries,
        note="Local scheduled-arrival replay, fixed EOS-disabled output lengths. 72 requests/case, empirical tails, not an HTTP production SLA."))


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--section", choices=["all", "import", "memory", "scheduling"], default="all")
    section = parser.parse_args().section
    OUT.mkdir(parents=True, exist_ok=True)
    args = SimpleNamespace(models="models.lock.json", eager=False, import_mode="scatter")
    reference = OfficialC2C(args.models)
    backend = Nano(args)
    try:
        if section in ("all", "import"):
            import_ablation(reference, backend)
        if section in ("all", "memory"):
            memory_lifetimes(reference, backend)
        if section in ("all", "scheduling"):
            scheduling(reference, backend)
    finally:
        backend.close()
    print("EXPERIMENTS_COMPLETE", flush=True)


if __name__ == "__main__":
    main()

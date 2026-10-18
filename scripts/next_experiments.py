"""Paired producer/input experiments, including official C2C correctness gates."""
import argparse
import gc
import json
from pathlib import Path
import random
from statistics import median
import time
from types import SimpleNamespace
import torch
from c2c_decode.optimized import OptimizedC2C
from c2c_decode.reference import OfficialC2C
from c2c_decode.evaluate import token_prompt, difference, metadata
from c2c_decode.nano_benchmark import Nano

OUT = Path("results/next")


def save(name, data):
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT/name).write_text(json.dumps(data, indent=2)+"\n")


def produce(reference, ids, mode):
    if mode == "official":
        return OfficialC2C.prefill(reference, ids)
    if mode == "lean":
        return reference.prefill(ids)
    for h in reference.prefill_steps(ids, int(mode.split("-")[1])):
        if h is not None:
            return h


def cache_difference(a, b):
    exact, max_abs, square, count, dot, na, nb = True, 0., 0., 0, 0., 0., 0.
    for left, right in zip(a, b):
        for x, y in zip(left, right):
            exact &= torch.equal(x, y)
            x, y = x.float(), y.float()
            delta = x-y
            max_abs = max(max_abs, delta.abs().max().item())
            square += delta.square().sum().item()
            count += x.numel()
            dot += (x*y).sum().item()
            na += x.square().sum().item()
            nb += y.square().sum().item()
    return dict(exact=exact, max_abs=max_abs, rmse=(square/count)**.5, cosine=dot/(na*nb)**.5)


@torch.inference_mode()
def producer(reference):
    report = dict(metadata=metadata(SimpleNamespace(models="models.lock.json")), validation=[], trials=[], memory=[])
    modes = ["official", "lean", "chunk-256", "chunk-512", "chunk-1024"]
    for length in (31, 255, 256, 257, 1024, 2048):
        ids = token_prompt(reference, length)
        baseline = produce(reference, ids, "official")
        for mode in modes[1:]:
            actual = produce(reference, ids, mode)
            row = dict(length=length, mode=mode, cache=cache_difference(actual.cache, baseline.cache),
                logits=difference(actual.logits, baseline.logits))
            row["numerical_passed"] = row["logits"]["cosine"] >= .999 and row["logits"]["rmse"] <= .15 and row["logits"]["max_abs"] <= 1
            if mode == "lean" and (not row["cache"]["exact"] or not row["logits"]["top1_match"]):
                raise AssertionError("Lean producer changed KV or first token")
            report["validation"].append(row)
            del actual
        del baseline
        save("producer.json", report)
        print("PRODUCER_VALIDATED", length, flush=True)
    rng = random.Random(20261018)
    for length in (1024, 2048):
        ids = token_prompt(reference, length)
        for mode in modes:
            h = produce(reference, ids, mode)
            del h
        for repeat in range(5):
            order = rng.sample(modes, len(modes))
            for mode in order:
                torch.cuda.synchronize()
                start = time.perf_counter()
                h = produce(reference, ids, mode)
                torch.cuda.synchronize()
                elapsed = (time.perf_counter()-start)*1000
                report["trials"].append(dict(length=length, mode=mode, repeat=repeat, order=order,
                    ms=elapsed, first_token=h.first_token, phases=h.timings))
                del h
        for mode in modes:
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            before = torch.cuda.memory_allocated()
            h = produce(reference, ids, mode)
            peak = torch.cuda.max_memory_allocated()
            del h
            gc.collect()
            report["memory"].append(dict(length=length, mode=mode, baseline=before, peak=peak,
                after=torch.cuda.memory_allocated()))
        save("producer.json", report)
        print("PRODUCER_BENCH", length, flush=True)


@torch.inference_mode()
def cpu_inputs(reference, backend):
    report = dict(trials=[], validation=[])
    rng = random.Random(20261018)
    for length, batch in ((257, 4), (1024, 1), (2048, 8)):
        hs = [reference.prefill(token_prompt(reference, length-17*i)) for i in range(batch)]
        for fast in (False, True):
            backend.decoder.fast_inputs = fast
            backend.run(hs, [[]]*batch, 32)
        for repeat in range(5):
            pair = []
            for fast in rng.sample([False, True], 2):
                backend.decoder.fast_inputs = fast
                row, _ = backend.run(hs, [[]]*batch, 32)
                row.update(fast_inputs=fast, length=length, batch=batch, repeat=repeat)
                report["trials"].append(row)
                pair.append(row["tokens"])
            assert pair[0] == pair[1], "Input buffers changed generated tokens"
        report["validation"].append(dict(length=length, batch=batch, tokens_equal=True))
        del hs
        save("cpu-inputs.json", report)
        print("CPU_INPUTS", length, batch, flush=True)


@torch.inference_mode()
def end_to_end(reference, backend):
    rows, rng = [], random.Random(20261018)
    modes = [("official", False), ("lean", False), ("lean", True), ("chunk-512", True)]
    for length in (1024, 2048):
        for batch, count in ((1, 32), (1, 128), (8, 32)):
            ids = [token_prompt(reference, length-17*i) for i in range(batch)]
            for repeat in range(-1, 5):
                pair = {}
                for mode, fast in rng.sample(modes, len(modes)):
                    backend.decoder.fast_inputs = fast
                    torch.cuda.synchronize()
                    before = time.perf_counter()
                    hs = [produce(reference, prompt, mode) for prompt in ids]
                    prefill_ms = (time.perf_counter()-before)*1000
                    row, _ = backend.run(hs, [[]]*batch, count)
                    row.update(length=length, batch=batch, count=count, mode=mode, fast_inputs=fast,
                        repeat=repeat, prefill_ms=prefill_ms, e2e_ms=(time.perf_counter()-before)*1000)
                    pair[mode, fast] = row["tokens"]
                    if repeat >= 0:
                        rows.append(row)
                    del hs
                assert pair["official", False] == pair["lean", False] == pair["lean", True]
            save("e2e.json", dict(trials=rows, exact_lean_tokens=True,
                note="Fresh serial official prefills + import + decode in one process; no IPC or HTTP. Randomized mode order, five measured repetitions."))
            print("E2E", length, batch, count, flush=True)


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--section", choices=["all", "producer", "cpu", "e2e"], default="all")
    args = p.parse_args()
    reference = OptimizedC2C()
    if args.section in ("all", "producer"):
        producer(reference)
    if args.section in ("all", "cpu", "e2e"):
        backend = Nano(SimpleNamespace(models="models.lock.json", eager=False))
        try:
            if args.section in ("all", "cpu"):
                cpu_inputs(reference, backend)
            if args.section in ("all", "e2e"):
                end_to_end(reference, backend)
        finally:
            backend.close()
    print("EXPERIMENT_COMPLETE", flush=True)


if __name__ == "__main__":
    main()

"""nano-vLLM audit harness with an isolated official C2C producer."""
import argparse
import gc
import importlib.metadata
import json
from pathlib import Path
import time
import torch
from .bridge import OfficialBridge


def sync():
    torch.cuda.synchronize()


def diff(actual, expected):
    a, b = actual.float().flatten(), expected.float().flatten()
    return dict(max_abs=(a-b).abs().max().item(), rmse=(a-b).square().mean().sqrt().item(),
        cosine=torch.nn.functional.cosine_similarity(a, b, dim=0).item(),
        top1_match=a.argmax().item() == b.argmax().item())


class Nano:
    def __init__(self, args):
        from .paged import PagedDecoder
        self.decoder = PagedDecoder(args.models, kv_pool_mib=2048, max_batch=8, graph=not args.eager,
                                    import_mode=getattr(args, "import_mode", "scatter"))
        self.info = dict(pool_bytes=self.decoder.pool_bytes, num_blocks=self.decoder.num_blocks,
            block_size=self.decoder.block_size, graph=not args.eager)

    def run(self, handoffs, histories, count, force=False, check=False):
        d, seqs = self.decoder, []
        diagnostics = [dict(logits=[]) for h in handoffs]
        sync()
        start = time.perf_counter()
        try:
            for h in handoffs:
                seqs.append(d.import_cache(h))
            sync()
            import_ms = (time.perf_counter()-start)*1000
            if check:
                for h, seq, info in zip(handoffs, seqs, diagnostics):
                    exported = d.export_cache(seq, len(h.prompt_ids))
                    info["cache_exact"] = all(torch.equal(a, b) for p, q in zip(h.cache, exported)
                        for a, b in zip(p, q))
                    if not info["cache_exact"]:
                        raise AssertionError("nano KV import changed values")
            for step in range(count-1):
                logits = d.step(seqs)
                if check:
                    for i, info in enumerate(diagnostics):
                        info["logits"].append(logits[i].clone())
                if force:
                    logits.fill_(-float("inf"))
                    for i, history in enumerate(histories):
                        logits[i, history[step+1]] = 0
                tokens = logits.argmax(-1).tolist()
                for seq, token in zip(seqs, tokens):
                    seq.append_token(token)
            sync()
            elapsed = (time.perf_counter()-start)*1000
            result = dict(handoff_decode_ms=elapsed, import_ms=import_ms,
                tokens=[list(s.completion_token_ids) for s in seqs])
        finally:
            for seq in seqs:
                d.release(seq)
        result["all_blocks_reclaimed"] = len(d.blocks.free_block_ids) == d.num_blocks
        if not result["all_blocks_reclaimed"]:
            raise AssertionError("nano leaked request blocks")
        return result, diagnostics

    def close(self):
        self.decoder.close()


def finish_row(row, length, batch, count, repeat, forced):
    row.update(prompt_tokens=length, batch=batch, output_tokens=count, repeat=repeat,
        teacher_forced=forced, decode_tokens=batch*(count-1))
    row["decode_ms_excluding_import"] = row["handoff_decode_ms"]-row["import_ms"]
    row["handoff_tokens_per_s"] = row["decode_tokens"]*1000/row["handoff_decode_ms"]
    row["decode_tokens_per_s"] = row["decode_tokens"]*1000/row["decode_ms_excluding_import"]
    return row


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--models", default="models.lock.json")
    p.add_argument("--producer-python", default="/root/autodl-tmp/envs/c2c-decode/bin/python")
    p.add_argument("--eager", action="store_true")
    p.add_argument("--import-mode", choices=["loop", "scatter"], default="scatter")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--output", required=True)
    args = p.parse_args()
    bridge = OfficialBridge(args.producer_python, args.models)
    print("Official producer ready", flush=True)
    start = time.perf_counter()
    backend = Nano(args)
    report = dict(args=vars(args), official_environment=bridge.info,
        backend_versions={n: importlib.metadata.version(n) for n in
            ["torch", "transformers"]},
        initialization_seconds=time.perf_counter()-start, backend_config=backend.info,
        validation=[], trials=[], e2e=[])
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)

    def save():
        destination.write_text(json.dumps(report, indent=2)+"\n")

    try:
        # Non-aligned lengths prove no fused suffix is accidentally recomputed.
        for length in ([257] if args.smoke else [31, 255, 256, 257, 1024, 2048]):
            h = bridge.prefill(length=length)
            reference = bridge.call("history", key=h.key, count=16, check_steps=15)
            row, info = backend.run([h], [reference["history"]], 16, force=True, check=True)
            comparisons = [diff(a, b) for a, b in zip(info[0]["logits"], reference["logits"])]
            if len(comparisons) != 15:
                raise AssertionError("Missing validation logits")
            report["validation"].append(dict(prompt_tokens=length,
                cache_exact=info[0]["cache_exact"], steps=comparisons,
                forced_history_exact=row["tokens"][0] == reference["history"]))
            key = h.key
            del h, reference, row, info
            bridge.release(key)
            print(f"VALIDATED {length}", flush=True)
            save()
        if not args.smoke:
            handoffs = [bridge.prefill(length=n) for n in [31, 255, 256, 257]]
            references = [bridge.call("history", key=h.key, count=16, check_steps=15) for h in handoffs]
            row, info = backend.run(handoffs, [r["history"] for r in references], 16, force=True, check=True)
            for i, (h, ref, diagnostic) in enumerate(zip(handoffs, references, info)):
                report["validation"].append(dict(prompt_tokens=len(h.prompt_ids), batch=4,
                    cache_exact=diagnostic["cache_exact"],
                    forced_history_exact=row["tokens"][i] == ref["history"],
                    steps=[diff(a, b) for a, b in zip(diagnostic["logits"], ref["logits"])]))
            keys = [h.key for h in handoffs]
            del handoffs, references, row, info, h, ref, diagnostic
            for key in keys:
                bridge.release(key)
            print("VALIDATED mixed batch", flush=True)
            save()
        for length in ([1024] if args.smoke else [1024, 2048]):
            for batch in ([1] if args.smoke else [1, 2, 4, 8]):
                handoffs = [bridge.prefill(length=length-17*i) for i in range(batch)]
                histories = [bridge.call("history", key=h.key, count=128, check_steps=0)["history"]
                             for h in handoffs]
                for count in ([32] if args.smoke or batch > 1 else [32, 128]):
                    for force in (False, True):
                        for repeat in range(-1, 1 if args.smoke else args.repeats):
                            row, info = backend.run(handoffs, histories, count, force=force)
                            if force and any(a != b[:count] for a, b in zip(row["tokens"], histories)):
                                raise AssertionError("Teacher forcing failed")
                            if repeat >= 0:
                                report["trials"].append(finish_row(row, length, batch, count, repeat, force))
                        print(f"BENCH {length=} {batch=} {count=} {force=}", flush=True)
                        save()
                keys = [h.key for h in handoffs]
                del handoffs
                for key in keys:
                    bridge.release(key)
            # Real fresh-request E2E includes IPC and all original C2C phases.
            for repeat in range(-1, 1 if args.smoke else args.repeats):
                sync()
                start = time.perf_counter()
                h = bridge.prefill(length=length)
                row, info = backend.run([h], [[]], 32)
                sync()
                row.update(e2e_ms=(time.perf_counter()-start)*1000,
                    prefill_bridge_ms=h.bridge_prefill_ms, official_phases=h.timings,
                    prompt_tokens=length, repeat=repeat)
                if repeat >= 0:
                    report["e2e"].append(row)
                key = h.key
                del h
                bridge.release(key)
            save()
        report["numerical_passed"] = all(s["cosine"] >= .999 and s["rmse"] <= .15 and s["max_abs"] <= 1
            for case in report["validation"] for s in case["steps"])
        report["strict_top1"] = all(s["top1_match"] for case in report["validation"] for s in case["steps"])
        report["completed"] = True
        save()
    finally:
        backend.close()
        gc.collect()
        bridge.close()
    print("COMPLETE", destination, flush=True)


if __name__ == "__main__":
    main()

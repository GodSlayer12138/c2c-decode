"""GPU validation and reproducible phase-aware benchmark CLI."""
import argparse
from dataclasses import asdict
import importlib.metadata
import json
from pathlib import Path
import platform
import time
import torch
from .reference import OfficialC2C
from .paged import PagedDecoder
from .scheduler import Request, Scheduler


def token_prompt(reference, length):
    base = reference.encode("Explain why a paged KV cache is useful in one short sentence.")
    fill = reference.tokenizer.encode(" The cache stores attention keys and values.", add_special_tokens=False)
    if length < len(base):
        return base[:length-1] + base[-1:]
    extra = length-len(base)
    return base[:-1] + (fill * ((extra+len(fill)-1)//len(fill)))[:extra] + base[-1:]


def metadata(args):
    versions = {}
    for package in ("torch", "transformers", "flash-attn", "triton", "huggingface-hub"):
        versions[package] = importlib.metadata.version(package)
    return {"python": platform.python_version(), "gpu": torch.cuda.get_device_name(),
            "vram_bytes": torch.cuda.get_device_properties(0).total_memory,
            "cuda": torch.version.cuda, "versions": versions, "args": vars(args),
            "upstream": json.loads(Path("upstream.lock.json").read_text()),
            "models": json.loads(Path(args.models).read_text())}


def difference(actual, expected):
    a, b = actual.float().flatten(), expected.float().flatten()
    delta = a-b
    top = b.topk(2).values
    return {"max_abs": delta.abs().max().item(), "rmse": delta.square().mean().sqrt().item(),
            "cosine": torch.nn.functional.cosine_similarity(a, b, dim=0).item(),
            "top1_match": bool(a.argmax() == b.argmax()),
            "reference_top1_margin": (top[0]-top[1]).item(),
            "finite": bool(torch.isfinite(a).all() and torch.isfinite(b).all())}


@torch.inference_mode()
def validate(reference, decoder, args):
    records, owned = [], []
    # Check lossless import at partial/exact block boundaries and 1K/2K.
    lengths = [31, 255, 256, 257, 1024, 2048]
    for start in range(0, len(lengths), decoder.config.max_num_seqs):
        group = lengths[start:start+decoder.config.max_num_seqs]
        handoffs, seqs = [], []
        try:
            for length in group:
                handoff = reference.prefill(token_prompt(reference, length))
                seq = decoder.import_cache(handoff)
                owned.append(seq)
                exported = decoder.export_cache(seq, length)
                exact = all(torch.equal(a, b) for old, new in zip(handoff.cache, exported)
                            for a, b in zip(old, new))
                if not exact or seq.last_token != handoff.first_token:
                    raise AssertionError("KV roundtrip / first-token handoff failed")
                records.append({"prompt_tokens": length, "cache_roundtrip_exact": exact,
                                "first_token_exact": True, "steps": []})
                handoffs.append(handoff)
                seqs.append(seq)
                del exported
            for step in range(args.tokens-1):
                logits = decoder.step(seqs)
                for i, (handoff, seq) in enumerate(zip(handoffs, seqs)):
                    expected, handoff.cache = reference.decode(seq.last_token, handoff.cache)
                    row = difference(logits[i], expected[0])
                    records[start+i]["steps"].append(row)
                    # Teacher forcing keeps the history identical; otherwise a
                    # single argmax tie makes every later numerical comparison invalid.
                    seq.append_token(int(expected.argmax(-1).item()))
                del logits
        finally:
            for seq in owned:
                if seq.seq_id in decoder._live:
                    decoder.release(seq)
            owned.clear()
        del handoffs, seqs

    prompts = [reference.encode(text) for text in (
        "What is 2 + 2? Answer with one number.",
        "List three colors, separated by commas.",
        "用一句话解释什么是缓存。",
        "Complete the sequence: 1, 2, 3, 4,")]
    expected = [reference.generate(ids, args.tokens, ignore_eos=True) for ids in prompts]
    scheduler = Scheduler(reference, decoder)
    requests = [Request(str(i), ids, args.tokens, True) for i, ids in enumerate(prompts)]
    results = scheduler.run(requests)
    greedy = [{"request": str(i), "official_tokens": e, "paged_tokens": r.token_ids,
               "exact": e == r.token_ids, "text": reference.tokenizer.decode(r.token_ids)}
              for i, (e, r) in enumerate(zip(expected, results))]
    # Repeat with shorter outputs to exercise pool reuse and completion mid-batch.
    reuse = scheduler.run([Request(f"reuse-{i}", ids, i+1, True) for i, ids in enumerate(prompts)])
    reuse_ok = all(r.token_ids == expected[i][:i+1] for i, r in enumerate(reuse))
    eos_expected = reference.generate(prompts[0], args.tokens, ignore_eos=False)
    eos_result = scheduler.run([Request("eos", prompts[0], args.tokens, False)])[0]
    eos_ok = eos_expected == eos_result.token_ids
    free_ok = len(decoder.blocks.free_block_ids) == decoder.num_blocks and not decoder._live
    numerical = all(s["finite"] and s["cosine"] >= .999 and s["rmse"] <= .15 and s["max_abs"] <= 1.
                    for record in records for s in record["steps"])
    compared = [s for record in records for s in record["steps"]]
    # Thresholds are predeclared BF16 diagnostics, not a claim of bitwise logits.
    return {"passed": numerical and all(r["exact"] for r in greedy) and reuse_ok and free_ok and eos_ok,
            "numerical_passed": numerical, "thresholds": {"cosine_min": .999, "rmse_max": .15, "max_abs_max": 1.},
            "teacher_forced_top1_matches": sum(s["top1_match"] for s in compared),
            "teacher_forced_comparisons": len(compared),
            "strict_token_equivalence": all(s["top1_match"] for s in compared) and all(r["exact"] for r in greedy),
            "teacher_forced": records, "greedy": greedy,
            "reuse_exact": reuse_ok, "all_blocks_reclaimed": free_ok,
            "eos_generation_exact": eos_ok, "eos_finish_reason": eos_result.finish_reason,
            "scheduler": scheduler.last_stats}


@torch.inference_mode()
def benchmark(reference, decoder, args):
    scheduler = Scheduler(reference, decoder)
    rows = []
    for concurrency in (1, 2, 4, 8):
        if concurrency > decoder.config.max_num_seqs:
            continue
        for length in args.lengths:
            # Warm each measured batch/length, including the special batch=1
            # compiled kernels. Warming only the maximum batch biases early rows.
            scheduler.run([Request(f"warm-{i}", token_prompt(reference, length-17*i), 3, True)
                           for i in range(concurrency)])
            for repeat in range(args.repeats):
                requests = [Request(str(i), token_prompt(reference, length-17*i), args.tokens, True)
                            for i in range(concurrency)]
                torch.cuda.reset_peak_memory_stats()
                results = scheduler.run(requests)
                rows.append({"concurrency": concurrency, "nominal_prompt_tokens": length,
                    "prompt_lengths": [len(r.prompt_ids) for r in requests], "repeat": repeat,
                    "stats": dict(scheduler.last_stats),
                    "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                    "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
                    "requests": [asdict(r) for r in results]})
                print(json.dumps({k: v for k, v in rows[-1].items() if k != "requests"}), flush=True)
    return {"runs": rows, "notes": [
        "All requests arrive at time zero; FIFO prefill batch=1, continuous greedy decode.",
        "TTFT includes queue time; first token comes from official prefill logits.",
        "Decode throughput excludes first tokens; end-to-end throughput includes them.",
        "Fusion_and_overhead includes official cache clones, fusion and host dispatch.",
        "Peak allocated/reserved includes all resident models plus the fixed KV pool.",
        "No quantization, no prefix sharing, no training. JIT warmup excluded."]}


@torch.inference_mode()
def benchmark_reference(reference, args):
    """Batch=1 official cached decode, without a second receiver or paged pool."""
    warm = reference.prefill(token_prompt(reference, 64))
    reference.decode(warm.first_token, warm.cache)
    del warm
    rows = []
    for length in args.lengths:
        warm = reference.prefill(token_prompt(reference, length))
        reference.decode(warm.first_token, warm.cache)
        del warm
        for repeat in range(args.repeats):
            ids = token_prompt(reference, length)
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            started = time.perf_counter()
            handoff = reference.prefill(ids)
            token = handoff.first_token
            tokens = [token]
            torch.cuda.synchronize()
            ttft = (time.perf_counter()-started)*1000
            decode_start = time.perf_counter()
            for _ in range(args.tokens-1):
                logits, handoff.cache = reference.decode(token, handoff.cache)
                token = int(logits.argmax(-1).item())
                tokens.append(token)
            torch.cuda.synchronize()
            decode_ms = (time.perf_counter()-decode_start)*1000
            e2e_ms = (time.perf_counter()-started)*1000
            rows.append({"concurrency": 1, "prompt_tokens": length, "repeat": repeat,
                "token_ids": tokens, "timings": dict(handoff.timings, ttft_ms=ttft,
                    decode_ms=decode_ms, e2e_ms=e2e_ms),
                "decode_tokens_per_s": (args.tokens-1)/(decode_ms/1000),
                "e2e_tokens_per_s": args.tokens/(e2e_ms/1000),
                "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                "peak_reserved_bytes": torch.cuda.max_memory_reserved()})
            del handoff
    return {"runs": rows, "notes": "Official C2C prefill and cached forward; batch=1; no paged model resident; fixed output count."}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["validate", "benchmark", "reference", "baseline"])
    parser.add_argument("--models", default="models.lock.json")
    parser.add_argument("--output", default="results/run.json")
    parser.add_argument("--tokens", type=int, default=16)
    parser.add_argument("--pool-mib", type=int, default=1024)
    parser.add_argument("--batch", type=int, choices=[1, 2, 4, 8], default=8)
    parser.add_argument("--graph", action="store_true")
    parser.add_argument("--lengths", type=int, nargs="+", default=[1024, 2048])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--strict-tokens", action="store_true",
                        help="Also fail validation on any teacher-forced top-1 difference")
    args = parser.parse_args()
    if args.tokens < 4 or args.tokens > 512 or args.repeats < 1:
        parser.error("Use 4..512 output tokens and positive repeats")
    torch.manual_seed(0)
    start = time.perf_counter()
    report = {"metadata": metadata(args)}
    reference = OfficialC2C(args.models)
    report["reference_load_seconds"] = time.perf_counter()-start
    if args.mode == "reference":
        ids = reference.encode("What is 2 + 2? Answer with one number.")
        ids_out = reference.generate(ids, args.tokens)
        report["reference"] = {"tokens": ids_out, "text": reference.tokenizer.decode(ids_out)}
    elif args.mode == "baseline":
        report["baseline"] = benchmark_reference(reference, args)
    else:
        decoder = PagedDecoder(args.models, kv_pool_mib=args.pool_mib,
                               max_batch=args.batch, graph=args.graph)
        try:
            report[args.mode] = (validate if args.mode == "validate" else benchmark)(reference, decoder, args)
        finally:
            decoder.close()
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print("Report:", destination, flush=True)
    if args.mode == "validate" and not report["validate"]["passed"]:
        raise SystemExit("Validation failed; inspect numerical and token diagnostics")
    if args.mode == "validate" and args.strict_tokens and not report["validate"]["strict_token_equivalence"]:
        raise SystemExit("Strict token equivalence failed; BF16 numerical tolerance is a separate gate")


if __name__ == "__main__":
    main()

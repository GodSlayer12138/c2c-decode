"""Matched-work efficiency audit: identical fused KV and teacher-forced history."""
import argparse
import hashlib
import json
from .reporting import write_reports
from pathlib import Path
import random
from statistics import median
import subprocess
import time
import torch
from transformers import DynamicCache
from .evaluate import metadata, token_prompt
from .reference import OfficialC2C
from .paged import PagedDecoder

MODES = ("official_sdpa", "official_flash2", "paged_eager", "paged_graph")


def synchronize():
    torch.cuda.synchronize()


def clone_cache(cache):
    return DynamicCache.from_legacy_cache(tuple((k.clone(), v.clone()) for k, v in cache))


def telemetry():
    return subprocess.check_output([
        "nvidia-smi", "--query-gpu=name,driver_version,memory.used,utilization.gpu,temperature.gpu,power.draw,clocks.sm",
        "--format=csv,noheader"], text=True).strip()


def prepare_history(reference, handoff, count):
    cache, tokens = clone_cache(handoff.cache), [handoff.first_token]
    for _ in range(count-1):
        logits, cache = reference.decode(tokens[-1], cache)
        tokens.append(int(logits.argmax(-1).item()))
    return tokens


def decode_trial(reference, decoder, handoff, history, mode, preserve=True):
    """Compute all logits and argmax in every mode, then force the same next ID.

    Cloning the shared HF prefix is setup, excluded from decode latency. Import
    is separately timed. E2E trials consume fresh HF caches without cloning.
    """
    is_paged = mode.startswith("paged")
    reference.receiver.config._attn_implementation = (
        "flash_attention_2" if mode == "official_flash2" else "sdpa")
    sequence = None
    import_ms = 0.
    try:
        if is_paged:
            decoder.enforce_eager = mode == "paged_eager"
            synchronize()
            start = time.perf_counter()
            sequence = decoder.import_cache(handoff)
            synchronize()
            import_ms = (time.perf_counter()-start)*1000
        else:
            cache = clone_cache(handoff.cache) if preserve else handoff.cache
        synchronize()
        start = time.perf_counter()
        observed = []
        for i, token in enumerate(history[:-1]):
            if is_paged:
                if sequence.last_token != token:
                    raise AssertionError("Teacher-forced histories diverged")
                logits = decoder.step([sequence])
            else:
                logits, cache = reference.decode(token, cache)
            observed.append(int(logits.argmax(-1).item()))
            if is_paged:
                sequence.append_token(history[i+1])
        synchronize()
        elapsed = (time.perf_counter()-start)*1000
        return {"mode": mode, "decode_ms": elapsed, "import_ms": import_ms,
                "decode_tokens": len(history)-1,
                "decode_tokens_per_s": (len(history)-1)/(elapsed/1000),
                "argmax_matches_reference": sum(a == b for a, b in zip(observed, history[1:])),
                "observed_argmax": observed}
    finally:
        reference.receiver.config._attn_implementation = "sdpa"
        if sequence is not None:
            decoder.release(sequence)


def summarize(rows, group_keys, metric, candidates):
    summaries = []
    groups = sorted({tuple(r[k] for k in group_keys) for r in rows})
    for group in groups:
        selected = [r for r in rows if tuple(r[k] for k in group_keys) == group]
        for mode in candidates:
            paired = []
            for repeat in sorted({r["repeat"] for r in selected}):
                by_mode = {r["mode"]: r for r in selected if r["repeat"] == repeat}
                paired.append(by_mode["official_sdpa"][metric] / by_mode[mode][metric])
            baseline = [r[metric] for r in selected if r["mode"] == "official_sdpa"]
            candidate = [r[metric] for r in selected if r["mode"] == mode]
            summaries.append(dict(zip(group_keys, group), mode=mode,
                baseline_median_ms=median(baseline), candidate_median_ms=median(candidate),
                paired_speedup_median=median(paired), paired_speedup_min=min(paired),
                paired_speedup_max=max(paired), faster_pairs=sum(x > 1 for x in paired),
                pairs=len(paired)))
    return summaries


def write_summary(report, destination):
    lines = ["# Matched-work efficiency verification", "",
        "One RTX 4090, BF16, batch=1, fixed output count. Five or more interleaved pairs per case.",
        "All decode paths start from the exact same official fused KV and consume the same token history.",
        "Every step still computes full logits and greedy argmax. EOS is disabled; argmax differences remain recorded.",
        "Cloning shared test fixtures and model/JIT/graph setup are excluded. Cache import is included in E2E.", "",
        "| Prompt | Output | Mode | Official decode ms | Candidate decode ms | Paired speedup | Observed range | Faster pairs |",
        "|---:|---:|---|---:|---:|---:|---:|---:|"]
    for row in report["decode_summary"]:
        lines.append(f"| {row['prompt_tokens']} | {row['output_tokens']} | {row['mode']} | "
            f"{row['baseline_median_ms']:.2f} | {row['candidate_median_ms']:.2f} | "
            f"{row['paired_speedup_median']:.2f}x | {row['paired_speedup_min']:.2f}–{row['paired_speedup_max']:.2f}x | "
            f"{row['faster_pairs']}/{row['pairs']} |")
    lines += ["", "Fresh-prefill E2E pairs (32 output tokens, including prefill/fusion/import/decode):", "",
        "| Prompt | Official E2E ms | Graph E2E ms | Paired speedup | Faster pairs |",
        "|---:|---:|---:|---:|---:|"]
    for row in report["e2e_summary"]:
        lines.append(f"| {row['prompt_tokens']} | {row['baseline_median_ms']:.2f} | "
            f"{row['candidate_median_ms']:.2f} | {row['paired_speedup_median']:.2f}x | "
            f"{row['faster_pairs']}/{row['pairs']} |")
    lines += ["", "Ranges are observed paired extrema, not confidence intervals.",
        "Both HF and nano receivers remain resident in this controlled experiment; it is not a memory-efficiency comparison.",
        "FlashAttention-2 is an additional HF decode control; official prefill remains SDPA in all cases.",
        "This validates warm execution efficiency, not universally identical text or equal task accuracy.",
        "The project integrates existing nano-vLLM kernels and CUDA Graphs; it does not invent those speedups.", ""]
    write_reports(destination, "\n".join(lines))


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", default="models.lock.json")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--output", default="results/efficiency-audit.json")
    args = parser.parse_args()
    if args.repeats < 5:
        parser.error("Use at least five paired repetitions")
    torch.manual_seed(0)
    rng = random.Random(20261017)
    report = {"metadata": metadata(args), "gpu_before": telemetry(),
              "decode_trials": [], "e2e_trials": [], "fixtures": []}
    start = time.perf_counter()
    reference = OfficialC2C(args.models)
    decoder = PagedDecoder(args.models, kv_pool_mib=2048, max_batch=8, graph=True)
    report["initialization_seconds"] = time.perf_counter()-start
    try:
        for length in (1024, 2048):
            ids = token_prompt(reference, length)
            handoff = reference.prefill(ids)
            history = prepare_history(reference, handoff, 128)
            # Validate byte-for-byte import outside the timed region.
            seq = decoder.import_cache(handoff)
            try:
                exported = decoder.export_cache(seq, length)
                if not all(torch.equal(x, y) for a, b in zip(handoff.cache, exported) for x, y in zip(a, b)):
                    raise AssertionError("KV import differs from official fixture")
            finally:
                decoder.release(seq)
            del exported
            report["fixtures"].append({"prompt_tokens": length, "prompt_ids": ids,
                "history": history, "history_sha256": hashlib.sha256(json.dumps(history).encode()).hexdigest(),
                "cache_roundtrip_exact": True})
            for count in (32, 128):
                for mode in MODES:
                    decode_trial(reference, decoder, handoff, history[:count], mode)
                for repeat in range(args.repeats):
                    order = list(MODES)
                    rng.shuffle(order)
                    for mode in order:
                        row = decode_trial(reference, decoder, handoff, history[:count], mode)
                        row.update(prompt_tokens=length, output_tokens=count, repeat=repeat, order=order)
                        report["decode_trials"].append(row)
                    print(f"DECODE prompt={length} output={count} repeat={repeat} complete", flush=True)
            del handoff
            # Fresh request E2E timing, alternating randomized order. No fixture cloning.
            for repeat in range(-1, args.repeats):
                order = ["official_sdpa", "paged_graph"]
                rng.shuffle(order)
                for mode in order:
                    synchronize()
                    start = time.perf_counter()
                    fresh = reference.prefill(ids)
                    if fresh.first_token != history[0]:
                        raise AssertionError("Fresh official prefill changed its first token")
                    row = decode_trial(reference, decoder, fresh, history[:32], mode, preserve=False)
                    synchronize()
                    row.update(e2e_ms=(time.perf_counter()-start)*1000,
                        prompt_tokens=length, output_tokens=32, repeat=repeat, order=order,
                        prefill_total_ms=fresh.timings["official_prefill_total_ms"])
                    if repeat >= 0:
                        report["e2e_trials"].append(row)
                    del fresh
            print(f"E2E prompt={length} complete", flush=True)
        report["all_blocks_reclaimed"] = not decoder._live and len(decoder.blocks.free_block_ids) == decoder.num_blocks
        report["decode_summary"] = summarize(report["decode_trials"], ["prompt_tokens", "output_tokens"], "decode_ms", MODES[1:])
        report["e2e_summary"] = summarize(report["e2e_trials"], ["prompt_tokens"], "e2e_ms", ["paged_graph"])
        report["gpu_after"] = telemetry()
    finally:
        decoder.close()
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2, ensure_ascii=False)+"\n")
    write_summary(report, destination.with_suffix(".md"))
    print("Report:", destination, flush=True)


if __name__ == "__main__":
    main()

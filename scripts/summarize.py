"""Summarize measured JSON reports without inventing missing runs."""
import argparse
import json
from c2c_decode.reporting import write_reports
from pathlib import Path
from statistics import median

p = argparse.ArgumentParser()
p.add_argument("report")
p.add_argument("--baseline")
p.add_argument("--output", default="results/summary.md")
args = p.parse_args()
data = json.loads(Path(args.report).read_text())
rows = data["benchmark"]["runs"]
baseline = {}
if args.baseline:
    for r in json.loads(Path(args.baseline).read_text())["baseline"]["runs"]:
        baseline.setdefault(r["prompt_tokens"], []).append(r)
lines = ["# Measured performance", "", f"GPU: {data['metadata']['gpu']}", "",
         "Medians over recorded repetitions; fixed output counts; warmed kernels.", "",
         "| Prompt nominal | Decode batch | Decode tok/s | E2E tok/s | TTFT p50 ms | Peak allocated GiB |",
         "|---:|---:|---:|---:|---:|---:|"]
for length, concurrency in sorted({(r["nominal_prompt_tokens"], r["concurrency"]) for r in rows}):
    group = [r for r in rows if r["nominal_prompt_tokens"] == length and r["concurrency"] == concurrency]
    ttft = [q["timings"]["ttft_ms"] for r in group for q in r["requests"]]
    lines.append(f"| {length} | {concurrency} | {median(r['stats']['decode_tokens_per_s'] for r in group):.1f} | "
                 f"{median(r['stats']['e2e_tokens_per_s'] for r in group):.1f} | {median(ttft):.1f} | "
                 f"{max(r['peak_allocated_bytes'] for r in group)/2**30:.2f} |")
if baseline:
    lines += ["", "Official C2C cached decode baseline (batch=1; no second receiver or paged pool):", "",
              "| Prompt | Decode tok/s | E2E tok/s | Peak allocated GiB |", "|---:|---:|---:|---:|"]
    for length, group in sorted(baseline.items()):
        lines.append(f"| {length} | {median(r['decode_tokens_per_s'] for r in group):.1f} | "
                     f"{median(r['e2e_tokens_per_s'] for r in group):.1f} | "
                     f"{max(r['peak_allocated_bytes'] for r in group)/2**30:.2f} |")
lines += ["", "TTFT includes FIFO queueing and cache import. Prefill batch is always one.",
          "Decode throughput excludes each request's first token; E2E includes all phases.",
          "Peak allocated is PyTorch memory, not total device usage. No claim of production serving performance.", ""]
write_reports(Path(args.output), "\n".join(lines))
print("\n".join(lines))

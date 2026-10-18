"""Generate measured tables and figures; never derive speedups from profiled runs."""
import gzip
import hashlib
import json
from c2c_decode.reporting import write_reports
from pathlib import Path
from statistics import median
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path("results/profiling")


def read(name):
    return json.loads((ROOT/name).read_text())


def med(rows, key):
    return median(r[key] for r in rows)


def table(lines, headers, rows):
    lines.extend(["| " + " | ".join(headers) + " |", "|" + "---|"*len(headers)])
    lines.extend("| " + " | ".join(map(str, row)) + " |" for row in rows)
    lines.append("")


def main():
    ab, memory, scheduling = [read(n) for n in ("import-ablation.json", "memory.json", "scheduling.json")]
    profiles = {name: read(name+"/report.json") for name in ("nano-graph", "nano-eager")}
    assert ab["token_equivalence"] and len(ab["rows"]) == 80
    assert len(scheduling["runs"]) == 18
    for run in scheduling["runs"]:
        assert run["stats"]["free_blocks_after"] == run["stats"]["total_blocks"]
        assert len(run["requests"]) == 24
        for r in run["requests"]:
            t = r["timings"]
            assert len(t["token_times_ms"]) == len(r["token_ids"])
            assert t["queue_ms"] >= 0 and t["ttft_ms"] >= t["queue_ms"]
    lines = ["# Profiling measurements", "", "RTX 4090 / BF16 / pinned Qwen3 C2C pair. See docs/profiling-results.md for interpretation.", "",
        "## KV import: paired, randomized order, profiler disabled", "",
        "Five pairs per case; 32 output tokens/request; mixed lengths N - 17*i; medians in milliseconds.", ""]
    rows, import_values = [], {}
    for s in ab["summaries"]:
        groups = {mode: [r for r in ab["rows"] if (r["length"], r["batch"], r["mode"]) ==
            (s["length"], s["batch"], mode)] for mode in ("loop", "scatter")}
        old, new = groups["loop"], groups["scatter"]
        import_values[s["length"], s["batch"]] = groups
        rows.append([s["length"], s["batch"], f'{med(old,"import_ms"):.3f}',
            f'{med(new,"import_ms"):.3f}', f'{s["paired_import_speedup"]:.2f}x',
            f'{med(old,"handoff_decode_ms"):.2f}', f'{med(new,"handoff_decode_ms"):.2f}',
            f'{100*(1-med(new,"handoff_decode_ms")/med(old,"handoff_decode_ms")):.1f}%',
            f'{med(new,"effective_payload_GB_s"):.1f}'])
    table(lines, ["Input", "Batch", "Loop import", "Scatter import", "Paired speedup", "Loop handoff+decode",
        "Scatter handoff+decode", "Latency reduction", "Payload GB/s"], rows)
    lines += ["All 40 pairs produce identical tokens. Payload GB/s counts each KV byte once, not DRAM bandwidth.", "",
        "## Host stages and CUDA trace", "", "Five profiler-disabled trials with removable host timers; 16 output tokens (15 decode steps). Nested stages must not be summed.", ""]
    rows, runtime_values = [], {}
    for name, report in profiles.items():
        for case in report["cases"]:
            decode = median((r["handoff_decode_ms"]-r["import_ms"])/15 for r in case["unprofiled_trials"])
            runtime_values[name, case["length"], case["batch"]] = decode
            trace = case["trace"]
            api = {r["name"]: r["count"] for r in trace["runtime"]}
            rows.append([name, case["length"], case["batch"], f'{decode:.3f}', trace["kernels"],
                sum(count for key, count in api.items() if "LaunchKernel" in key), api.get("cudaGraphLaunch", 0),
                f'{trace["observed_gpu_active_ms"]:.2f}', f'{trace["observed_gpu_active_fraction"]*100:.1f}%',
                f'{trace["largest_observed_gpu_gap_ms"]:.3f}'])
    table(lines, ["Engine", "Input", "Batch", "Decode ms/step", "Trace kernels", "Kernel launch API calls", "GraphLaunch calls",
        "Trace GPU union ms", "Trace GPU coverage", "Largest internal GPU gap ms"], rows)
    lines += ["Trace region includes import and decode. Coverage is the union of traced GPU activity divided by profiled region wall time; it is NOT SM utilization or occupancy. Eager capture records shapes and has substantial instrumentation overhead.", ""]
    for name in ("nano-graph",):
        case = profiles[name]["cases"][0]
        lines += [f"### {name}: 1K B1 host durations", ""]
        table(lines, ["Stage", "Calls in 5 runs", "Host ms / decode step"],
            [[k, v["calls"], f'{v["per_decode_step_ms"]:.4f}'] for k, v in case["host_stages"].items()])
        lines += [f"### {name}: largest GPU kernels in trace", ""]
        table(lines, ["Kernel (abbreviated)", "Count", "GPU ms"],
            [[r["name"][:110].replace("|", "/"), r["count"], f'{r["gpu_us"]/1000:.3f}']
             for r in case["trace"]["top_kernels"][:6]])
    lines += ["### Import-only trace", "", "Kernel counts explain launch overhead; trace timings are not performance benchmark values.", ""]
    table(lines, ["Importer", "Kernels", "CUDA launch calls", "Profiled region ms", "GPU union ms"],
        [[mode, t["kernels"], sum(r["count"] for r in t["runtime"] if "LaunchKernel" in r["name"]),
          f'{t["profiled_region_ms"]:.2f}', f'{t["observed_gpu_active_ms"]:.2f}']
         for mode in ("loop", "scatter") for t in [read(f"import-{mode}.summary.json")]])
    lines += ["## Memory lifetime", "", "Single process: official C2C + nano graph engine + 2044 MiB KV pool. Values are PyTorch allocator GiB, not total device memory.", ""]
    records = memory["records"]
    assert any(r["label"] == "model_0_forward_end" for r in records), "Missing direct-forward instrumentation"
    assert any(r["label"].startswith("fuser_") for r in records), "Missing Fuser instrumentation"
    selected = [r for r in records if not r["label"].startswith("fuser_")]
    table(lines, ["Stage", "Input/logits shape", "Allocated GiB", "Reserved GiB", "Cumulative peak GiB"],
        [[r["label"], r.get("length", r.get("logits_shape", "")),
          *[f'{r[k]/2**30:.3f}' for k in ("allocated", "reserved", "peak_allocated")]] for r in selected])
    lines += ["## Scheduled arrivals: empirical latency", "",
        "24 requests/run, 3 repetitions/case, 72 requests per row (432 total). Inputs 128/512/1024/2048; outputs 16/32/64/128; EOS disabled. Local arrival replay, no HTTP. Quantiles pool the three repetitions.", ""]
    table(lines, ["Arrival gap ms", "Prefills/round", "Throughput tok/s", "Queue P99 ms", "TTFT P50/P95/P99 ms", "Request TPOT P50/P95/P99 ms", "Inter-token gap P50/P95/P99 ms"],
        [[s["gap_ms"], s["prefill_quantum"] or "fill", f'{s["median_throughput"]:.1f}', f'{s["queue_ms"]["p99"]:.1f}',
          *[" / ".join(f'{s[k][p]:.2f}' for p in ("p50", "p95", "p99"))
            for k in ("ttft_ms", "request_tpot_ms", "inter_token_gap_ms")]] for s in scheduling["summaries"]])
    lines += ["TPOT = (last token - first token)/(output count - 1) per request. Inter-token gaps pool every adjacent token interval; these are different distributions. Throughput includes arrival idle time. Full request/token timestamps and E2E percentiles remain in scheduling.json.", ""]
    report = read("nano-scatter.json")
    assert report["completed"] and report["numerical_passed"]
    assert all(c["cache_exact"] and c["forced_history_exact"] for c in report["validation"])
    assert all(r["all_blocks_reclaimed"] for r in report["trials"]+report["e2e"])
    lines += ["## Nano scatter audit with isolated producer", "",
        "Five repetitions/case; graph decode, profiler disabled. Handoff excludes official prefill; E2E includes it and local CUDA IPC. Historical measured values are retained.", ""]
    for forced in (False, True):
        lines += [f"### {'Matched history' if forced else 'Free greedy'}: 32 outputs/request", ""]
        rows = []
        for length in (1024, 2048):
            for batch in (1, 2, 4, 8):
                group = [r for r in report["trials"] if r["teacher_forced"] == forced and
                    (r["prompt_tokens"], r["batch"], r["output_tokens"]) == (length, batch, 32)]
                assert len(group) == 5
                if forced:
                    assert all(r["tokens"] == group[0]["tokens"] for r in group)
                rows.append([length, batch, f'{med(group,"handoff_decode_ms"):.2f}', f'{med(group,"decode_tokens_per_s"):.1f}'])
        table(lines, ["Input", "Batch", "Handoff ms", "Decode tok/s"], rows)
    table(lines, ["Fresh E2E input", "Nano ms"],
        [[length, f'{med([r for r in report["e2e"] if r["prompt_tokens"] == length],"e2e_ms"):.2f}']
         for length in (1024, 2048)])
    write_reports((ROOT/"summary.md"), "\n".join(lines)+"\n")

    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), layout="constrained")
    ax = axes[0, 0]
    groups = import_values[2048, 8]
    ax.bar([0, 1], [med(groups[m], "import_ms") for m in ("loop", "scatter")], color=["#8494a7", "#137c66"])
    ax.set(xticks=[0, 1], xticklabels=["Per-page loop", "Layer scatter"], ylabel="Import latency (ms)", title="2K / B8: profiler-off paired import A/B")
    for i, mode in enumerate(("loop", "scatter")):
        ax.text(i, med(groups[mode], "import_ms")+.8, f'{med(groups[mode], "import_ms"):.2f} ms', ha="center")
    ax = axes[0, 1]
    names = ["nano-eager", "nano-graph"]
    values = [runtime_values[n, 1024, 1] for n in names]
    ax.bar(range(2), values, color=["#8494a7", "#137c66"])
    ax.set(xticks=range(2), xticklabels=names, ylabel="Decode step latency (ms)", title="1K / B1: host timers, PyTorch Profiler off")
    for i, value in enumerate(values):
        ax.text(i, value+.2, f'{value:.2f}', ha="center")
    ax = axes[1, 0]
    for field, label in [("allocated", "Live allocated"), ("reserved", "Allocator reserved"), ("peak_allocated", "Cumulative peak")]:
        ax.step([r["elapsed_ms"] for r in records], [r[field]/2**30 for r in records], where="post", label=label)
    ax.set(xlabel="Sample time (ms; includes GC overhead)", ylabel="GiB", title="Memory: 1K, 2K, repeated 2K requests")
    ax.legend(fontsize=8)
    ax = axes[1, 1]
    for quantum, label, marker in [(None, "Fill available slots", "o"), (1, "One prefill per round", "s")]:
        selected = [s for s in scheduling["summaries"] if s["prefill_quantum"] == quantum]
        ax.plot([s["gap_ms"] for s in selected], [s["inter_token_gap_ms"]["p99"] for s in selected], marker=marker, label=label)
    ax.set(xlabel="Arrival gap (ms; 0 = burst)", ylabel="P99 inter-token gap (ms)", title="Serial prefill stalls ongoing decode")
    ax.legend(fontsize=8)
    fig.suptitle("C2C-Decode profiling | RTX 4090 | BF16 | fixed trained C2C Fuser", fontsize=15)
    fig.savefig(ROOT/"overview.png", dpi=170)
    fig.savefig(ROOT/"overview.svg")
    plt.close(fig)

    # A short actual trace excerpt, not a schematic or hardware occupancy chart.
    fig, axes = plt.subplots(1, 1, figsize=(13, 3.5), layout="constrained")
    for ax, name in zip([axes], ("nano-graph",)):
        with gzip.open(ROOT/name/"trace-1024-b1.json.gz", "rt") as stream:
            events = json.load(stream)["traceEvents"]
        root = next(e for e in events if e.get("name") == "C2C/capture" and e.get("ph") == "X")
        lo = root["ts"]
        for lane, predicate, color in [(0, lambda e: e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset"), "#137c66"),
                (1, lambda e: e.get("cat") in ("cuda_runtime", "cuda_driver"), "#3878ba"),
                (2, lambda e: e.get("name", "").startswith("C2C/") and e.get("name") != "C2C/capture", "#c78532")]:
            intervals = [((e["ts"]-lo)/1000, e["dur"]/1000) for e in events if "dur" in e and
                lo <= e.get("ts", 0) < lo+10000 and predicate(e)]
            ax.broken_barh(intervals, (lane-.3, .6), facecolors=color)
        ax.set(xlim=(0, 10), yticks=[0, 1, 2], yticklabels=["GPU activity", "CUDA API", "Host regions"],
            xlabel="Time from profiled handoff start (ms)", title=name+": first 10 ms (instrumented)")
    fig.savefig(ROOT/"timeline.png", dpi=170)
    fig.savefig(ROOT/"timeline.svg")
    plt.close(fig)
    manifest = {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                for base in ("src/c2c_decode", "scripts", "tests") for p in sorted(Path(base).glob("*.py"))}
    (ROOT/"summary-source-sha256.json").write_text(json.dumps(manifest, indent=2)+"\n")
    print("REPORT_COMPLETE")


if __name__ == "__main__":
    main()

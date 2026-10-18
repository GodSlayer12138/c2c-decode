"""Audit experiment completeness and generate reproducible improvement tables/figures."""
import hashlib
import json
from c2c_decode.reporting import write_reports
from pathlib import Path
from statistics import median
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path("results/next")


def read(name):
    return json.loads((ROOT/name).read_text())


def table(lines, headers, rows):
    lines.extend(["| " + " | ".join(headers) + " |", "|" + "---|"*len(headers)])
    lines.extend("| " + " | ".join(map(str, row)) + " |" for row in rows)
    lines.append("")


def med(rows, key):
    return median(r[key] for r in rows)


def percentile(values, q):
    values = sorted(values)
    pos = (len(values)-1)*q
    lo, hi = int(pos), min(int(pos)+1, len(values)-1)
    return values[lo]*(1-(pos-lo))+values[hi]*(pos-lo)


def main():
    producer, inputs, e2e = [read(n) for n in ("producer.json", "cpu-inputs.json", "e2e.json")]
    quality = {name: read(f"quality-{name}.json") for name in ("nano",)}
    load, http, soak = [read(n) for n in ("service-load.json", "http.json", "soak.json")]
    assert len(producer["trials"]) == 50 and len(producer["validation"]) == 24
    assert len(inputs["trials"]) == 30 and len(e2e["trials"]) == 120 and e2e["exact_lean_tokens"]
    assert len(load["runs"]) == 54
    assert all(r["free_blocks_after"] == 73 and r["summary"]["completed"] == 24 for r in load["runs"])
    assert all(r["all_blocks_reclaimed"] for r in inputs["trials"]+e2e["trials"])
    assert all(r["complete"] for r in quality.values())
    assert http["queue_full_status"] == 429 and not http["engine_error"] and http["disconnect_retired"]
    assert http["cancel_stream"]["reason"] == "cancelled" and http["timeout_stream"]["reason"] == "timeout"
    assert http["free_blocks_after"] == soak["free_blocks_after"] == 73 and soak["reasons"]["error"] == 0
    lines = ["# Follow-up optimization measurements", "", "RTX 4090, BF16, unchanged pinned Qwen3 pair and trained C2C Fuser. All speed measurements have PyTorch Profiler disabled.", "",
        "## Producer: randomized mode order, 5 repetitions", "", "Memory peaks below include official producer models/Fuser only, without nano receiver or its KV pool.", ""]
    modes = ["official", "lean", "chunk-256", "chunk-512", "chunk-1024"]
    table(lines, ["Input", "Mode", "Median prefill ms", "Peak allocated GiB", "Validation cases", "Exact KV cases"],
        [[length, mode, f'{med([r for r in producer["trials"] if r["length"]==length and r["mode"]==mode],"ms"):.2f}',
          f'{next(r["peak"] for r in producer["memory"] if r["length"]==length and r["mode"]==mode)/2**30:.3f}',
          sum(r["numerical_passed"] for r in producer["validation"] if r["mode"]==mode),
          sum(r["cache"]["exact"] for r in producer["validation"] if r["mode"]==mode)]
         for length in (1024, 2048) for mode in modes])
    lines += ["Validation counts pool lengths 31/255/256/257/1024/2048; official is the comparator, not a test case. Chunk validation above covers handoff logits, not all continuation steps.", "",
        "## Input preparation: paired trials", ""]
    rows = []
    for length, batch in ((257, 4), (1024, 1), (2048, 8)):
        groups = [[r for r in inputs["trials"] if (r["length"], r["batch"], r["fast_inputs"]) == (length,batch,fast)] for fast in (False,True)]
        deltas = [next(r["handoff_decode_ms"] for r in groups[0] if r["repeat"]==i)-next(r["handoff_decode_ms"] for r in groups[1] if r["repeat"]==i) for i in range(5)]
        rows.append([length, batch, *[f'{med(g,"handoff_decode_ms"):.3f}' for g in groups],
            f'{median(deltas):.3f}', f'{min(deltas):.3f} .. {max(deltas):.3f}', sum(x>0 for x in deltas)])
    table(lines, ["Input", "Batch", "Original handoff ms", "Buffers handoff ms", "Paired saved ms", "Min..max saved", "Faster pairs / 5"], rows)
    lines += ["Token histories match within all 15 input-buffer pairs. This small gain remains opt-in.", "",
        "## Fresh-request end to end", "", "Includes fresh serial prefills, fusion, import and decode. One process, no IPC/HTTP. Five repetitions, randomized mode order. Batch>1 lengths are N-17*i.", ""]
    rows = []
    for length in (1024,2048):
        for batch, count in ((1,32),(1,128),(8,32)):
            groups = [[r for r in e2e["trials"] if (r["length"],r["batch"],r["count"],r["mode"],r["fast_inputs"]) ==
                (length,batch,count,mode,fast)] for mode,fast in (("official",False),("lean",False),("lean",True),("chunk-512",True))]
            assert all(len(g)==5 for g in groups)
            rows.append([length,batch,count,*[f'{med(g,"e2e_ms"):.2f}' for g in groups],
                f'{100*(1-med(groups[1],"e2e_ms")/med(groups[0],"e2e_ms")):.2f}%'])
    table(lines, ["Input","Batch","Output/request","Official ms","Lean ms","Lean+buffers ms","Chunk512+buffers ms","Lean reduction"],rows)
    lines += ["All official/lean/lean+buffer token histories match in this E2E matrix. Chunk is an experimental scheduling tradeoff and does not preserve bitwise KV.", "",
        "## Authored task quality and logit diagnostics", "", "36 authored cases: 10 arithmetic, 10 QA, 10 Python-output reasoning, 6 context retrieval. Strict normalized answer exact match also penalizes extra explanation. It is not semantic accuracy or a public benchmark. Generation is capped at 64 tokens.", ""]
    rows, gates = [], []
    for name, report in quality.items():
        for mode in ["official","lean","chunk-512"]:
            group = [r for r in report["rows"] if r["mode"]==mode]
            assert len(group)==36 and all(r["cache_exact"] and r["all_blocks_reclaimed"] for r in group)
            reference_rows = {r["id"]:r for r in quality["nano"]["rows"] if r["mode"]=="official"}
            assert all(r["official_tokens"]==reference_rows[r["id"]]["official_tokens"] for r in group)
            rows.append([name,mode,sum(r["official_correct"] for r in group),sum(r["correct"] for r in group),
                sum(r["exact_tokens"] for r in group),sum(r["truncated"] for r in group),
                sum(r["text"]!=reference_rows[r["id"]]["text"] for r in group)])
            for r in group:
                for i,s in enumerate(r["steps"]):
                    if s["cosine"]<.999 or s["rmse"]>.15 or s["max_abs"]>1:
                        gates.append(dict(backend=name,mode=mode,id=r["id"],step=i+1,
                            before_or_at_eos=i<len(r["official_tokens"]),**s))
    table(lines, ["Engine","Producer","Official exact answers / 36","Engine exact answers / 36","Exact sequences / 36","Truncated / 36","Changed text vs old nano / 36"], rows)
    lines += ["Numerical tolerances remain cosine>=0.999, RMSE<=0.15, max_abs<=1. The following failures are retained, not reclassified as passes. Teacher forcing also continues after EOS; before/at-EOS failures are distinguished explicitly.", ""]
    table(lines,["Engine","Producer","Failing checked steps","Failing steps before/at EOS"],
        [[name, mode, sum(g["backend"]==name and g["mode"]==mode for g in gates),
          sum(g["backend"]==name and g["mode"]==mode and g["before_or_at_eos"] for g in gates)]
         for name in quality for mode in ["official","lean","chunk-512"]])
    targeted = read("quality-nano-chunk1024-context.json")
    assert targeted["complete"] and len(targeted["rows"]) == 18
    context_rows = [r for r in targeted["rows"] if r["mode"] == "chunk-1024"]
    extra = []
    for r in context_rows:
        baseline = next(x for x in quality["nano"]["rows"] if x["id"]==r["id"] and x["mode"]=="official")
        assert r["official_tokens"] == baseline["official_tokens"] and r["cache_exact"] and r["all_blocks_reclaimed"]
        for i,s in enumerate(r["steps"]):
            if s["cosine"]<.999 or s["rmse"]>.15 or s["max_abs"]>1:
                extra.append(dict(backend="nano",mode="chunk-1024",id=r["id"],step=i+1,
                    before_or_at_eos=i<len(r["official_tokens"]),**s))
    lines += ["### Targeted 1024-token chunk follow-up", "", "Six long-context cases only, after observing the 512-token chunk failure; not a new independent full benchmark.", ""]
    table(lines,["Cases","Exact sequences vs official","Checked steps","Numerical failures","Before/at EOS failures"],
        [[len(context_rows),sum(r["exact_tokens"] for r in context_rows),sum(len(r["steps"]) for r in context_rows),len(extra),sum(g["before_or_at_eos"] for g in extra)]])
    gates.extend(extra)
    (ROOT/"numerical-failures.json").write_text(json.dumps(gates,indent=2)+"\n")
    lines += ["## Scheduled arrival replay", "", "Three fixed seeds, 24 requests/run, exponential arrival intervals (or burst), six modes: 1296 requests total. Each mode reuses the exact same arrival/length/output schedule per seed. Quantiles pool 72 requests per row; still small empirical tails.", ""]
    summaries = []
    for gap in (0,100,350):
        for mode in ("official","lean","lean-fast","chunk-256","chunk-512","chunk-1024"):
            runs = [r for r in load["runs"] if r["gap_ms"]==gap and r["mode"]==mode]
            assert len(runs)==3
            for run in runs:
                baseline = next(r for r in load["runs"] if r["gap_ms"]==gap and r["mode"]=="official" and r["repeat"]==run["repeat"])
                assert run["arrivals"]==baseline["arrivals"]
            jobs = [j for run in runs for j in run["jobs"]]
            intervals = [b-a for j in jobs for a,b in zip(j["token_times_ms"],j["token_times_ms"][1:])]
            s = dict(gap_ms=gap,mode=mode,tokens_per_s=median(r["summary"]["tokens_per_s"] for r in runs),
                ttft_p99=percentile([j["token_times_ms"][0] for j in jobs],.99),
                interval_p99=percentile(intervals,.99),interval_max=max(intervals),
                tpot_p95=percentile([(j["token_times_ms"][-1]-j["token_times_ms"][0])/(len(j["token_times_ms"])-1) for j in jobs],.95),
                slo_good=sum(r["summary"]["slo"]["good_requests"] for r in runs))
            summaries.append(s)
    table(lines,["Mean arrival gap ms","Mode","Median tok/s","TTFT P99 ms","Token gap P99 ms","Max token gap ms","Request TPOT P95 ms","SLO good / 72"],
        [[s["gap_ms"],s["mode"],*[f'{s[k]:.2f}' for k in ("tokens_per_s","ttft_p99","interval_p99","interval_max","tpot_p95")],s["slo_good"]] for s in summaries])
    lines += ["Illustrative SLO fixed before the runs: TTFT<=1000ms AND every token gap<=100ms. No mode guarantees this SLO. This table measures engine arrival replay, not HTTP latency.", "",
        "## HTTP lifecycle and soak", ""]
    table(lines,["Check","Observed"],[["Queue overflow HTTP status",http["queue_full_status"]],
        ["DELETE cancellation",http["cancel_stream"]["reason"]],["Deadline",http["timeout_stream"]["reason"]],
        ["Request elapsed at cancellation ms",f'{http["cancel_stream"]["elapsed_ms"]:.2f}'],
        ["1ms deadline event latency ms",f'{http["timeout_stream"]["elapsed_ms"]:.2f}'],
        ["Disconnected request retired",http["disconnect_retired"]],["Concurrent calls",len(http["concurrent"])],
        ["Soak seconds",f'{soak["duration_s"]:.1f}'],["Offered / rejected",f'{soak["offered"]} / {soak["rejected"]}'],
        ["Completion reasons",soak["reasons"]],["Final free pages",soak["free_blocks_after"]],
        ["Allocated before / after GiB",f'{soak["baseline_allocated"]/2**30:.4f} / {soak["final_allocated"]/2**30:.4f}']])
    lines += ["Cancellation/timeout are cooperative between GPU work units. The two-minute soak is a bounded stability experiment, not a long-term leak proof or production SLA.", ""]
    write_reports((ROOT/"summary.md"), "\n".join(lines)+"\n")
    (ROOT/"load-summary.json").write_text(json.dumps(summaries,indent=2)+"\n")
    plt.rcParams.update({"font.size":10,"axes.spines.top":False,"axes.spines.right":False})
    fig, axes = plt.subplots(2,2,figsize=(13,8),layout="constrained")
    ax=axes[0,0]
    for offset, mode in ((-.18,"official"),(.18,"lean")):
        vals=[med([r for r in producer["trials"] if r["length"]==n and r["mode"]==mode],"ms") for n in (1024,2048)]
        bars=ax.bar([i+offset for i in range(2)],vals,.36,label=mode)
        ax.bar_label(bars,fmt="%.1f",padding=3)
    ax.set(xticks=[0,1],xticklabels=["1K","2K"],ylabel="Producer wall time (ms)",title="Compute only needed logits")
    ax.legend()
    ax=axes[0,1]
    vals=[next(r["peak"] for r in producer["memory"] if r["length"]==2048 and r["mode"]==m)/2**30 for m in ("official","lean")]
    bars=ax.bar([0,1],vals,color=["#8192a7","#147f6a"])
    ax.bar_label(bars,fmt="%.2f",padding=3)
    ax.set(xticks=[0,1],xticklabels=["Official","Lean"],ylabel="Peak allocated GiB",title="2K producer only (no decode engine)")
    ax=axes[1,0]
    for mode in ("official","lean","chunk-256","chunk-512","chunk-1024"):
        selected=[s for s in summaries if s["mode"]==mode]
        ax.plot([s["gap_ms"] for s in selected],[s["interval_p99"] for s in selected],marker="o",label=mode)
    ax.set(xlabel="Mean arrival interval (ms)",ylabel="P99 inter-token gap (ms)",title="Chunking changes the latency tradeoff")
    ax.legend(fontsize=8)
    ax=axes[1,1]
    selected=[s for s in summaries if s["gap_ms"]==0 and s["mode"]!="lean-fast"]
    for s in selected:
        ax.scatter(s["tokens_per_s"],s["ttft_p99"],s=45)
        offset={"lean":(-25,-18),"official":(4,12),"chunk-1024":(-62,12)}.get(s["mode"],(4,6))
        ax.annotate(s["mode"],(s["tokens_per_s"],s["ttft_p99"]),fontsize=8,xytext=offset,textcoords="offset points")
    ax.margins(x=.12,y=.12)
    ax.set(xlabel="Aggregate throughput (tokens/s)",ylabel="P99 TTFT (ms)",title="Burst: lower token gaps can cost throughput")
    fig.suptitle("C2C-Decode follow-up experiments | RTX 4090 | BF16",fontsize=15)
    fig.savefig(ROOT/"overview.png",dpi=170)
    fig.savefig(ROOT/"overview.svg")
    plt.close(fig)
    manifest={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for folder in ("src/c2c_decode","scripts","tests") for p in sorted(Path(folder).glob("*.py"))}
    (ROOT/"summary-source-sha256.json").write_text(json.dumps(manifest,indent=2)+"\n")
    print("SUMMARY_COMPLETE")


if __name__=="__main__":
    main()

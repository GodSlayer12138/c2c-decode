"""Trace extraction and removable instrumentation, outside production code."""
from collections import defaultdict
from contextlib import nullcontext
import gzip
import json
from pathlib import Path
import time
import torch


class Hooks:
    def __init__(self):
        self.originals, self.samples = [], defaultdict(list)
        self.annotate = False

    def wrap(self, obj, method, name):
        original = getattr(obj, method)
        def measured(*args, **kwargs):
            start = time.perf_counter()
            with torch.profiler.record_function("C2C/"+name) if self.annotate else nullcontext():
                result = original(*args, **kwargs)
            self.samples[name].append((time.perf_counter()-start)*1000)
            return result
        self.originals.append((obj, method, original))
        setattr(obj, method, measured)

    def close(self):
        for obj, method, original in reversed(self.originals):
            setattr(obj, method, original)


def interval_union(intervals):
    result = []
    for start, end in sorted(intervals):
        if result and start <= result[-1][1]:
            result[-1][1] = max(result[-1][1], end)
        else:
            result.append([start, end])
    return result


def trace_summary(trace):
    events = trace["traceEvents"]
    roots = [e for e in events if e.get("name") == "C2C/capture" and e.get("ph") == "X"]
    root = roots[0]
    lo, hi = root["ts"], root["ts"]+root["dur"]
    gpu = [e for e in events if e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset")
           and "dur" in e and lo <= e["ts"] < hi]
    kernels = [e for e in gpu if e["cat"] == "kernel"]
    merged = interval_union([(e["ts"], min(hi, e["ts"]+e["dur"])) for e in gpu])
    gaps = [b[0]-a[1] for a, b in zip(merged, merged[1:])]
    names = defaultdict(lambda: dict(count=0, gpu_us=0.))
    for e in kernels:
        names[e["name"]]["count"] += 1
        names[e["name"]]["gpu_us"] += e["dur"]
    runtime = [e for e in events if e.get("cat") in ("cuda_runtime", "cuda_driver")
               and "dur" in e and lo <= e["ts"] < hi]
    grouped = defaultdict(lambda: dict(count=0, host_us=0.))
    for e in runtime:
        grouped[e["name"]]["count"] += 1
        grouped[e["name"]]["host_us"] += e["dur"]
    return dict(profiled_region_ms=root["dur"]/1000, gpu_events=len(gpu), kernels=len(kernels),
        observed_gpu_active_ms=sum(b-a for a, b in merged)/1000,
        observed_gpu_active_fraction=sum(b-a for a, b in merged)/root["dur"],
        largest_observed_gpu_gap_ms=max(gaps, default=0)/1000,
        top_kernels=sorted([dict(name=k, **v) for k, v in names.items()], key=lambda r: -r["gpu_us"]),
        runtime=sorted([dict(name=k, **v) for k, v in grouped.items()], key=lambda r: -r["host_us"]))


def capture(function, destination, hooks=None, shapes=False):
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if hooks:
        hooks.annotate = True
    try:
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA], record_shapes=shapes) as profile:
            with torch.profiler.record_function("C2C/capture"):
                function()
                torch.cuda.synchronize()
        profile.export_chrome_trace(str(destination))
        trace = json.loads(destination.read_text())
        summary = trace_summary(trace)
        operators = [dict(name=e.key, count=e.count, self_cpu_us=e.self_cpu_time_total,
            cpu_us=e.cpu_time_total, self_gpu_us=e.self_device_time_total,
            gpu_us=e.device_time_total, shapes=str(e.input_shapes))
            for e in profile.key_averages(group_by_input_shape=shapes)]
        destination.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2))
        destination.with_suffix(".operators.json").write_text(json.dumps(operators, indent=2))
        with gzip.open(str(destination)+".gz", "wt") as stream:
            json.dump(trace, stream)
        destination.unlink()  # This exact file was created above; retain compressed trace.
        return summary
    finally:
        if hooks:
            hooks.annotate = False

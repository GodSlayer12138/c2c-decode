"""Real-engine CPU/CUDA traces and low-overhead host-stage timing."""
import argparse
import gc
import json
from pathlib import Path
from statistics import median
import time
import torch
from c2c_decode.bridge import OfficialBridge
from c2c_decode.nano_benchmark import Nano
from profiling_utils import Hooks, capture


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--models", default="models.lock.json")
    p.add_argument("--eager", action="store_true")
    p.add_argument("--import-mode", default="scatter")
    p.add_argument("--output-dir", required=True)
    args = p.parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    bridge = OfficialBridge("/root/autodl-tmp/envs/c2c-decode/bin/python", args.models)
    backend = Nano(args)
    hooks = Hooks()
    for method, name in [("import_cache", "import"), ("prepare_decode", "prepare_inputs"),
                         ("run_model", "model_call")]:
        hooks.wrap(backend.decoder, method, name)
    hooks.wrap(backend.decoder.model, "compute_logits", "vocabulary_projection")
    report = dict(args=vars(args), backend=backend.info, cases=[])
    try:
        for length, batch in ([(1024, 1)] if args.eager else [(1024, 1), (2048, 8)]):
            handoffs = [bridge.prefill(length=length-i*17) for i in range(batch)]
            def run():
                return backend.run(handoffs, [[]]*batch, 16)
            run()
            hooks.samples.clear()
            trials = [run()[0] for _ in range(5)]
            stages = {name: dict(calls=len(values), total_ms=sum(values),
                median_ms=median(values), per_decode_step_ms=sum(values)/(5*15))
                for name, values in hooks.samples.items()}
            trace = capture(run, out/f"trace-{length}-b{batch}.json", hooks, shapes=args.eager)
            report["cases"].append(dict(length=length, batch=batch, output_tokens=16,
                unprofiled_trials=trials, host_stages=stages, trace=trace))
            keys = [h.key for h in handoffs]
            del handoffs
            gc.collect()
            for key in keys:
                bridge.release(key)
            (out/"report.json").write_text(json.dumps(report, indent=2))
            print("PROFILE_COMPLETE", "nano", length, batch, flush=True)
    finally:
        hooks.close()
        backend.close()
        bridge.close()


if __name__ == "__main__":
    main()

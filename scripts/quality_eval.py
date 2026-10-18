"""Reference/free-greedy task scores plus matched-history logit checks, through real engines."""
import argparse
import gc
import json
from pathlib import Path
import torch
from transformers import AutoTokenizer
from c2c_decode.bridge import OfficialBridge
from c2c_decode.nano_benchmark import Nano
from c2c_decode.paths import model_paths
from quality_cases import cases, score


def trim(tokens, eos):
    return tokens[:next((i for i, t in enumerate(tokens) if t in eos), len(tokens))]


def compare(a, b):
    a, b = a.float().flatten(), b.float().flatten()
    top = b.topk(2).values
    return dict(rmse=(a-b).square().mean().sqrt().item(), max_abs=(a-b).abs().max().item(),
        cosine=torch.nn.functional.cosine_similarity(a, b, dim=0).item(),
        top1_match=a.argmax().item() == b.argmax().item(), margin=(top[0]-top[1]).item())


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--chunk-size", type=int, choices=[256,512,1024], default=512)
    p.add_argument("--context-only", action="store_true")
    p.add_argument("--output")
    args = p.parse_args()
    args.models, args.eager = "models.lock.json", False
    bridge = OfficialBridge("/root/autodl-tmp/envs/c2c-decode/bin/python")
    backend = Nano(args)
    tokenizer = AutoTokenizer.from_pretrained(model_paths()["receiver"], local_files_only=True)
    config = json.loads((Path(model_paths()["receiver"])/"generation_config.json").read_text())
    eos = config["eos_token_id"]
    eos = set(eos if isinstance(eos, list) else [eos])
    backend.decoder.fast_inputs = True
    selected = [c for c in cases() if not args.context_only or c["category"] == "context"]
    report = dict(backend="nano", cases=selected, rows=[], metadata=bridge.info,
        note="36 authored functional cases, strict normalized answer scoring. Code-output reasoning only; no generated code execution. Not a public benchmark.")
    output = Path(args.output or "results/next/quality-nano.json")
    try:
        for case in report["cases"]:
            reference_tokens, expected_logits = None, None
            for mode in ["official", "lean", f"chunk-{args.chunk_size}"]:
                h = bridge.prefill(prompt=case["prompt"], mode=mode)
                if reference_tokens is None:
                    ref = bridge.call("history", key=h.key, count=64, check_steps=15)
                    reference_tokens = ref["history"]
                    expected_logits = [x.cpu() for x in ref["logits"]]
                    del ref
                free, _ = backend.run([h], [[]], 64)
                forced, diagnostic = backend.run([h], [reference_tokens], 16, force=True, check=True)
                assert forced["tokens"][0] == reference_tokens[:16]
                errors = [compare(a.cpu(), b) for a, b in zip(diagnostic[0]["logits"], expected_logits)]
                actual = trim(free["tokens"][0], eos)
                official = trim(reference_tokens, eos)
                text, official_text = [tokenizer.decode(t, skip_special_tokens=True) for t in (actual, official)]
                report["rows"].append(dict(id=case["id"], category=case["category"], mode=mode,
                    prompt_tokens=len(h.prompt_ids), expected=case["answer"], text=text,
                    official_text=official_text, correct=score(text, case), official_correct=score(official_text, case),
                    tokens=actual, official_tokens=official, exact_tokens=actual == official,
                    truncated=len(actual) == 64, official_truncated=len(official) == 64,
                    steps=errors, cache_exact=diagnostic[0]["cache_exact"],
                    all_blocks_reclaimed=free["all_blocks_reclaimed"]))
                key = h.key
                del h, diagnostic, free, forced
                gc.collect()
                bridge.release(key)
            output.write_text(json.dumps(report, indent=2)+"\n")
            print("QUALITY", "nano", case["id"], flush=True)
        report["complete"] = True
        output.write_text(json.dumps(report, indent=2)+"\n")
    finally:
        backend.close()
        bridge.close()


if __name__ == "__main__":
    main()

"""Small runnable demo; no HTTP server or training dependencies."""
import argparse
from dataclasses import asdict
import json
from .reference import OfficialC2C
from .optimized import OptimizedC2C
from .paged import PagedDecoder
from .scheduler import Request, Scheduler


def main():
    p = argparse.ArgumentParser(description="Run prompts through official C2C + paged decode")
    p.add_argument("--prompt", action="append", required=True, help="Repeat for multiple requests")
    p.add_argument("--models", default="models.lock.json")
    p.add_argument("--max-new-tokens", type=int, default=64)
    p.add_argument("--pool-mib", type=int, default=2048)
    p.add_argument("--batch", type=int, choices=[1, 2, 4, 8], default=8)
    p.add_argument("--graph", action="store_true")
    p.add_argument("--official-prefill", action="store_true", help="Keep all original prefix logits for the reference path")
    args = p.parse_args()
    reference = (OfficialC2C if args.official_prefill else OptimizedC2C)(args.models)
    decoder = PagedDecoder(args.models, args.pool_mib, max_batch=args.batch, graph=args.graph)
    try:
        scheduler = Scheduler(reference, decoder)
        requests = [Request(str(i), reference.encode(prompt), args.max_new_tokens)
                    for i, prompt in enumerate(args.prompt)]
        for result in scheduler.run(requests):
            print(json.dumps(dict(asdict(result), text=reference.tokenizer.decode(
                result.token_ids, skip_special_tokens=True)), ensure_ascii=False))
        print(json.dumps(scheduler.last_stats))
    finally:
        decoder.close()


if __name__ == "__main__":
    main()

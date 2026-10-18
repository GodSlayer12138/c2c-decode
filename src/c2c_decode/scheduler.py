"""FIFO admission, serial official prefill, and continuous batched decode."""
from collections import deque
from dataclasses import dataclass, field
import math
import time
import torch


@dataclass
class Request:
    request_id: str
    prompt_ids: list[int]
    max_new_tokens: int = 32
    ignore_eos: bool = False
    arrival_ms: float = 0.


@dataclass
class Result:
    request_id: str
    token_ids: list[int]
    timings: dict
    finish_reason: str


@dataclass
class Active:
    request: Request
    sequence: object
    reservation: int
    start: float
    timings: dict = field(default_factory=dict)


class Scheduler:
    def __init__(self, reference, decoder, prefill_limit_per_round=None):
        if prefill_limit_per_round is not None and prefill_limit_per_round < 1:
            raise ValueError("prefill_limit_per_round must be positive")
        self.reference, self.decoder = reference, decoder
        self.prefill_limit_per_round = prefill_limit_per_round
        self.last_stats = {}

    @torch.inference_mode()
    def run(self, requests):
        """Replay scheduled arrivals; TTFT includes queueing and earlier prefills.

        Reserve each admitted request's full maximum block demand on the CPU,
        but physically allocate blocks only as its KV grows. This prevents OOM
        deadlocks without eviction or recomputing the expensive C2C prefix.
        """
        requests = list(requests)
        if len({r.request_id for r in requests}) != len(requests):
            raise ValueError("Request IDs must be unique")
        d = self.decoder
        for r in requests:
            if not math.isfinite(r.arrival_ms) or r.arrival_ms < 0:
                raise ValueError("arrival_ms must be finite and nonnegative")
            if r.max_new_tokens < 1 or len(r.prompt_ids) < 2:
                raise ValueError("Need at least two prompt tokens and one output token")
            if len(r.prompt_ids) + r.max_new_tokens > d.config.max_model_len:
                raise ValueError("Prompt + output exceeds the total length limit")
            if math.ceil((len(r.prompt_ids)+r.max_new_tokens-1)/d.block_size) > d.num_blocks:
                raise ValueError("One request exceeds the KV pool")
        if d._live:
            raise RuntimeError("Scheduler requires an idle decoder")
        pending = deque(sorted(requests, key=lambda r: r.arrival_ms))
        active, results = [], {}
        reserved, high_water, decode_steps, decoded_tokens = 0, 0, 0, 0
        decode_ms, import_ms = 0., 0.
        torch.cuda.synchronize()
        arrival = time.perf_counter()

        def finish(state, reason):
            nonlocal reserved
            state.timings["e2e_ms"] = (time.perf_counter()-arrival)*1000-state.request.arrival_ms
            tokens = list(state.sequence.completion_token_ids)
            # The pinned official generate replaces its terminal EOS with pad.
            pad = getattr(self.reference, "pad_token_id", None)
            if reason == "eos" and pad is not None:
                tokens[-1] = pad
            results[state.request.request_id] = Result(state.request.request_id,
                tokens, state.timings, reason)
            d.release(state.sequence)
            reserved -= state.reservation

        try:
            while pending or active:
                admitted = 0
                while pending and len(active) < d.config.max_num_seqs:
                    r = pending[0]
                    if arrival+r.arrival_ms/1000 > time.perf_counter():
                        break
                    if self.prefill_limit_per_round is not None and admitted >= self.prefill_limit_per_round:
                        break
                    demand = math.ceil((len(r.prompt_ids)+r.max_new_tokens-1)/d.block_size)
                    if reserved + demand > d.num_blocks:
                        break
                    pending.popleft()
                    admitted += 1
                    started = time.perf_counter()
                    handoff = self.reference.prefill(r.prompt_ids)
                    t = time.perf_counter()
                    seq = d.import_cache(handoff)
                    torch.cuda.synchronize()
                    elapsed = (time.perf_counter()-t)*1000
                    ready = (time.perf_counter()-arrival)*1000-r.arrival_ms
                    timings = dict(handoff.timings, cache_import_ms=elapsed,
                                   queue_ms=(started-arrival)*1000-r.arrival_ms,
                                   scheduled_arrival_ms=r.arrival_ms, ttft_ms=ready,
                                   token_times_ms=[ready])
                    import_ms += elapsed
                    del handoff
                    reserved += demand
                    high_water = max(high_water, len(d.blocks.used_block_ids))
                    state = Active(r, seq, demand, started, timings)
                    if not r.ignore_eos and seq.last_token in self.reference.eos_ids:
                        finish(state, "eos")
                    elif r.max_new_tokens == 1:
                        finish(state, "length")
                    else:
                        active.append(state)
                if not active:
                    if pending:
                        if admitted:
                            continue
                        wait_s = (arrival+pending[0].arrival_ms/1000)-time.perf_counter()
                        if wait_s > 0:
                            time.sleep(min(wait_s, .05))
                            continue
                        raise RuntimeError("Admission made no progress")
                    break
                t = time.perf_counter()
                logits = d.step([s.sequence for s in active])
                next_tokens = logits.argmax(-1).tolist()
                torch.cuda.synchronize()
                step_ms = (time.perf_counter()-t)*1000
                decode_ms += step_ms
                decode_steps += 1
                decoded_tokens += len(active)
                high_water = max(high_water, len(d.blocks.used_block_ids))
                continuing = []
                for state, token in zip(active, next_tokens):
                    seq, r = state.sequence, state.request
                    seq.append_token(token)
                    state.timings["token_times_ms"].append((time.perf_counter()-arrival)*1000-r.arrival_ms)
                    if not r.ignore_eos and token in self.reference.eos_ids:
                        finish(state, "eos")
                    elif seq.num_completion_tokens >= r.max_new_tokens:
                        finish(state, "length")
                    else:
                        continuing.append(state)
                active = continuing
                del logits
        finally:
            for seq in list(d._live.values()):
                d.release(seq)
        wall = time.perf_counter()-arrival
        total_tokens = sum(len(r.token_ids) for r in results.values())
        self.last_stats = {"e2e_ms": wall*1000, "decode_ms": decode_ms,
            "cache_import_ms": import_ms, "decode_steps": decode_steps,
            "decode_tokens": decoded_tokens, "output_tokens": total_tokens,
            "decode_tokens_per_s": decoded_tokens/(decode_ms/1000) if decode_ms else 0.,
            "e2e_tokens_per_s": total_tokens/wall if wall else 0.,
            "pool_bytes": d.pool_bytes, "peak_used_blocks": high_water,
            "free_blocks_after": len(d.blocks.free_block_ids), "total_blocks": d.num_blocks}
        return [results[r.request_id] for r in requests]

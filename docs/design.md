# Design

[中文版](design.zh-CN.md)

This project adapts the existing Cache-to-Cache method to a small paged decode
backend. Official Rosetta code owns all
prefill and fusion math; nano-vLLM owns the Qwen3 model, weights loader, RoPE,
FlashAttention decode, block allocator and optional CUDA graphs.

## Handoff invariant

For N input tokens, the official example uses two sections: `[1,0]` for N-1
tokens, then `[-1,0]` for one token. The receiver and sender prefill the first
section. The Fuser modifies receiver KV, and the receiver processes the final
prompt token using that fused state. We call this complete official forward.

At handoff, all layers contain N cached tokens and the final receiver logits
predict generated token t0. Import copies `[1,H,N,D]` into physical pages
`[block,offset,H,D]` without applying RoPE again. t0 is appended to sequence
metadata but is not cached yet. The first nano decode processes t0 at position
N, writes its KV at slot N, attends to N+1 cached entries, and predicts t1.
Repeating t0, replaying the last prompt token, or starting at position N-1
would be incorrect.

The default importer uses a layer-wise indexed scatter from logical positions
to `(physical block, offset)`, including partial and noncontiguous pages.
`PagedDecoder(import_mode="loop")` retains the original per-page copy for A/B
measurement. Both paths write only valid token slots and preserve post-RoPE KV.

The adapter currently accepts unpadded batch=1 prefill. Decode packs one token
per active request with independent positions, block tables and context lengths.
No tokenizer conversion is needed for the chosen matching Qwen3 pair.
The pinned official `generate` replaces a terminal EOS with its pad token;
returned completed sequences follow that convention as well.

## Memory and scheduling

The pool has an explicit MiB budget, independent of free-memory percentage.
For Qwen3-0.6B (28 layers, 8 KV heads, head dimension 128, BF16), a token uses
114,688 bytes, a 256-token block uses 28 MiB. A 1024 MiB budget fits 36 blocks
(1008 MiB actual allocation). Eight 2K prompts plus generation need more than
36 blocks, so the 1/2/4/8 benchmark uses a 2048 MiB budget (73 blocks).

Admission reserves `ceil((prompt+max_new_tokens-1)/256)` blocks for the whole
remaining request. Physical pages are allocated initially for prompt KV and
as decoding crosses boundaries. No output token beyond the requested count
is fed back into the model. Reservations prevent active requests from mutually
exhausting the pool. FIFO waiting requests are admitted as previous requests
finish. The scheduler releases every live sequence on errors.

Requests may specify a nonnegative `arrival_ms` relative to the run's start.
FIFO admission uses arrival order, while results retain caller order. Per-request
`token_times_ms` are relative to that request's scheduled arrival. The default
still admits until capacity is reached; optional `prefill_limit_per_round=1`
interleaves one full prefill with each decode batch. It does not chunk or overlap
prefill. The profiling report measures the resulting latency tradeoff.

Prefix hashing is deliberately disabled: a token-only hash cannot identify a
fused prefix. Future reuse would have to include sender/Fuser revisions, input,
alignment, fusion settings, precision and positional context. GPU work is serialized; serial prefill can stall active
decode. The optional serving loop can interleave clean-prefill chunks with
decode and exposes a loopback SSE demo; see the local serving section below for its limits.

## Measurement contract

- `prefill_ms`: receiver prefix forward + sender prefix forward, CUDA events.
- `fusion_ms`: time from sender forward end to last-prompt-token forward start;
  includes official cache clones and fusion dispatch, not just projector kernels.
- `last_prompt_token_ms`: official final prompt-token pass.
- `fusion_and_overhead_ms`: total wall prefill minus model event intervals;
  broader diagnostic, not an additional disjoint phase to add to `fusion_ms`.
- `official_prefill_total_ms`: synchronized wall time for the whole official path.
- `cache_import_ms`: validation, block allocation, KV copy and first-token extraction.
- `decode_ms`: synchronized batch decode plus greedy sampling and metadata preparation.
- TTFT: from scheduled request arrival until first token and imported cache are
  ready. Includes FIFO queueing and preceding prefills.
- E2E: from arrival through final generated token. Whole-run throughput includes
  first tokens; decode throughput excludes first tokens already produced by prefill.
- Memory: peak PyTorch allocated and reserved bytes, all resident models and pool.
  This does not include every CUDA driver allocation. `nvidia-smi` is separate.

Warmup is excluded from reported benchmark rows; model loading is separately
recorded. An explicit reference-only smoke run uses official `generate`.
Numerical tests compare the official C2C continuation, not an unfused Qwen run.
Teacher-forced logits and free-running greedy equality answer different questions.
JSON records BF16 error metrics and top-1 agreement separately.

## Scope

Single GPU, BF16, greedy sampling, max total length 4096, prefill batch=1,
decode batch at most 8. CUDA graphs are optional, with eager mode retained for
debugging.

## Implemented paths

`OfficialC2C` remains the correctness reference. `OptimizedC2C` temporarily sets both models' `logits_to_keep` to 1 within a scoped wrapper and restores the original methods afterwards. The official outer parameter is not forwarded to the inner models, so changing the outer call alone is insufficient. Full-vocabulary logits for the receiver's final prompt token are still computed. Fusion mathematics, clean-cache cloning, RoPE, first-token handoff and paged import are unchanged.

CLI and serving default to this lean path; `--official-prefill` restores the original CLI path. Existing validation/benchmark entries retain the official reference. Results must be interpreted using their recorded producer and importer configuration.

Chunked `prefill_steps` computes each model's **clean** prefix KV and yields the GPU to decode between chunks. Only after completing the prefix are both outputs passed back to original Rosetta forward for cloning, fusion and the final prompt token. A partially fused chunk is never fed back as clean prefill state. Positions and attention masks use absolute token positions; fusion remains one indivisible execution interval. Chunking changes attention shapes, so elementwise KV equality is not guaranteed on long inputs.

`PagedDecoder(fast_inputs=True)` reuses two fixed pinned CPU staging buffers and corresponding GPU buffers to reduce allocations and H2D calls. CUDA events protect reuse until previous copies complete. Slot mapping remains contiguous as required by the upstream Triton kernel. This option remains opt-in.

`ServingEngine` gives one thread ownership of GPU execution. It supports dynamic arrivals, a bounded waiting queue, page reservation for the full requested output, EOS, cancellation, timeouts, per-token events, and page cleanup on errors/shutdown. Cancellation and timeouts are cooperative between GPU work units, not kernel preemption. For example, a 1 ms HTTP deadline returned timeout after about 123 ms.

## Local serving and reproduction

Run in the server project directory using the existing dedicated Linux CUDA environment:

```bash
export OMP_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false
export TORCHINDUCTOR_CACHE_DIR=$PWD/.cache/inductor
export TRITON_CACHE_DIR=$PWD/.cache/triton
export MPLCONFIGDIR=$PWD/.cache/matplotlib
PY=/root/autodl-tmp/envs/c2c-decode/bin/python

$PY -m pytest -q
$PY scripts/next_experiments.py --section producer
$PY scripts/next_experiments.py --section cpu
$PY scripts/quality_eval.py
$PY scripts/quality_eval.py --chunk-size 1024 --context-only \
  --output results/next/quality-nano-chunk1024-context.json
$PY scripts/service_experiments.py
$PY scripts/next_experiments.py --section e2e
$PY scripts/summarize_next.py

# Lean logits by default; chunking is an optional latency/throughput tradeoff.
$PY -m c2c_decode.server --port 8000
# Alternatively: $PY -m c2c_decode.server --port 8000 --chunk-tokens 1024
```

The service binds only `127.0.0.1` and exposes `GET /health`, `POST /generate` and `DELETE /requests/<id>`. POST returns SSE: request ID, token IDs, cumulative text and completion events. Cumulative text is decoded at each event's own token index so slow clients do not see future tokens.

```bash
curl -N http://127.0.0.1:8000/generate \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"Explain a KV cache briefly.","max_new_tokens":64,"timeout_s":30}'
```

Full queues return 429, invalid inputs 400/413, and engine failure/shutdown 503. Load tables measure engine arrival replay; HTTP experiments check real socket/SSE lifecycle.

Service experiments cover six configurations × three arrival gaps × three fixed seeds × 24 requests, totaling 1296 requests. Configurations share arrivals, input and output lengths within a seed. Each percentile row has only 72 requests. A separate bounded soak and real HTTP lifecycle check supplement these replays. The illustrative SLO was fixed before execution: TTFT≤1000ms and every token gap≤100ms, without tuning after results.

Evaluation scripts are in `scripts/`.

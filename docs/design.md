# Design

[中文版](design.zh-CN.md)

This project adapts the existing Cache-to-Cache method to a small paged decode
backend. It does not introduce or train C2C. Official Rosetta code owns all
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
alignment, fusion settings, precision and positional context. No preemption or
swap is implemented. GPU work is serialized; serial prefill can stall active
decode. The optional serving loop can interleave clean-prefill chunks with
decode and exposes a loopback SSE demo; see `next-improvements.md` for its limits.

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
BF16 tolerances are declared in code before measurements; failures stay visible.
`validate --strict-tokens` additionally returns failure for any teacher-forced
top-1 difference. The default numerical gate must not be described as universal
token equivalence; the JSON records both conclusions separately.

## Scope

Single GPU, BF16, greedy sampling, max total length 4096, prefill batch=1,
decode batch at most 8. CUDA graphs are optional, with eager mode retained for
debugging. No training, quantization, tensor parallel deployment, or claimed
production serving equivalence. Official serving progress must be revisited as
the upstream evolves; the checked main README still announces it as forthcoming.

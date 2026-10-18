# Round Two: Prefill, Scheduling, Serving and Quality

[中文版](next-improvements.zh-CN.md)

This round retains Qwen3-4B → Qwen3-0.6B, the official Fuser, BF16 and RTX 4090, without training or quantization. Performance measurements disable PyTorch Profiler. Raw records and generated summaries are in `results/next/`.

## Implemented paths

`OfficialC2C` remains the correctness reference. `OptimizedC2C` temporarily sets both models' `logits_to_keep` to 1 within a scoped wrapper and restores the original methods afterwards. The official outer parameter is not forwarded to the inner models, so changing the outer call alone is insufficient. Full-vocabulary logits for the receiver's final prompt token are still computed. Fusion mathematics, clean-cache cloning, RoPE, first-token handoff and paged import are unchanged.

CLI and serving default to this lean path; `--official-prefill` restores the original CLI path. Existing validation/benchmark entries retain the official reference. Historical results do not measure the new default.

Chunked `prefill_steps` computes each model's **clean** prefix KV and yields the GPU to decode between chunks. Only after completing the prefix are both outputs passed back to original Rosetta forward for cloning, fusion and the final prompt token. A partially fused chunk is never fed back as clean prefill state. Positions and attention masks use absolute token positions; fusion remains one indivisible execution interval. Chunking changes attention shapes, so elementwise KV equality is not guaranteed on long inputs.

`PagedDecoder(fast_inputs=True)` reuses two fixed pinned CPU staging buffers and corresponding GPU buffers to reduce allocations and H2D calls. CUDA events protect reuse until previous copies complete. Slot mapping remains contiguous as required by the upstream Triton kernel. This option remains opt-in.

`ServingEngine` gives one thread ownership of GPU execution. It supports dynamic arrivals, a bounded waiting queue, page reservation for the full requested output, EOS, cancellation, timeouts, per-token events, and page cleanup on errors/shutdown. Cancellation and timeouts are cooperative between GPU work units, not kernel preemption. For example, a 1 ms HTTP deadline returned timeout after about 123 ms; this is not hard real-time enforcement.

## Producer and CPU optimization results

Five seeded randomized pairs per input length give these producer medians:

| Input | Official prefill | Lean logits | Time reduction |
|---:|---:|---:|---:|
| 1024 | 156.70ms | 152.79ms | 2.5% |
| 2048 | 233.83ms | 219.99ms | 5.9% |

This memory experiment includes only official producer models and Fuser, **without the nano receiver or paged pool**. The 2K allocated peak falls **12.469 → 11.309GiB**, about **1.16GiB**. Lean fused KV is elementwise identical on six boundary/long-input cases, with final-prompt logits RMSE 0. These benefits cannot simply be added to the previous 26.4% import optimization.

Independent input-buffer pairs reduce median import+decode **71.603 → 70.918ms** at 1K/B1 and **144.430 → 143.307ms** at 2K/B8. One of five 257/B4 pairs is slower; the overall benefit is about 1%, so reuse is not default. All 15 pairs generate identical tokens.

Chunking slows single-request prefill: 2K takes **514.95/324.34/258.10ms** with 256/512/1024-token chunks. Its purpose is to reduce decode stalls, not accelerate isolated prefill. See the generated summary for fresh-request E2E, randomized arrivals, maximum gaps and SLO data.

## End-to-end behavior and defaults

Every fresh request reruns prefill, fusion, import and decode. Each configuration has five repetitions in randomized mode order. These are single-process nano timings, excluding IPC, HTTP, tokenization, loading and compilation; do not directly subtract historical IPC timings.

| Input / concurrency / output per request | Original producer E2E | Lean logits E2E | Time reduction |
|---|---:|---:|---:|
| 1K / 1 / 32 | 228.49ms | 225.95ms | 1.1% |
| 1K / 1 / 128 | 450.65ms | 445.61ms | 1.1% |
| 1K / 8 / 32 | 1362.67ms | 1325.72ms | 2.7% |
| 2K / 1 / 32 | 311.56ms | 297.64ms | 4.5% |
| 2K / 1 / 128 | 547.21ms | 532.76ms | 2.6% |
| 2K / 8 / 32 | 2009.88ms | 1899.04ms | 5.5% |

Across 120 E2E records, original, lean and lean+buffer modes have identical token histories. At 2K/B8, adding buffer reuse gives 1900.09ms, slightly slower than 1899.04ms. Its small benefit is not a stable E2E speedup claim. Only lean logits is enabled by default in CLI/serving.

The chunked policy also yields between the two models and fusion; effects are not solely attributable to token chunk size. Both rows below enable lean logits and buffer reuse, differing only in chunking. Percentiles pool 72 burst-arrival requests over three replays; throughput is the median across replays.

| Policy | Token-gap P99 | Maximum token gap | TTFT P99 | Throughput |
|---|---:|---:|---:|---:|
| Whole prefill | 388.24ms | 980.81ms | 4002.99ms | 347.38 tok/s |
| 1024-token chunk | 81.51ms | 83.00ms | 4233.56ms | 329.18 tok/s |

Token-gap P99 falls **79.0%**, at a cost of **5.2%** lower throughput and **5.8%** higher TTFT P99. At mean arrival gaps of 350ms, fixed-SLO passes rise from 36/72 to 72/72: a finite-sample observation, not a service guarantee. The 256-token chunk reaches only 243.53 tok/s under bursts; finer chunking is not automatically better. Chunking remains disabled by default, selectable according to stall/waiting preferences.

All 37 tests pass; all 54 replays restore the pool to 73 pages. A separate 120.4-second soak submits 589 requests: 458 complete, 85 cancel, 46 time out, and zero engine errors. Live allocated memory is 13,678,994,432 bytes both initially and finally, with all pages reclaimed. The HTTP check deliberately sets queue capacity to 2: four client threads issue eight calls, three complete and five receive HTTP 429. DELETE cancellation, timeout events, client disconnect and health checks are also verified.

![Measured overview](../results/next/overview.png)

[Scalable SVG](../results/next/overview.svg)

[Complete generated summary](../results/next/summary.md) retains all configurations and numerical-failure counts; raw JSON is alongside it.

## Expanded correctness and task quality

The 36 authored functional cases include ten math, ten QA, ten Python-output reasoning and six long-context retrieval cases. Prompts, expected answers, official/adapted outputs and token IDs are retained; generated code is not executed. This is not a standard GSM8K/HumanEval evaluation or a measure of overall model capability.

Strict normalized-answer matching scores **7/36** for official and all adapted paths. Extra explanation counts as mismatch, so the score mixes content correctness with format compliance. For example, official output `The answer is 323.` fails the answer-only format. Genuine content errors also occur: official answers `3*x+7=28` with `x=3`.

Nano original, lean and 512-token chunk paths each match official output token-for-token on **35/36** cases. The remaining FIFO answer differs by comma/hyphen at an official top-1/top-2 margin of **0**; official itself also violates the requested answer format. Adapted answers match each other. No new answer degradation was observed here, but general quality equivalence is not established.

Expanded stepwise logits checks **do not all pass the existing thresholds** of cosine≥0.999, RMSE≤0.15 and max_abs≤1. Nano has cosine about **0.998749** on one valid `math-7` generation step, identically across all three producer modes: an existing backend difference, not introduced by lean logits. The 512-token chunk adds a valid-step threshold failure on `context-0`. Unchanged output does not make that configuration numerically validated. All chunked modes remain experimental. Other failures occur after forced generation beyond EOS; both all-step and pre-/at-EOS classifications remain in the summary and `numerical-failures.json`, without relabeling failures as passes.

A targeted 1024-token chunk follow-up on the six existing long-context cases matches official tokens on 6/6, passes numerical thresholds for checked steps, imports KV exactly and reclaims all pages. It is a supplementary check on existing cases, not an independent full quality benchmark; 1024-token chunking remains experimental and off by default.

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

Full queues return 429, invalid inputs 400/413, and engine failure/shutdown 503. This local demonstration has no public deployment, authentication, multi-GPU or OpenAI API compatibility guarantee. Load tables measure engine arrival replay; HTTP experiments check real socket/SSE lifecycle. They do not establish HTTP tail latency. A two-minute soak is not long-term stability evidence.

Service experiments cover six configurations × three arrival gaps × three fixed seeds × 24 requests, totaling 1296 requests. Configurations share arrivals, input and output lengths within a seed. Each percentile row has only 72 requests. A separate bounded soak and real HTTP lifecycle check supplement these replays. The illustrative SLO was fixed before execution: TTFT≤1000ms and every token gap≤100ms, without tuning after results.

Core code is about 1738 lines including blanks, within the intended scope. Evaluation scripts are in `scripts/`. Models, Fuser, attention kernels and the cache allocator reuse upstream implementations.

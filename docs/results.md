# Performance and Correctness Evaluation

[中文版](results.zh-CN.md)

Experiments use the pinned Qwen3-4B → Qwen3-0.6B pair, official Fuser, BF16 and RTX 4090. Results are organized by system component and measured configuration; costs from different configurations must not be added or substituted.

CLI/serving defaults to lean prefix logits, scatter import and paged decoding. Chunked prefill and input buffers are optional; Graph and eager are measured separately. See [design and serving](design.md), [matched-work efficiency](efficiency-verification.md) and [profiling](profiling-results.md).

Measured 2026-10-17 with pinned upstream commits and model revisions in BF16.

## Correctness and limits

- Official C2C generation runs with the selected Qwen3-4B → Qwen3-0.6B Fuser.
- Imported KV is elementwise identical at lengths 31/255/256/257/1024/2048.
- Four ordinary prompts match official 16-token greedy generation exactly.
- EOS, variable output lengths, reuse and complete block reclamation pass.
- Infrastructure tests cover CUDA paging, scheduling and the strict-token CLI
  exit gate; all downloaded weight hashes verify. See `results/tests.txt`.
- Long-input teacher-forced logits pass predeclared BF16 thresholds, but strict
  token equivalence does **not** hold: eager top-1 matches 88/90 steps and CUDA
  Graph matches 87/90.

Eager worst absolute logit error: 0.375; worst RMSE: 0.08594; minimum cosine:
0.999341. The two differing argmax steps have official top-1 margins 0.125 and
0.0625. This is consistent with different BF16 arithmetic paths. The default `passed` field means the documented
tolerance and fixture gates; `strict_token_equivalence` is separate.

Raw records are in `results/validation-eager.json` and `validation-graph.json`.

## Official producer and page-loop import control

This section measures full official logits with loop import, not the CLI/serving default.

Medians of three repetitions, 32 generated tokens, warmed per batch/length.
Prefill batch=1; decode requests have different prompt lengths. Nominal length
is the longest request, with each subsequent request 17 tokens shorter.

| Prompt | Official decode B1 | Eager decode B1 | Graph decode B1 | Graph decode B8 | Graph E2E B8 |
|---:|---:|---:|---:|---:|---:|
| 1024 | 42.4 tok/s | 54.3 tok/s | 434.5 tok/s | 2413.5 tok/s | 182.8 tok/s |
| 2048 | 42.6 tok/s | 54.6 tok/s | 405.2 tok/s | 1816.6 tok/s | 123.0 tok/s |

Compare B1 columns for single-request speedup. B8 is aggregate throughput.
Single-request graph E2E is 135.1/98.7 tok/s at 1K/2K, against official
35.8/33.2 tok/s. These are synthetic performance tests, not an accuracy benchmark.

Graph TTFT at B1 is 165.3/247.5 ms (1K/2K). At B8, median request TTFT rises
to 738.4/1106.3 ms because it includes serial prefill queueing.

| Single-request phase | 1K median | 2K median |
|---|---:|---:|
| Receiver + sender prefix forward | 93.206 ms | 168.065 ms |
| Fusion region, including official cache management | 43.529 ms | 45.904 ms |
| Final prompt-token forward | 23.403 ms | 23.952 ms |
| Cache validation/allocation/import | 4.273 ms | 8.400 ms |

Independent medians do not sum exactly to E2E. See `docs/design.md` for timer
definitions. Peak PyTorch allocated memory is 15.60 GiB with the paged model
and graphs, versus 12.47 GiB for the official-only baseline. The bounded KV
pool allocates 73 × 28 MiB = 2044 MiB under its 2048 MiB budget. Driver memory
outside PyTorch is not included in the allocated-memory peak.

CUDA Graph replay removes much of the host launch overhead that remains in
eager decode. Batching raises aggregate throughput. Serial prefill and fusion
dominate E2E in this control configuration; cache import is a smaller cost. Useful future experiments
include prefill/decode overlap and correctly keyed fused-prefix reuse.

## Producer and CPU optimization results

Five seeded randomized pairs per input length give these producer medians:

| Input | Official prefill | Lean logits | Time reduction |
|---:|---:|---:|---:|
| 1024 | 156.70ms | 152.79ms | 2.5% |
| 2048 | 233.83ms | 219.99ms | 5.9% |

This memory experiment includes only official producer models and Fuser, **without the nano receiver or paged pool**. The 2K allocated peak falls **12.469 → 11.309GiB**, about **1.16GiB**. Lean fused KV is elementwise identical on six boundary/long-input cases, with final-prompt logits RMSE 0. These benefits cannot simply be added to the separate 26.4% import optimization.

Independent input-buffer pairs reduce median import+decode **71.603 → 70.918ms** at 1K/B1 and **144.430 → 143.307ms** at 2K/B8. One of five 257/B4 pairs is slower; the overall benefit is about 1%, so reuse is not default. All 15 pairs generate identical tokens.

Chunking slows single-request prefill: 2K takes **514.95/324.34/258.10ms** with 256/512/1024-token chunks. Its purpose is to reduce decode stalls. See the generated summary for fresh-request E2E, randomized arrivals, maximum gaps and SLO data.

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

Nano original, lean and 512-token chunk paths each match official output token-for-token on **35/36** cases. The remaining FIFO answer differs by comma/hyphen at an official top-1/top-2 margin of **0**; official itself also violates the requested answer format. Adapted answers match each other. No new answer degradation was observed in these cases.

Expanded stepwise logits checks **do not all pass the existing thresholds** of cosine≥0.999, RMSE≤0.15 and max_abs≤1. Nano has cosine about **0.998749** on one valid `math-7` generation step, identically across all three producer modes: an existing backend difference, not introduced by lean logits. The 512-token chunk adds a valid-step threshold failure on `context-0`. All chunked modes remain experimental. Other failures occur after forced generation beyond EOS; both all-step and pre-/at-EOS classifications remain in the summary and `numerical-failures.json`.

A targeted 1024-token chunk follow-up on the six existing long-context cases matches official tokens on 6/6, passes numerical thresholds for checked steps, imports KV exactly and reclaims all pages. It is a supplementary check on existing cases, not an independent full quality benchmark; 1024-token chunking remains experimental and off by default.

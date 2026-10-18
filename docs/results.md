# First measured results — RTX 4090

[中文版](results.zh-CN.md)

This document records the first measurement round. Later serving and optimization work is described in [round two](next-improvements.md); historical scope statements below are not the current feature list.

Measured 2026-10-17 with pinned upstream commits and model revisions, BF16,
without quantization or training. This adapts the existing C2C method.

## Correctness and limits

- Official C2C generation runs with the selected Qwen3-4B → Qwen3-0.6B Fuser.
- Imported KV is elementwise identical at lengths 31/255/256/257/1024/2048.
- Four ordinary prompts match official 16-token greedy generation exactly.
- EOS, variable output lengths, reuse and complete block reclamation pass.
- Infrastructure tests cover CUDA paging, scheduling and the strict-token CLI
  exit gate; all downloaded weight hashes verify. See `results/tests.txt`.
- Long-input teacher-forced logits pass predeclared BF16 thresholds, but strict
  token equivalence does **not** hold: eager top-1 matches 88/90 steps and CUDA
  Graph matches 87/90. Do not claim universally identical generated sequences.

Eager worst absolute logit error: 0.375; worst RMSE: 0.08594; minimum cosine:
0.999341. The two differing argmax steps have official top-1 margins 0.125 and
0.0625. This is consistent with different BF16 arithmetic paths, but does not
prove that every future difference has that cause. `validate --strict-tokens`
exposes a stricter, failing gate. The default `passed` field means the documented
tolerance and fixture gates; `strict_token_equivalence` is separate.

Raw records are in `results/validation-eager.json` and `validation-graph.json`.

## Performance

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
now dominate E2E; cache import is a smaller cost. Useful future experiments
include prefill/decode overlap and correctly keyed fused-prefix reuse.

New code is approximately 756 core Python lines and about 1100 lines including
tests and scripts, excluding upstream repositories. This prototype has no HTTP
server, cancellation API, preemption, multi-GPU support or training pipeline.

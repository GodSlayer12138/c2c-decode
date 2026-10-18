# Matched-work efficiency verification

One RTX 4090, BF16, batch=1, fixed output count. Five or more interleaved pairs per case.
All decode paths start from the exact same official fused KV and consume the same token history.
Every step still computes full logits and greedy argmax. EOS is disabled; argmax differences remain recorded.
Cloning shared test fixtures and model/JIT/graph setup are excluded. Cache import is included in E2E.

| Prompt | Output | Mode | Official decode ms | Candidate decode ms | Paired speedup | Observed range | Faster pairs |
|---:|---:|---|---:|---:|---:|---:|---:|
| 1024 | 32 | official_flash2 | 717.78 | 743.92 | 0.96x | 0.95–1.08x | 1/5 |
| 1024 | 32 | paged_eager | 717.78 | 546.96 | 1.31x | 1.30–1.46x | 5/5 |
| 1024 | 32 | paged_graph | 717.78 | 72.27 | 10.00x | 9.93–11.06x | 5/5 |
| 1024 | 128 | official_flash2 | 2898.40 | 3037.06 | 0.96x | 0.95–0.96x | 0/5 |
| 1024 | 128 | paged_eager | 2898.40 | 2218.84 | 1.31x | 1.30–1.32x | 5/5 |
| 1024 | 128 | paged_graph | 2898.40 | 292.50 | 9.98x | 9.83–10.04x | 5/5 |
| 2048 | 32 | official_flash2 | 711.54 | 739.45 | 0.96x | 0.96–0.96x | 0/5 |
| 2048 | 32 | paged_eager | 711.54 | 542.82 | 1.31x | 1.31–1.32x | 5/5 |
| 2048 | 32 | paged_graph | 711.54 | 77.37 | 9.21x | 9.18–9.43x | 5/5 |
| 2048 | 128 | official_flash2 | 2945.62 | 3055.08 | 0.96x | 0.95–0.96x | 0/5 |
| 2048 | 128 | paged_eager | 2945.62 | 2242.16 | 1.31x | 1.31–1.32x | 5/5 |
| 2048 | 128 | paged_graph | 2945.62 | 313.36 | 9.36x | 9.29–9.43x | 5/5 |

Fresh-prefill E2E pairs (32 output tokens, including prefill/fusion/import/decode):

| Prompt | Official E2E ms | Graph E2E ms | Paired speedup | Faster pairs |
|---:|---:|---:|---:|---:|
| 1024 | 865.59 | 232.34 | 3.73x | 5/5 |
| 2048 | 952.28 | 320.41 | 2.97x | 5/5 |

Ranges are observed paired extrema, not confidence intervals.
Both HF and nano receivers remain resident in this controlled experiment; it is not a memory-efficiency comparison.
FlashAttention-2 is an additional HF decode control; official prefill remains SDPA in all cases.
This validates warm execution efficiency, not universally identical text or equal task accuracy.
The project integrates existing nano-vLLM kernels and CUDA Graphs; it does not invent those speedups.

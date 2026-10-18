# Measured performance

GPU: NVIDIA GeForce RTX 4090

Medians over recorded repetitions; fixed output counts; warmed kernels.

| Prompt nominal | Decode batch | Decode tok/s | E2E tok/s | TTFT p50 ms | Peak allocated GiB |
|---:|---:|---:|---:|---:|---:|
| 1024 | 1 | 434.5 | 135.1 | 165.3 | 14.17 |
| 1024 | 2 | 766.1 | 155.7 | 247.9 | 14.17 |
| 1024 | 4 | 1380.8 | 169.7 | 414.3 | 14.17 |
| 1024 | 8 | 2413.5 | 182.8 | 738.4 | 14.17 |
| 2048 | 1 | 405.2 | 98.7 | 247.5 | 15.60 |
| 2048 | 2 | 680.2 | 109.7 | 368.4 | 15.60 |
| 2048 | 4 | 1190.0 | 118.1 | 613.2 | 15.60 |
| 2048 | 8 | 1816.6 | 123.0 | 1106.3 | 15.60 |

Official C2C cached decode baseline (batch=1; no second receiver or paged pool):

| Prompt | Decode tok/s | E2E tok/s | Peak allocated GiB |
|---:|---:|---:|---:|
| 1024 | 42.4 | 35.8 | 11.05 |
| 2048 | 42.6 | 33.2 | 12.47 |

TTFT includes FIFO queueing and cache import. Prefill batch is always one.
Decode throughput excludes each request's first token; E2E includes all phases.
Peak allocated is PyTorch memory, not total device usage. No claim of production serving performance.

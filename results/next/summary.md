# System configuration measurements

RTX 4090, BF16, unchanged pinned Qwen3 pair and trained C2C Fuser. All speed measurements have PyTorch Profiler disabled.

## Producer: randomized mode order, 5 repetitions

Memory peaks below include official producer models/Fuser only, without nano receiver or its KV pool.

| Input | Mode | Median prefill ms | Peak allocated GiB | Validation cases | Exact KV cases |
|---|---|---|---|---|---|
| 1024 | official | 156.70 | 11.041 | 0 | 0 |
| 1024 | lean | 152.79 | 10.463 | 6 | 6 |
| 1024 | chunk-256 | 297.70 | 10.474 | 6 | 4 |
| 1024 | chunk-512 | 191.92 | 10.488 | 6 | 4 |
| 1024 | chunk-1024 | 154.66 | 10.470 | 6 | 5 |
| 2048 | official | 233.83 | 12.469 | 0 | 0 |
| 2048 | lean | 219.99 | 11.309 | 6 | 6 |
| 2048 | chunk-256 | 514.95 | 11.326 | 6 | 4 |
| 2048 | chunk-512 | 324.34 | 11.326 | 6 | 4 |
| 2048 | chunk-1024 | 258.10 | 11.339 | 6 | 5 |

Validation counts pool lengths 31/255/256/257/1024/2048; official is the comparator, not a test case. Chunk validation above covers handoff logits, not all continuation steps.

## Input preparation: paired trials

| Input | Batch | Original handoff ms | Buffers handoff ms | Paired saved ms | Min..max saved | Faster pairs / 5 |
|---|---|---|---|---|---|---|
| 257 | 4 | 84.539 | 83.526 | 0.638 | -2.727 .. 1.013 | 4 |
| 1024 | 1 | 71.603 | 70.918 | 0.695 | 0.464 .. 0.905 | 5 |
| 2048 | 8 | 144.430 | 143.307 | 1.062 | 0.574 .. 2.680 | 5 |

Token histories match within all 15 input-buffer pairs. This small gain remains opt-in.

## Fresh-request end to end

Includes fresh serial prefills, fusion, import and decode. One process, no IPC/HTTP. Five repetitions, randomized mode order. Batch>1 lengths are N-17*i.

| Input | Batch | Output/request | Official ms | Lean ms | Lean+buffers ms | Chunk512+buffers ms | Lean reduction |
|---|---|---|---|---|---|---|---|
| 1024 | 1 | 32 | 228.49 | 225.95 | 224.15 | 262.60 | 1.11% |
| 1024 | 1 | 128 | 450.65 | 445.61 | 442.54 | 481.08 | 1.12% |
| 1024 | 8 | 32 | 1362.67 | 1325.72 | 1324.14 | 1631.74 | 2.71% |
| 2048 | 1 | 32 | 311.56 | 297.64 | 296.38 | 399.10 | 4.47% |
| 2048 | 1 | 128 | 547.21 | 532.76 | 529.44 | 630.93 | 2.64% |
| 2048 | 8 | 32 | 2009.88 | 1899.04 | 1900.09 | 2720.12 | 5.51% |

All official/lean/lean+buffer token histories match in this E2E matrix. Chunk is an experimental scheduling tradeoff and does not preserve bitwise KV.

## Authored task quality and logit diagnostics

36 authored cases: 10 arithmetic, 10 QA, 10 Python-output reasoning, 6 context retrieval. Strict normalized answer exact match also penalizes extra explanation. It is not semantic accuracy or a public benchmark. Generation is capped at 64 tokens.

| Engine | Producer | Official exact answers / 36 | Engine exact answers / 36 | Exact sequences / 36 | Truncated / 36 | Changed text vs old nano / 36 |
|---|---|---|---|---|---|---|
| nano | official | 7 | 7 | 35 | 1 | 0 |
| nano | lean | 7 | 7 | 35 | 1 | 0 |
| nano | chunk-512 | 7 | 7 | 35 | 1 | 0 |

Numerical tolerances remain cosine>=0.999, RMSE<=0.15, max_abs<=1. The following failures are retained, not reclassified as passes. Teacher forcing also continues after EOS; before/at-EOS failures are distinguished explicitly.

| Engine | Producer | Failing checked steps | Failing steps before/at EOS |
|---|---|---|---|
| nano | official | 12 | 1 |
| nano | lean | 12 | 1 |
| nano | chunk-512 | 13 | 2 |

### Targeted 1024-token chunk follow-up

Six long-context cases only, after observing the 512-token chunk failure; not a new independent full benchmark.

| Cases | Exact sequences vs official | Checked steps | Numerical failures | Before/at EOS failures |
|---|---|---|---|---|
| 6 | 6 | 90 | 0 | 0 |

## Scheduled arrival replay

Three fixed seeds, 24 requests/run, exponential arrival intervals (or burst), six modes: 1296 requests total. Each mode reuses the exact same arrival/length/output schedule per seed. Quantiles pool 72 requests per row; still small empirical tails.

| Mean arrival gap ms | Mode | Median tok/s | TTFT P99 ms | Token gap P99 ms | Max token gap ms | Request TPOT P95 ms | SLO good / 72 |
|---|---|---|---|---|---|---|---|
| 0 | official | 340.48 | 4092.77 | 385.41 | 1009.76 | 32.76 | 0 |
| 0 | lean | 347.81 | 4004.35 | 386.28 | 990.59 | 32.65 | 0 |
| 0 | lean-fast | 347.38 | 4002.99 | 388.24 | 980.81 | 32.35 | 0 |
| 0 | chunk-256 | 243.53 | 6524.46 | 77.39 | 104.40 | 34.79 | 9 |
| 0 | chunk-512 | 306.62 | 4684.27 | 76.50 | 77.11 | 31.23 | 16 |
| 0 | chunk-1024 | 329.18 | 4233.56 | 81.51 | 83.00 | 32.82 | 19 |
| 100 | official | 249.03 | 1946.61 | 371.40 | 911.77 | 52.60 | 2 |
| 100 | lean | 253.49 | 1870.06 | 294.99 | 871.72 | 53.14 | 3 |
| 100 | lean-fast | 254.24 | 1872.98 | 361.74 | 875.43 | 53.34 | 3 |
| 100 | chunk-256 | 141.51 | 5348.11 | 77.59 | 84.20 | 38.55 | 11 |
| 100 | chunk-512 | 207.11 | 2769.38 | 76.85 | 89.01 | 41.13 | 29 |
| 100 | chunk-1024 | 240.74 | 2131.76 | 81.94 | 82.05 | 47.36 | 36 |
| 350 | official | 116.34 | 548.42 | 167.63 | 480.22 | 30.06 | 37 |
| 350 | lean | 116.49 | 525.00 | 167.64 | 401.00 | 29.91 | 37 |
| 350 | lean-fast | 116.59 | 510.71 | 162.59 | 385.15 | 28.85 | 36 |
| 350 | chunk-256 | 111.13 | 1668.52 | 76.61 | 88.60 | 36.83 | 53 |
| 350 | chunk-512 | 115.27 | 856.83 | 75.45 | 79.33 | 36.88 | 72 |
| 350 | chunk-1024 | 116.11 | 647.38 | 81.13 | 83.82 | 26.67 | 72 |

Illustrative SLO fixed before the runs: TTFT<=1000ms AND every token gap<=100ms. No mode guarantees this SLO. This table measures engine arrival replay, not HTTP latency.

## HTTP lifecycle and soak

| Check | Observed |
|---|---|
| Queue overflow HTTP status | 429 |
| DELETE cancellation | cancelled |
| Deadline | timeout |
| Request elapsed at cancellation ms | 147.82 |
| 1ms deadline event latency ms | 123.37 |
| Disconnected request retired | True |
| Concurrent calls | 8 |
| Soak seconds | 120.4 |
| Offered / rejected | 589 / 0 |
| Completion reasons | {'length': 458, 'cancelled': 85, 'timeout': 46, 'error': 0} |
| Final free pages | 73 |
| Allocated before / after GiB | 12.7396 / 12.7396 |

Cancellation/timeout are cooperative between GPU work units. The two-minute soak is a bounded stability experiment, not a long-term leak proof or production SLA.


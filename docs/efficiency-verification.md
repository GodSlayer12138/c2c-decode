# Efficiency Verification

[中文版](efficiency-verification.zh-CN.md)

Measured on a rented RTX 4090 24GB on 2026-10-17. In warmed BF16, batch=1 tests with 1K/2K inputs, nano-vLLM CUDA Graph decoding was approximately **9.2–10.0× faster**, and end-to-end requests producing 32 tokens were **3.0–3.7× faster**, relative to the pinned official C2C/Hugging Face execution path.

## Comparison method

- Models, Fuser and upstream revisions follow the lock file. No training or quantization.
- All decode paths start from the same official fused KV; imported KV is read back and checked elementwise. Token histories are identical.
- Four paths: official SDPA, HF FlashAttention-2 decode, paged eager, and paged CUDA Graph. The FlashAttention-2 control changes decode attention only; prefill remains SDPA.
- After warmup, each configuration uses five paired repetitions in seeded randomized order: 80 decode records and 20 end-to-end records.
- Outputs of 32/128 tokens require 31/127 decode steps; the first token comes from official prefill. Every path computes full logits and argmax at every step, then receives the same forced continuation token. EOS truncation is disabled to match work; argmax differences remain recorded.
- Decode timing excludes fixture cloning and KV import. End-to-end timing reruns official prefill, fusion and the final prompt token, and includes import and decode. The official path does not clone a fixture.
- CUDA is synchronized around timing. All four paths alternate in one process with two receiver copies resident. Loading, compilation and graph capture are excluded from request latency; initialization took about 50.65 seconds.

## Results

Times are medians of five repetitions; speedups are medians of paired ratios.

| Input / output tokens | Official decode | Graph decode | Decode speedup | Graph faster |
|---|---:|---:|---:|---:|
| 1024 / 32 | 717.78 ms | 72.27 ms | 10.00× | 5/5 |
| 1024 / 128 | 2898.40 ms | 292.50 ms | 9.98× | 5/5 |
| 2048 / 32 | 711.54 ms | 77.37 ms | 9.21× | 5/5 |
| 2048 / 128 | 2945.62 ms | 313.36 ms | 9.36× | 5/5 |

| Input / output tokens | Official E2E | Graph E2E | E2E speedup | Graph faster |
|---|---:|---:|---:|---:|
| 1024 / 32 | 865.59 ms | 232.34 ms | 3.73× | 5/5 |
| 2048 / 32 | 952.28 ms | 320.41 ms | 2.97× | 5/5 |

Paged eager is approximately **1.31×** faster across all four configurations. Graph execution further reduces paged decode latency by approximately 7–8×. Most of the observed benefit comes from reusing nano-vLLM's CUDA Graph execution; it cannot all be attributed to paging or claimed as an invention of this project. This experiment does not separately attribute eager model, KV append and individual operator costs.

HF FlashAttention-2 decode achieves approximately **0.96×** paired speedup over SDPA, offering no improvement here. Changing that attention kernel does not remove the observed gap; other engines or optimized HF implementations may behave differently.

Mean official prefill time is nearly identical between the official/Graph E2E paths: 158.08/157.97 ms at 1K and 236.45/236.93 ms at 2K. Graph adds mean import costs of 3.90/7.64 ms. This explains the absence of a first-token speedup and why prefill and fusion dilute the decode improvement.

## Scope and limitations

- This is a warmed execution benchmark on fixed-length synthetic inputs, not a task accuracy evaluation. At 1K, Graph agrees with official top-1 on 30/31 and 126/127 steps; at 2K, agreement is 31/31 and 127/127. **Strict output equivalence or lossless task quality is not established.**
- Earlier independent measurements report official/Graph B1 TTFT of 161.9/165.3 ms at 1K and 238.4/247.5 ms at 2K. These are first-round results, not new TTFT measurements in this audit.
- This audit does not compare memory savings. Earlier independent peaks rise from 12.47 GiB official to 15.60 GiB Graph, including an extra receiver and a 2044 MiB KV pool. Paging provides bounded capacity and allocation/reclamation; lower peak memory has not been demonstrated.
- Earlier batch=8 throughput of 2413.5/1816.6 tok/s is aggregate decode throughput, not evidence for B1 speedup. All efficiency conclusions here use B1 controls.
- Ranges across five repetitions are not confidence intervals. All pages were reclaimed and GPU memory/utilization returned to zero after exit. This round did not test long-running service loads, P99 or energy consumption.

## Reproduction and evidence

Run in the dedicated Linux CUDA environment:

```bash
python -m c2c_decode.efficiency --repeats 5 \
  --output results/efficiency-audit.json
```

- [Verification code](../src/c2c_decode/efficiency.py)
- [Raw repetitions, token histories and versions](../results/efficiency-audit.json)
- [Generated complete comparison](../results/efficiency-audit.md)
- [First-round correctness, throughput and memory results](results.md)

Suggested project description: Imported official C2C fused KV into nano-vLLM paged decoding and CUDA Graph on RTX 4090. Matched-state, matched-history single-request tests with 1K/2K inputs show approximately 9–10× faster decode and 3.0–3.7× faster end-to-end execution for 32-token outputs, with BF16 numerical differences and additional memory costs explicitly recorded.

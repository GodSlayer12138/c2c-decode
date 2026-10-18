# Profiling measurements

RTX 4090 / BF16 / pinned Qwen3 C2C pair. See docs/profiling-results.md for interpretation.

## KV import: paired, randomized order, profiler disabled

Five pairs per case; 32 output tokens/request; mixed lengths N - 17*i; medians in milliseconds.

| Input | Batch | Loop import | Scatter import | Paired speedup | Loop handoff+decode | Scatter handoff+decode | Latency reduction | Payload GB/s |
|---|---|---|---|---|---|---|---|---|
| 1024 | 1 | 3.859 | 1.104 | 3.47x | 74.27 | 71.64 | 3.5% | 106.3 |
| 1024 | 2 | 7.657 | 2.140 | 3.57x | 87.63 | 82.08 | 6.3% | 108.8 |
| 1024 | 4 | 15.294 | 4.221 | 3.62x | 104.21 | 93.15 | 10.6% | 108.5 |
| 1024 | 8 | 30.642 | 8.273 | 3.71x | 132.54 | 110.02 | 17.0% | 107.0 |
| 2048 | 1 | 7.538 | 1.173 | 6.43x | 83.11 | 76.72 | 7.7% | 200.2 |
| 2048 | 2 | 14.965 | 2.210 | 6.83x | 105.16 | 92.43 | 12.1% | 211.7 |
| 2048 | 4 | 29.932 | 4.206 | 7.12x | 133.22 | 107.55 | 19.3% | 220.6 |
| 2048 | 8 | 59.978 | 8.365 | 7.17x | 195.63 | 143.97 | 26.4% | 218.1 |

All 40 pairs produce identical tokens. Payload GB/s counts each KV byte once, not DRAM bandwidth.

## Host stages and CUDA trace

Five profiler-disabled trials with removable host timers; 16 output tokens (15 decode steps). Nested stages must not be summed.

| Engine | Input | Batch | Decode ms/step | Trace kernels | Kernel launch API calls | GraphLaunch calls | Trace GPU union ms | Trace GPU coverage | Largest internal GPU gap ms |
|---|---|---|---|---|---|---|---|---|---|
| nano-graph | 1024 | 1 | 2.268 | 6045 | 120 | 15 | 32.21 | 66.2% | 0.517 |
| nano-graph | 2048 | 8 | 4.494 | 6900 | 555 | 15 | 68.49 | 73.9% | 0.708 |
| nano-eager | 1024 | 1 | 17.056 | 6000 | 6000 | 0 | 32.44 | 7.3% | 0.360 |

Trace region includes import and decode. Coverage is the union of traced GPU activity divided by profiled region wall time; it is NOT SM utilization or occupancy. Eager capture records shapes and has substantial instrumentation overhead.

### nano-graph: 1K B1 host durations

| Stage | Calls in 5 runs | Host ms / decode step |
|---|---|---|
| import | 5 | 0.0649 |
| prepare_inputs | 75 | 0.0896 |
| vocabulary_projection | 75 | 0.0320 |
| model_call | 75 | 0.1120 |

### nano-graph: largest GPU kernels in trace

| Kernel (abbreviated) | Count | GPU ms |
|---|---|---|
| std::enable_if<!(false), void>::type internal::gemvx::kernel<int, int, __nv_bfloat16, __nv_bfloat16, __nv_bflo | 1695 | 22.015 |
| void flash::flash_fwd_splitkv_kernel<Flash_fwd_kernel_traits<128, 64, 128, 4, false, false, cutlass::bfloat16_ | 420 | 3.579 |
| void flash::flash_fwd_splitkv_combine_kernel<Flash_fwd_kernel_traits<128, 64, 128, 4, false, false, cutlass::b | 420 | 2.441 |
| triton_per_fused__to_copy_add_mean_mul_pow_rsqrt_0 | 1695 | 1.807 |
| triton_poi_fused__to_copy_cat_0 | 420 | 0.435 |
| triton_poi_fused__to_copy_cat_1 | 420 | 0.430 |

### Import-only trace

Kernel counts explain launch overhead; trace timings are not performance benchmark values.

| Importer | Kernels | CUDA launch calls | Profiled region ms | GPU union ms |
|---|---|---|---|---|
| loop | 3592 | 3592 | 136.90 | 11.21 |
| scatter | 488 | 488 | 18.67 | 5.94 |

## Memory lifetime

Single process: official C2C + nano graph engine + 2044 MiB KV pool. Values are PyTorch allocator GiB, not total device memory.

| Stage | Input/logits shape | Allocated GiB | Reserved GiB | Cumulative peak GiB |
|---|---|---|---|---|
| resident_models_and_pool |  | 12.732 | 12.807 | 12.732 |
| before_prefill | 1024 | 12.732 | 12.807 | 12.732 |
| model_0_forward_end | [1, 1023, 151936] | 13.130 | 13.197 | 13.132 |
| model_1_forward_end | [1, 1023, 151936] | 13.681 | 13.812 | 13.686 |
| model_0_forward_end | [1, 1, 151936] | 14.160 | 14.223 | 14.179 |
| after_prefill_clean_cache_cleared | 1024 | 12.841 | 14.223 | 14.179 |
| after_import_both_caches_live | 1024 | 12.841 | 14.223 | 14.179 |
| after_drop_hf_handoff | 1024 | 12.732 | 14.223 | 14.179 |
| after_decode_release | 1024 | 12.732 | 14.223 | 14.179 |
| before_prefill | 2048 | 12.732 | 14.223 | 14.179 |
| model_0_forward_end | [1, 2047, 151936] | 13.530 | 14.805 | 14.179 |
| model_1_forward_end | [1, 2047, 151936] | 14.611 | 15.385 | 14.620 |
| model_0_forward_end | [1, 1, 151936] | 15.564 | 15.775 | 15.602 |
| after_prefill_clean_cache_cleared | 2048 | 12.951 | 15.775 | 15.602 |
| after_import_both_caches_live | 2048 | 12.951 | 15.775 | 15.602 |
| after_drop_hf_handoff | 2048 | 12.732 | 15.775 | 15.602 |
| after_decode_release | 2048 | 12.732 | 15.775 | 15.602 |
| before_prefill | 2048 | 12.732 | 15.775 | 15.602 |
| model_0_forward_end | [1, 2047, 151936] | 13.530 | 15.775 | 15.602 |
| model_1_forward_end | [1, 2047, 151936] | 14.611 | 15.775 | 15.602 |
| model_0_forward_end | [1, 1, 151936] | 15.564 | 15.775 | 15.603 |
| after_prefill_clean_cache_cleared | 2048 | 12.951 | 15.775 | 15.603 |
| after_import_both_caches_live | 2048 | 12.951 | 15.775 | 15.603 |
| after_drop_hf_handoff | 2048 | 12.732 | 15.775 | 15.603 |
| after_decode_release | 2048 | 12.732 | 15.775 | 15.603 |
| after_empty_cache |  | 12.732 | 12.807 | 15.603 |

## Scheduled arrivals: empirical latency

24 requests/run, 3 repetitions/case, 72 requests per row (432 total). Inputs 128/512/1024/2048; outputs 16/32/64/128; EOS disabled. Local arrival replay, no HTTP. Quantiles pool the three repetitions.

| Arrival gap ms | Prefills/round | Throughput tok/s | Queue P99 ms | TTFT P50/P95/P99 ms | Request TPOT P50/P95/P99 ms | Inter-token gap P50/P95/P99 ms |
|---|---|---|---|---|---|---|
| 0 | fill | 266.0 | 4802.2 | 2500.12 / 4802.23 / 5061.75 | 19.84 / 44.84 / 95.52 | 3.47 / 164.45 / 426.63 |
| 0 | 1 | 265.4 | 4816.8 | 2513.06 / 4809.78 / 5076.19 | 19.86 / 54.89 / 94.76 | 3.48 / 188.01 / 265.27 |
| 100 | fill | 267.7 | 2483.8 | 1348.16 / 2591.94 / 2743.36 | 19.76 / 44.67 / 94.90 | 3.45 / 163.32 / 427.34 |
| 100 | 1 | 266.7 | 2498.3 | 1352.02 / 2594.95 / 2758.27 | 19.78 / 54.62 / 94.98 | 3.48 / 187.11 / 263.91 |
| 350 | fill | 171.6 | 2.2 | 183.73 / 262.41 / 267.76 | 2.67 / 6.86 / 7.31 | 2.61 / 2.74 / 3.11 |
| 350 | 1 | 171.6 | 2.3 | 183.33 / 262.41 / 263.49 | 2.66 / 6.86 / 6.88 | 2.60 / 2.73 / 3.13 |

TPOT = (last token - first token)/(output count - 1) per request. Inter-token gaps pool every adjacent token interval; these are different distributions. Throughput includes arrival idle time. Full request/token timestamps and E2E percentiles remain in scheduling.json.

## Nano scatter audit with isolated producer

Five repetitions/case; graph decode, profiler disabled. Handoff excludes official prefill; E2E includes it and local CUDA IPC. Historical measured values are retained.

### Free greedy: 32 outputs/request

| Input | Batch | Handoff ms | Decode tok/s |
|---|---|---|---|
| 1024 | 1 | 71.36 | 440.7 |
| 1024 | 2 | 81.95 | 774.8 |
| 1024 | 4 | 92.66 | 1396.1 |
| 1024 | 8 | 109.58 | 2428.8 |
| 2048 | 1 | 76.48 | 411.0 |
| 2048 | 2 | 92.29 | 686.2 |
| 2048 | 4 | 107.29 | 1199.0 |
| 2048 | 8 | 143.07 | 1831.2 |

### Matched history: 32 outputs/request

| Input | Batch | Handoff ms | Decode tok/s |
|---|---|---|---|
| 1024 | 1 | 72.05 | 436.2 |
| 1024 | 2 | 83.01 | 764.4 |
| 1024 | 4 | 94.26 | 1369.7 |
| 1024 | 8 | 112.56 | 2364.2 |
| 2048 | 1 | 77.08 | 407.6 |
| 2048 | 2 | 93.22 | 679.6 |
| 2048 | 4 | 108.85 | 1181.1 |
| 2048 | 8 | 145.88 | 1792.2 |

| Fresh E2E input | Nano ms |
|---|---|
| 1024 | 244.77 |
| 2048 | 331.19 |


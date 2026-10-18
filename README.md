# C2C-Decode

[中文版](README.zh-CN.md)

An inference systems adaptation of [official C2C](https://github.com/thu-nics/C2C)
to [nano-vLLM](https://github.com/GeeeekExplorer/nano-vllm) paged decode. Focus:
KV handoff, paging, admission, variable-length batches and measured correctness.
We reuse pretrained models and the published Fuser; we do not train or invent C2C.

Exact pair: `Qwen/Qwen3-4B` → `Qwen/Qwen3-0.6B`, with
`nics-efc/C2C_Fuser/qwen3_0.6b+qwen3_4b_Fuser/final`. Base and other Qwen variants
are not interchangeable. Upstream commits are in `upstream.lock.json`; model
revisions and resolved paths are recorded by `scripts/download_models.py`.

## System and measured results

The system reuses official prefill/fusion and imports receiver fused KV exactly into nano-vLLM's paged pool, with variable-length batched decode, capacity reservation, reclamation and optional CUDA Graph. CLI/serving defaults to lean prefix logits and layer-wise scatter; chunked prefill and input buffers remain optional.

The loopback HTTP/SSE service supports bounded queueing, token events, EOS, cancellation and cooperative timeouts. Architecture, first-token handoff and reproduction are described together in [system design](docs/design.md).

- Matched-work B1 controls show 9.2–10.0× faster Graph decode for 1K/2K inputs and 3.0–3.7× faster E2E for 32-token outputs; see [efficiency verification](docs/efficiency-verification.md).
- Scatter import is 3.5–7.2× faster than page-loop copying; lean logits lowers the separately measured 2K producer-only peak by about 1.16GiB. Configurations, timing boundaries and limitations are in [performance and correctness](docs/results.md) and [profiling](docs/profiling-results.md).
- In the measured burst workload, 1024-token chunks reduce token-gap P99 by 79.0%, at the cost of 5.2% lower throughput and 5.8% higher TTFT P99. Chunking is off by default.
- KV import is lossless; 37 tests and a GPU handoff/decode smoke check pass. Expanded tasks retain BF16 threshold failures and top-1 differences, so universal output equivalence is not claimed. See [validation](docs/validation.md).

## Run

Use a dedicated Linux CUDA environment. The current validation target is one
RTX 4090 24 GB, 16 allocated CPU cores, 120 GiB RAM and a 100 GB data disk.
The tested environment and measured results are described under `results/`.
Do not install into the local Windows base environment.

```bash
# Run at project root in the dedicated environment:
python scripts/bootstrap_upstream.py
python -m pip install -e .
# Also install compatible flash-attn and Triton.
export HF_HOME=/path/on/data-disk/huggingface
python scripts/download_models.py
python -m pytest -q

python -m c2c_decode.evaluate reference --output results/reference.json
python -m c2c_decode.evaluate validate --tokens 16 --pool-mib 2048 \
  --output results/validation-eager.json
python -m c2c_decode.evaluate benchmark --graph --tokens 32 \
  --pool-mib 2048 --repeats 3 --output results/benchmark-graph.json
python -m c2c_decode.evaluate baseline --tokens 32 \
  --output results/baseline.json
python scripts/summarize.py results/benchmark-graph.json \
  --baseline results/baseline.json

# Matched-work interleaved B1 audit, five pairs per case:
python -m c2c_decode.efficiency --repeats 5 \
  --output results/efficiency-audit.json

# Text generation with EOS handling:
python -m c2c_decode.cli --graph --prompt "用一句话解释 KV Cache" \
  --prompt "What is 17 times 19?"
```

The benchmark varies decode concurrency 1/2/4/8 with 1K/2K nominal input lengths;
requests in each batch intentionally have different lengths. All output is greedy
BF16, total length capped at 4K. JSON reports include revisions, versions, timings,
memory, generated IDs and diagnostics. Do not claim performance before inspecting
the actual reports. A nonzero validation exit means a correctness gate failed.

## Layout

| File | Responsibility |
|---|---|
| `src/c2c_decode/reference.py` | Load official weights; call and instrument official fusion |
| `src/c2c_decode/paged.py` | Fixed pool, lossless import, nano-vLLM decode, block lifetime |
| `src/c2c_decode/scheduler.py` | FIFO admission, capacity reservation, mixed completion |
| `src/c2c_decode/evaluate.py` | Official smoke, numerical/greedy checks, benchmark matrix |
| `src/c2c_decode/efficiency.py` | Matched fused KV/history, HF attention and CUDA Graph controls |
| `src/c2c_decode/bridge.py` | Original-environment C2C producer and same-GPU CUDA IPC |
| `src/c2c_decode/nano_benchmark.py` | Nano audit harness with an isolated official producer |
| `src/c2c_decode/optimized.py` | Reduced prefix logits, clean-prefix chunking, input staging buffers |
| `src/c2c_decode/serving.py`, `server.py` | One GPU owner, bounded dynamic admission, cancellation, deadlines and loopback SSE |
| `tests/` | Admission, EOS, exception cleanup and reuse tests |
| `scripts/profile_*.py`, `scripts/profiling_utils.py` | Removable trace/timing instrumentation, import A/B, memory and arrival replay |
| `scripts/summarize_profiling.py` | Validated data tables, timeline and performance figures |
| `docs/design.md` | Positions, first-token handoff and metric definitions |

Upstream code is checked out separately and not counted as new core code.
Original upstream licenses remain in each checkout (C2C Apache-2.0 repository
license; nano-vLLM MIT). Model and Fuser licenses remain with their repositories.
The system serializes GPU work, with optional clean-prefill chunks interleaved
with decode. The nano demo has a loopback HTTP endpoint; it has no prefix sharing,
KV eviction/preemption or multi-GPU serving.

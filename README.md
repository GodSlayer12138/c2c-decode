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

The first RTX 4090 implementation and measurements are complete. See
[measured results and limitations](docs/results.md) and [raw benchmark summary](results/summary.md).
The [matched-work efficiency audit](docs/efficiency-verification.md) verifies
9.2–10.0x warm B1 decode speedup and 3.0–3.7x E2E speedup for the measured cases,
with interleaved repetitions and eager/FlashAttention-2 controls.
KV import is lossless. The initial BF16 numerical suite passed; expanded task
checks expose numerical-threshold failures and top-1 differences versus the official path. Use
`validate --strict-tokens` to enforce strict token equivalence.

The [profiling and optimization study](docs/profiling-results.md) adds real CPU/CUDA
traces, operator analysis, memory snapshots and 432 scheduled-arrival requests.
The nano importer now defaults to layer-wise scatter: 3.5–7.2x faster import in
40 paired measurements, with identical generated tokens between import modes.
Historical reports above retain the old per-page importer; use the refreshed
nano audit in `results/profiling/summary.md` for the optimized implementation.

The [second optimization study](docs/next-improvements.md) adds reduced prefix
logits, optional chunked prefill, reusable decode input buffers, and a loopback
HTTP/SSE demo with bounded admission, cancellation and deadlines. CLI/server
default to reduced logits; `--official-prefill` restores the CLI reference path.
Chunking and input buffers remain opt-in. Expanded authored task checks preserve
the measured answers but expose additional backend numerical-threshold failures;
these remain visible in the report rather than relaxing the tolerances.

The current nano-only package passes 37 tests and a real GPU handoff/decode smoke
check; see [current validation](docs/validation.md). This regression check does
not replace the historical performance or expanded numerical-quality results.

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

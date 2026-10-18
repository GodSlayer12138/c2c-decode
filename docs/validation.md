# Current Nano Validation

[中文版](validation.zh-CN.md)

2026-10-18, using the existing RTX 4090 and dedicated `c2c-decode` environment.

- `compileall` passed; `pytest -q`: **37 passed in 4.90s**.
- `profile_runtime.py --help` and `quality_eval.py --help` load the current nano entry points successfully.
- `python -m c2c_decode.nano_benchmark --smoke --output results/cleanup/nano-smoke.json` ran the official producer, fused-KV import and nano CUDA Graph decode.
- The 257-token case checks logits over 15 matched-history steps: exact KV, forced history, numerical thresholds and top-1 checks passed in this run. Additional 1K/B1, 32-token free/forced runs and fresh-request E2E completed.
- All smoke-test pages were reclaimed. Device memory returned to 0 MiB and utilization to 0% after exit.
- Both summary scripts regenerated tables and plots from retained nano measurements.

This page records functional regression checks. Expanded-task stepwise numerical checks are in `results/next/numerical-failures.json`.

Evidence: [test output](../results/cleanup/tests.txt), [raw GPU smoke results](../results/cleanup/nano-smoke.json).

`measurement-provenance.json` records retained-module hashes from historical measurements. `summary-source-sha256.json` records source at summary generation. Raw nano timing, outputs and traces remain unchanged, including historical path/argument fields. The current benchmark entry is `nano_benchmark.py`.

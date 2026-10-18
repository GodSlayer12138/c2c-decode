#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p results
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export TOKENIZERS_PARALLELISM=false
export TORCHINDUCTOR_CACHE_DIR="$PWD/.cache/inductor"
export TRITON_CACHE_DIR="$PWD/.cache/triton"
python -m pytest -q tests | tee results/tests.txt
python -m c2c_decode.evaluate reference --output results/reference.json
python -m c2c_decode.evaluate validate --tokens 16 --pool-mib 2048 --output results/validation-eager.json
python -m c2c_decode.evaluate validate --graph --tokens 16 --pool-mib 2048 --output results/validation-graph.json
python -m c2c_decode.evaluate baseline --tokens 32 --repeats 3 --output results/baseline.json
python -m c2c_decode.evaluate benchmark --tokens 32 --pool-mib 2048 --repeats 3 --output results/benchmark-eager.json
python -m c2c_decode.evaluate benchmark --graph --tokens 32 --pool-mib 2048 --repeats 3 --output results/benchmark-graph.json
python scripts/summarize.py results/benchmark-graph.json --baseline results/baseline.json

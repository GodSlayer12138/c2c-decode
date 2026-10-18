# Environment record

[中文版](environment.zh-CN.md)

Observed on 2026-10-17 (Asia/Shanghai). No local base Python environment was used.

Local workspace: `D:\code\C2C`. WSL2 Ubuntu 24.04 is installed but was stopped;
the local RTX 4060 Laptop 8 GB is not used for this experiment.

Remote project: `/root/autodl-tmp/c2c-decode`.
Dedicated conda environment: `/root/autodl-tmp/envs/c2c-decode`.
Model cache: `/root/autodl-tmp/huggingface`.

The remote environment was cloned from the preinstalled AutoDL environment
without modifying its base. Transformers was changed from 4.57.1 to 4.52.4 to
match C2C's pinned DynamicCache API. PyTorch 2.8.0+cu128 and FlashAttention 2.8.3
were retained from the image. This differs from the official C2C package's
torch==2.6.0 requirement. Reference and adapted paths run under the same recorded
runtime; do not represent them as an exact reproduction of the authors' environment.

| Resource | Observed |
|---|---|
| OS | Ubuntu 22.04.5 |
| GPU | One RTX 4090, 24,564 MiB |
| Driver | 580.105.08 |
| PyTorch CUDA runtime | 12.8 |
| Python | 3.12.3 |
| CPU quota | 16 cores; host Xeon Platinum 8358P |
| Memory cgroup limit | 128,849,018,880 bytes (120 GiB) |
| Data disk | 100 GB, initially empty |
| System disk | 30 GB, initially ~12 GB free |
| nvcc | Not on PATH; no extension compilation assumed |

Driver-reported CUDA 13.0 is its supported runtime ceiling, not the installed
toolkit. BF16 matrix multiplication and FlashAttention causal kernels were
executed successfully before implementation. The large host RAM/CPU figures in
`free`/`lscpu` are not the container's allocated resources; cgroup quotas above
are the relevant limits.

Direct Hugging Face access initially timed out. The AutoDL-provided
`/etc/network_turbo` shell setup enabled access to official Hugging Face endpoints.
Downloads resume in the data-disk cache and record immutable repository revisions.
No SSH password or access token is stored in project source or reports.

The cloned image contains unrelated training packages (for example TRL) whose
declared Transformers requirement differs from 4.52.4. The runtime imports only
the inference dependencies used here. For a clean installation, use
`environment.yml`, then a compatible FlashAttention wheel, then `pip install -e .`.

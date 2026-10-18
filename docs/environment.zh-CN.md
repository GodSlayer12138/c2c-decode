# 环境记录

[English](environment.md)

观测日期为 2026-10-17（Asia/Shanghai），未使用本地 base Python 环境。

本地项目：`D:\code\C2C`。已安装 WSL2 Ubuntu 24.04，但当时处于停止状态；本地 RTX 4060 Laptop 8GB 未用于实验。

服务器项目：`/root/autodl-tmp/c2c-decode`；专用 conda 环境：`/root/autodl-tmp/envs/c2c-decode`；模型缓存：`/root/autodl-tmp/huggingface`。

远程环境从 AutoDL 预装环境克隆，没有修改 base。为匹配 C2C 的 DynamicCache API，将 Transformers 从 4.57.1 改为 4.52.4；保留镜像中的 PyTorch 2.8.0+cu128 和 FlashAttention 2.8.3。这不同于官方声明的 torch==2.6.0。参考与适配路径使用同一记录环境，不应描述为完全复刻作者运行环境。

| 资源 | 观测值 |
|---|---|
| 操作系统 | Ubuntu 22.04.5 |
| GPU | 单张 RTX 4090，24,564MiB |
| 驱动 | 580.105.08 |
| PyTorch CUDA 运行时 | 12.8 |
| Python | 3.12.3 |
| CPU 配额 | 16 核；宿主机 Xeon Platinum 8358P |
| 内存 cgroup 上限 | 128,849,018,880 bytes（120GiB） |
| 数据盘 | 100GB，初始为空 |
| 系统盘 | 30GB，初始约 12GB 可用 |
| nvcc | 不在 PATH 中，未假设能现场编译扩展 |

驱动显示的 CUDA 13.0 是支持的运行时上限，不代表已安装该工具链。实现前已成功执行 BF16 矩阵乘法和 FlashAttention 因果内核。`free`/`lscpu` 的宿主机总资源不能当作容器可用配额，应使用上表 cgroup 限额。

起初直接访问 Hugging Face 超时，AutoDL 的 `/etc/network_turbo` 配置使官方端点可访问。下载在数据盘缓存中续传，并记录不可变 revision。源码与报告不保存 SSH 密码或访问 token。

克隆镜像包含 TRL 等无关训练包，其声明的 Transformers 依赖与 4.52.4 不同；运行时仅导入所需推理依赖。干净安装可从 `environment.yml` 开始，安装兼容 FlashAttention wheel，再执行 `pip install -e .`。

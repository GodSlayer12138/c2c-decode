# C2C-Decode

[English](README.md)

将[官方 C2C](https://github.com/thu-nics/C2C) 融合状态适配到 [nano-vLLM](https://github.com/GeeeekExplorer/nano-vllm) 分页解码，重点展示 KV 交接、分页管理、准入、变长批处理与实测正确性。复用预训练模型和官方 Fuser，不训练模型，也不宣称发明 C2C。

模型固定为 `Qwen/Qwen3-4B` → `Qwen/Qwen3-0.6B`，融合权重为 `nics-efc/C2C_Fuser/qwen3_0.6b+qwen3_4b_Fuser/final`，不能任意替换 Base 或其他 Qwen 变体。上游 commit 见 `upstream.lock.json`，模型 revision 与路径由 `scripts/download_models.py` 记录。

## 系统实现与实测效果

系统复用官方 prefill 和融合，将接收端融合 KV 精确导入 nano-vLLM 分页池，支持变长批量解码、容量预留、页面回收及可选 CUDA Graph。CLI/服务默认精简 prefix logits 和逐层 scatter 导入；分块 prefill、输入缓冲复用保持可选。

本机 HTTP/SSE 服务提供有界排队、逐 token 输出、EOS、取消和协作式超时。核心设计、首 token 交接及运行方式统一见 [系统设计](docs/design.zh-CN.md)。

- 同工作量 B1 对照中，1K/2K 输入的 Graph decode 加速 9.2–10.0 倍，32-token 请求端到端加速 3.0–3.7 倍；详见 [效率验证](docs/efficiency-verification.zh-CN.md)。
- Scatter 相对逐页导入加速 3.5–7.2 倍；精简 logits 在生产端单独测量时降低 2K 峰值约 1.16GiB。各实验的配置、计时边界与适用范围见 [性能与正确性评估](docs/results.zh-CN.md) 和 [性能分析](docs/profiling-results.zh-CN.md)。
- 1024-token 分块在所测突发负载中降低 token 间隔 P99 79.0%，代价是吞吐下降 5.2%、TTFT P99 上升 5.8%，默认关闭。
- KV 导入无损；37 项测试与 GPU 交接/解码冒烟通过。扩展任务仍有 BF16 数值门槛失败和 top-1 差异，不宣称普遍输出等价；见 [验证记录](docs/validation.zh-CN.md)。

## 运行

使用专用 Linux CUDA 环境；实测资源为单张 RTX 4090 24GB、16 核配额、120GiB RAM、100GB 数据盘。环境与测量记录见 `results/`。不要安装到本地 Windows base 环境。

```bash
# 在专用环境的项目根目录执行
python scripts/bootstrap_upstream.py
python -m pip install -e .
# 另安装匹配 PyTorch 的依赖
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

# 同工作量交错 B1 审计，每档五组
python -m c2c_decode.efficiency --repeats 5 \
  --output results/efficiency-audit.json

# 实际文本生成，处理 EOS
python -m c2c_decode.cli --graph --prompt "用一句话解释 KV Cache" \
  --prompt "What is 17 times 19?"
```

基准覆盖 1K/2K 名义输入、1/2/4/8 解码并发，同批请求故意采用不同长度。使用 greedy BF16，总长度上限 4K。JSON 包含 revision、依赖版本、计时、显存、输出 IDs 和诊断；性能结论应以实际记录为准。验证入口非零退出表示正确性门槛失败。

## 文件职责

| 文件 | 职责 |
|---|---|
| `src/c2c_decode/reference.py` | 加载官方权重，调用并计时官方融合 |
| `paged.py` | 固定池、无损导入、nano 解码、页面生命周期 |
| `scheduler.py` | FIFO 准入、容量预留、不同完成时刻的处理 |
| `evaluate.py` | 官方冒烟、数值/greedy 检查、基准矩阵 |
| `efficiency.py` | 相同融合 KV/历史，HF attention 与图执行对照 |
| `bridge.py` | 独立官方生产端与同卡 CUDA IPC |
| `nano_benchmark.py` | 独立生产端的 nano 审计入口 |
| `optimized.py` | 精简 logits、干净 prefix 分块、输入暂存缓冲区 |
| `serving.py`、`server.py` | GPU 单一执行线程、动态有界准入、取消、超时、本机 SSE |
| `tests/` | 准入、EOS、异常清理与复用测试 |
| `scripts/profile_*.py`、`profiling_utils.py` | 可移除计时/trace、导入 A/B、显存与到达回放 |
| `scripts/summarize_profiling.py` | 校验后的数据表、时间线和性能图 |
| `docs/design.md` | 位置、首 token 交接、指标定义 |

表中简写的 Python 文件均位于 `src/c2c_decode/`。上游另行检出，不计入新增核心代码；C2C 仓库 Apache-2.0、nano-vLLM MIT 及模型/Fuser 原许可证仍适用。GPU 工作串行，可选在干净 prefill 块间插入 decode；本机演示有 HTTP 入口，尚无前缀共享、KV 驱逐/抢占或多 GPU 服务。

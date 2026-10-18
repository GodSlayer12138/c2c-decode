# C2C-Decode

[English](README.md)

将[官方 C2C](https://github.com/thu-nics/C2C) 融合状态适配到 [nano-vLLM](https://github.com/GeeeekExplorer/nano-vllm) 分页解码，重点展示 KV 交接、分页管理、准入、变长批处理与实测正确性。复用预训练模型和官方 Fuser，不训练模型，也不宣称发明 C2C。

模型固定为 `Qwen/Qwen3-4B` → `Qwen/Qwen3-0.6B`，融合权重为 `nics-efc/C2C_Fuser/qwen3_0.6b+qwen3_4b_Fuser/final`，不能任意替换 Base 或其他 Qwen 变体。上游 commit 见 `upstream.lock.json`，模型 revision 与路径由 `scripts/download_models.py` 记录。

已在 RTX 4090 完成实现和测量：[首轮结果与限制](docs/results.zh-CN.md)、[原始汇总](results/summary.zh-CN.md)。[同工作量效率审计](docs/efficiency-verification.zh-CN.md)使用交错重复及 eager/FlashAttention-2 对照，所测 B1 热启动 decode 加速 9.2–10.0 倍，端到端 3.0–3.7 倍。KV 导入无损；初始 BF16 数值集通过，但扩展任务暴露阈值失败和 top-1 差异，不能声称普遍等价。`validate --strict-tokens` 可执行严格 token 检查。

[Profiling 与优化](docs/profiling-results.zh-CN.md)包含真实 CPU/CUDA trace、算子分析、显存快照与 432 个到达回放请求。当前默认逐层 scatter 导入，40 组配对中的导入加速为 3.5–7.2 倍，优化前后生成 token 一致。早期文档保留旧逐页导入数据，当前实现参考 `results/profiling/summary.md`。

[第二轮优化](docs/next-improvements.zh-CN.md)增加精简 prefix logits、可选分块 prefill、输入缓冲区复用，以及仅本机访问的 HTTP/SSE 服务，支持有界准入、取消与 deadline。CLI/server 默认精简 logits，CLI 的 `--official-prefill` 恢复原路径；分块和缓冲区复用默认关闭。扩展任务中答案保持一致，但数值门槛失败仍如实保留。

当前 nano 项目通过 37 项测试与真实 GPU 交接/解码冒烟检查，见[当前验证](docs/validation.zh-CN.md)。这不替代历史性能或扩展数值评测。

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

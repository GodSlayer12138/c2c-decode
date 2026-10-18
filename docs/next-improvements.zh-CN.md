# 第二轮优化：prefill、调度、服务与质量

[English](next-improvements.md)

本轮固定原 Qwen3-4B → Qwen3-0.6B、官方 Fuser、BF16、RTX 4090，未训练或量化。
性能数据均关闭 PyTorch Profiler；原始记录与自动汇总位于 `results/next/`。

## 已实现的路径

`OfficialC2C` 保留原始正确性参考。新增 `OptimizedC2C` 在限定作用域内将两个模型的
`logits_to_keep` 设为 1，返回后恢复原方法。官方外层虽有该参数，但没有传给内部模型，
所以不能只修改外层调用。接收模型末 prompt token 的 logits 仍完整计算；融合公式、
干净缓存克隆、RoPE、首个生成 token 和分页导入没有变化。

CLI 和新服务默认使用这个精简路径；CLI 的 `--official-prefill` 可恢复原始路径。
原有验证/benchmark 入口仍保留官方参考，不能将历史数据误认为新默认的测量。

分块路径 `prefill_steps` 分别计算两模型的**干净** prefix KV，在块间让出 GPU 给 decode。
prefix 完整后，把这两份输出交回原版 Rosetta forward，由它执行原有克隆、融合和最后
prompt token。没有提前融合某个 chunk，再把融合缓存作为后续干净 prefill 的输入。
位置与 attention mask 使用绝对 token 位置；融合仍是一次不可分割的执行区间。
分块改变 attention 的计算形状，因此长输入不保证逐元素 KV 相等。

`PagedDecoder(fast_inputs=True)` 使用两个固定 pinned CPU staging buffer 和对应 GPU
buffer，复用内存并减少 H2D 调用；CUDA event 在 CPU 再次写入前确认上次拷贝完成。
slot mapping 保持连续，遵守上游 Triton kernel 的布局要求。它仍为可选项。

新增 `ServingEngine` 由单个线程持有 GPU 执行权；支持动态入队、最大等待队列、完整
输出长度的 KV 页预留、EOS、取消、超时、逐 token 事件，以及异常/关闭时释放所有页。
取消和超时是协作式的，在 GPU 工作单元之间检查，不会强行中断正在执行的 kernel。
例如 HTTP 检查中的 1ms deadline 实际在约 123ms 后返回 timeout，不能宣传为硬实时超时。

## 生产端与 CPU 优化的结论

1K/2K、五次随机顺序配对中，生产端完整路径与精简 logits 的中位数分别为：

| 输入 | 官方 prefill | 精简 logits | 耗时下降 |
|---:|---:|---:|---:|
| 1024 | 156.70ms | 152.79ms | 2.5% |
| 2048 | 233.83ms | 219.99ms | 5.9% |

这组显存实验只包含官方生产端模型和 Fuser，**没有 nano 接收模型或分页池**。
2K 的 allocated 峰值 **12.469 → 11.309GiB**，降低约 **1.16GiB**。
六组边界/长输入的精简路径融合 KV 逐元素相同，末 prompt logits 的 RMSE 为 0。
它的性能收益不能与前轮导入优化的 26.4% 简单相加。

输入缓冲区复用的独立配对实验中，1K/B1 导入＋解码中位数 **71.603 → 70.918ms**，
2K/B8 **144.430 → 143.307ms**。257/B4 的五组配对中有一组变慢，整体收益约 1%，
因此不作为默认。所有 15 组配对生成 token 相同。

分块单请求 prefill 明显变慢：2K 的 256/512/1024-token chunk 分别约
**514.95/324.34/258.10ms**。它针对解码停顿，不应宣传为更快的单请求 prefill。
完整新请求端到端、随机到达负载、最长停顿和 SLO 数据见自动汇总。

## 完整链路效果与默认选择

新请求每次重新执行 prefill、融合、导入和 decode。每档五次，模式顺序随机；这是
单进程 nano 的计时，不包含 IPC、HTTP、分词、模型加载或编译，也不与历史 IPC 数据直接相减。

| 输入 / 并发 / 每请求输出 | 原始生产端 E2E | 精简 logits E2E | 耗时下降 |
|---|---:|---:|---:|
| 1K / 1 / 32 | 228.49ms | 225.95ms | 1.1% |
| 1K / 1 / 128 | 450.65ms | 445.61ms | 1.1% |
| 1K / 8 / 32 | 1362.67ms | 1325.72ms | 2.7% |
| 2K / 1 / 32 | 311.56ms | 297.64ms | 4.5% |
| 2K / 1 / 128 | 547.21ms | 532.76ms | 2.6% |
| 2K / 8 / 32 | 2009.88ms | 1899.04ms | 5.5% |

120 条端到端记录中，原始生产端、精简 logits、精简＋输入复用的 token 历史均一致。
在 2K/B8 下，额外开启输入复用为 1900.09ms，略慢于 1899.04ms，进一步说明其收益
很小，不宜作为稳定的端到端提速宣传。因此 CLI/服务只默认开启精简 logits。

分块方案包括两模型及融合阶段之间的调度让出，不把效果全部归因于 token chunk 大小。
以下两行都开启精简 logits 和输入缓冲区复用，只比较是否分块；为突发到达、三次回放
共 72 请求的合并分位数，吞吐取各次中位数：

| 策略 | token 间隔 P99 | 最长 token 间隔 | TTFT P99 | 吞吐 |
|---|---:|---:|---:|---:|
| 完整 prefill | 388.24ms | 980.81ms | 4002.99ms | 347.38 tok/s |
| 1024-token chunk | 81.51ms | 83.00ms | 4233.56ms | 329.18 tok/s |

P99 token 间隔降低 **79.0%**，代价是吞吐下降 **5.2%**、TTFT P99 上升 **5.8%**。
在平均到达间隔 350ms 的样本中，固定 SLO 达标请求由 36/72 提升到 72/72；
这只是该有限样本的观察，不是服务承诺。256-token chunk 的突发吞吐仅 243.53 tok/s，
分得更细并不自动更好。分块默认关闭，用户可按停顿与等待时间偏好选择。

37 项测试通过，所有 54 次回放结束后页池恢复到 73 页。
单独的 120.4 秒压力运行提交 589 请求：458 正常完成、85 取消、46 超时，0 引擎错误。
开始与结束的 live allocated 均为 13,678,994,432 bytes，页全部回收。
HTTP 故意把等待队列设为 2，用 4 个客户端线程发起 8 次调用，3 次完成、5 次以 429 拒绝；
另外验证了 DELETE 取消、超时事件、客户端断开和健康检查。

![本轮实测概览](../results/next/overview.png)

[可缩放矢量图](../results/next/overview.svg)

[完整自动汇总](../results/next/summary.zh-CN.md)保留所有配置与数值失败统计，原始 JSON 位于同目录。

## 扩展正确性与任务质量

新增 36 个自编功能样例：10 个数学、10 个问答、10 个 Python 输出推理、6 个长上下文
检索。保留题目、期望答案、官方输出、各后端输出和 token IDs；不执行模型生成代码。
这不是 GSM8K/HumanEval 等标准评测，也不能代表整体模型能力。

原始自动评分是严格的规范化答案匹配，额外解释也算不匹配。官方和各适配路径都是
**7/36**；这个分数同时衡量答案内容和格式遵循，不能直接解释成只有七道题内容正确。
例如官方回答 `The answer is 323.` 会因未遵循“只返回答案”而被严格评分判错。
同时也确实存在内容错误，例如官方将 `3*x+7=28` 回答为 `x=3`。

nano 原生产端、精简 logits、512-token 分块均有 **35/36** 个样例
与官方输出逐 token 相同。剩余样例是 FIFO 问题的逗号/连字符不同，对应官方 logits
的 top-1/top-2 margin 为 **0**。该题官方本身也没有按要求回答。各适配路径的回答
彼此一致；本轮没有观察到新增答案退化，但不据此宣称通用质量等价。

扩展的逐步 logits 检查**并非全部通过旧门槛**，不修改 cosine≥0.999、RMSE≤0.15、
max_abs≤1 的标准：nano 在 `math-7` 的一个有效生成步骤 cosine 约 **0.998749**。
nano 的三种生产端在该
失败步骤表现相同，属于既有后端数值差异，不是本轮 logits 优化引入。
512-token 分块还在 `context-0` 新增了一个有效生成步骤的门槛失败；即使本次输出
没有变化，也不能将该配置描述为数值验证全面通过。因此所有分块选项均保持实验状态。
另外一些失败出现在强制生成越过 EOS 后；报告同时保留全部检查和 EOS 前/处的分类。
详情见自动汇总和 `numerical-failures.json`，不将它们重新归类为“通过”。

针对上述分块风险，补测了 1024-token 分块的 6 个长上下文样例：6/6 输出与官方
逐 token 相同，所检查步骤没有触发上述数值门槛，缓存导入精确且页面全部回收。
这只是针对已有长上下文样例的补充检查，并非独立的完整质量评测；1024-token
分块仍保持实验状态、默认关闭。

## 本地服务与复现

使用既有专用 Linux CUDA 环境，在服务器项目目录执行：

```bash
export OMP_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false
export TORCHINDUCTOR_CACHE_DIR=$PWD/.cache/inductor
export TRITON_CACHE_DIR=$PWD/.cache/triton
export MPLCONFIGDIR=$PWD/.cache/matplotlib
PY=/root/autodl-tmp/envs/c2c-decode/bin/python

$PY -m pytest -q
$PY scripts/next_experiments.py --section producer
$PY scripts/next_experiments.py --section cpu
$PY scripts/quality_eval.py
$PY scripts/quality_eval.py --chunk-size 1024 --context-only \
  --output results/next/quality-nano-chunk1024-context.json
$PY scripts/service_experiments.py
$PY scripts/next_experiments.py --section e2e
$PY scripts/summarize_next.py

# 默认精简 logits，分块可选
$PY -m c2c_decode.server --port 8000
# 或
```

服务只绑定 `127.0.0.1`，提供 `GET /health`、`POST /generate`、
`DELETE /requests/<id>`。POST 返回 SSE，先发送 request ID，再发送 token ID、
累计文本和完成事件。累计文本按事件自身的 token index 解码，避免慢客户端看到未来 token。

```bash
curl -N http://127.0.0.1:8000/generate \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"Explain a KV cache briefly.","max_new_tokens":64,"timeout_s":30}'
```

队列满返回 429，非法输入返回 400/413，引擎失败或关闭返回 503。
它是本机演示服务，没有公开部署、认证、多 GPU 或 OpenAI API 兼容性承诺。
负载性能表是引擎内到达回放；HTTP 实验验证真实 socket/SSE 生命周期，不能将二者
混称为已经测得 HTTP 尾延迟。两分钟压力运行也不构成长期稳定性证明。

服务实验包含：六种配置 × 三档到达间隔 × 三个固定随机种子，每次 24 请求，共
1296 请求；同一 seed 的各配置使用相同到达、输入长度和输出长度。分位数来自每档
72 请求，仍是小样本。另做单独有界时长压力运行和真实 HTTP 生命周期检查。
示例 SLO 在运行前固定为 TTFT≤1000ms 且每个 token 间隔≤100ms，不根据结果调整。

当前核心代码约 1738 行（含空行），保持在原定规模内。新的评测脚本位于 `scripts/`，
上游模型、Fuser、attention kernel 和缓存分配器仍复用已有实现。

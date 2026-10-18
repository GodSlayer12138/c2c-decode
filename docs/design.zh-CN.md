# C2C-Decode 设计

[English](design.md)

本项目把已有 Cache-to-Cache 方法适配到小型分页解码后端，不提出或训练 C2C。官方 Rosetta 负责 prefill 与融合数学，nano-vLLM 负责 Qwen3、权重加载、RoPE、FlashAttention decode、页分配器及可选 CUDA Graph。

## 交接不变量

对于 N 个输入 token，官方例程先用 `[1,0]` 处理 N−1 个 token，再用 `[-1,0]` 处理一个 token。接收端和发送端 prefill 第一段，Fuser 修改接收端 KV，接收端基于融合状态处理最后一个 prompt token；项目调用完整官方 forward。

交接时每层已有 N 个缓存 token，最终 logits 预测首个生成 token t0。导入将 `[1,H,N,D]` 复制到 `[block,offset,H,D]`，不重复应用 RoPE。t0 已加入序列元数据，但尚未缓存。首次 nano decode 在位置 N 输入 t0，将其 KV 写到槽位 N，关注 N+1 个条目，并预测 t1。重复 t0、重放末 prompt 或从 N−1 开始都不正确。

默认逐层索引 scatter，把逻辑位置映射到物理页与页内 offset，支持非连续页和不满尾页；`PagedDecoder(import_mode="loop")` 保留逐页复制以供 A/B。两者只写有效槽位，并保留已完成 RoPE 的 KV。

当前 prefill 为无 padding 的 batch=1；decode 为每个活动请求打包一个 token，各自保留 position、页表和上下文长度。所选 Qwen3 配对不需 tokenizer 转换。固定官方 `generate` 会将终止 EOS 替换为 pad token；相应完成序列沿用该约定。

## 显存和调度

KV 池使用显式 MiB 预算，不按空闲显存比例占满。Qwen3-0.6B 的 28 层、8 KV heads、head dimension 128、BF16 对应每 token 114,688 bytes，每 256-token 页 28MiB。1024MiB 预算可容纳 36 页，实际 1008MiB；八个 2K prompt 加生成需要更多页，所以并发 1/2/4/8 基准使用 2048MiB 预算，共 73 页。

准入预留 `ceil((prompt+max_new_tokens-1)/256)` 页容量；物理页在 prompt 导入及 decode 跨页时实际分配。输出上限后的 token 不会继续喂回模型。预留避免活动请求相互耗尽池；FIFO 请求随着已有请求完成而获准进入。异常时释放所有存活序列。

请求可设置相对运行开始的非负 `arrival_ms`；按到达顺序准入，结果维持调用方顺序。每请求 `token_times_ms` 相对计划到达时刻计时。默认持续准入至容量上限；可选 `prefill_limit_per_round=1` 在每批 decode 前最多完成一次完整 prefill，它不是分块或并行执行，相关取舍见 profiling 文档。

禁用 prefix hash，因为仅 token 哈希不足以标识融合前缀。未来复用还需发送端/Fuser revision、输入、对齐、融合设置、精度与位置上下文。当前无抢占或 swap；GPU 工作串行，prefill 会阻塞活动 decode。可选服务循环在干净 prefix 块间插入 decode，并提供本机 SSE，限制见下文服务说明。

## 测量约定

- `prefill_ms`：两模型 prefix forward 的 CUDA event 时间之和。
- `fusion_ms`：发送端结束到末 prompt forward 开始，包括官方缓存克隆、融合 dispatch，并非仅投影 kernel。
- `last_prompt_token_ms`：官方末 prompt token forward。
- `fusion_and_overhead_ms`：完整 prefill wall time 减模型 event 区间，是较宽诊断余项，不能再与 fusion 相加。
- `official_prefill_total_ms`：完整官方路径的同步 wall time。
- `cache_import_ms`：校验、页分配、KV 复制和首 token 提取。
- `decode_ms`：同步批量 decode、greedy 采样与元数据准备。
- TTFT：从计划到达到首 token 和导入缓存就绪，包括 FIFO 排队及前序 prefill。
- E2E：从到达到最后生成 token。整次吞吐含首 token；decode 吞吐排除 prefill 已生成的首 token。
- 显存：所有常驻模型与池的 PyTorch allocated/reserved 峰值，不含全部驱动分配；`nvidia-smi` 单列。

计时不含 warmup，模型加载另记；参考冒烟显式调用官方 `generate`。数值对齐官方融合 continuation，而非未融合 Qwen。Teacher forcing 和自由 greedy 相等性回答不同问题。BF16 阈值测量前确定，失败如实保留；`validate --strict-tokens` 还会因任一同历史 top-1 差异返回失败。默认数值通过不等于通用 token 等价，JSON 分别记录结论。

## 范围

单 GPU、BF16、greedy、总长度上限 4096、prefill batch=1、decode batch≤8。CUDA Graph 可选，保留 eager 供调试。无训练、量化、张量并行或生产服务等价承诺。上游 serving 状态应随版本升级重新检查；当时检查的 README 仍为待发布公告。

## 已实现的路径

`OfficialC2C` 保留原始正确性参考。`OptimizedC2C` 在限定作用域内将两个模型的
`logits_to_keep` 设为 1，返回后恢复原方法。官方外层虽有该参数，但没有传给内部模型，
所以不能只修改外层调用。接收模型末 prompt token 的 logits 仍完整计算；融合公式、
干净缓存克隆、RoPE、首个生成 token 和分页导入没有变化。

CLI 和服务默认使用这个精简路径；CLI 的 `--official-prefill` 可恢复原始路径。
验证/benchmark 入口保留官方参考；结果应按所记录的生产端与导入配置解读。

分块路径 `prefill_steps` 分别计算两模型的**干净** prefix KV，在块间让出 GPU 给 decode。
prefix 完整后，把这两份输出交回原版 Rosetta forward，由它执行原有克隆、融合和最后
prompt token。没有提前融合某个 chunk，再把融合缓存作为后续干净 prefill 的输入。
位置与 attention mask 使用绝对 token 位置；融合仍是一次不可分割的执行区间。
分块改变 attention 的计算形状，因此长输入不保证逐元素 KV 相等。

`PagedDecoder(fast_inputs=True)` 使用两个固定 pinned CPU staging buffer 和对应 GPU
buffer，复用内存并减少 H2D 调用；CUDA event 在 CPU 再次写入前确认上次拷贝完成。
slot mapping 保持连续，遵守上游 Triton kernel 的布局要求。它仍为可选项。

`ServingEngine` 由单个线程持有 GPU 执行权；支持动态入队、最大等待队列、完整
输出长度的 KV 页预留、EOS、取消、超时、逐 token 事件，以及异常/关闭时释放所有页。
取消和超时是协作式的，在 GPU 工作单元之间检查，不会强行中断正在执行的 kernel。
例如 HTTP 检查中的 1ms deadline 实际在约 123ms 后返回 timeout，不能宣传为硬实时超时。


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
# 可选分块模式：
$PY -m c2c_decode.server --port 8000 --chunk-tokens 1024
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

当前核心代码约 1738 行（含空行），保持在原定规模内。评测脚本位于 `scripts/`，
上游模型、Fuser、attention kernel 和缓存分配器仍复用已有实现。

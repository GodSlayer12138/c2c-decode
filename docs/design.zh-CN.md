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

禁用 prefix hash，因为仅 token 哈希不足以标识融合前缀。未来复用还需发送端/Fuser revision、输入、对齐、融合设置、精度与位置上下文。当前无抢占或 swap；GPU 工作串行，prefill 会阻塞活动 decode。可选服务循环在干净 prefix 块间插入 decode，并提供本机 SSE，限制见 [第二轮优化](next-improvements.zh-CN.md)。

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

# C2C-Decode 效率验证

[English](efficiency-verification.md)

2026-10-17 在租用的单张 RTX 4090 24GB 上完成复测。结论：在本次
BF16、batch=1、1K/2K 输入的预热测试中，接入 nano-vLLM CUDA Graph 后，
decode 约快 9.2–10.0 倍；32 token 输出的端到端请求约快 3.0–3.7 倍。
这是相对当前固定版本的官方 C2C/Hugging Face 执行路径的结果。

## 对照方法

- 模型、Fuser 和上游版本沿用锁文件；不量化、不训练。
- 官方 prefill 和融合保持一致。每组 decode 从同一份融合 KV 开始，
  导入后逐元素回读一致；各路径使用完全相同的 token 历史。
- 四条路径：官方 SDPA、官方改用 FlashAttention-2 解码、分页 eager、
  分页 CUDA Graph。FlashAttention-2 对照只更换解码 attention，prefill 仍为 SDPA。
- 每组先预热，再按固定随机种子打乱运行顺序，做 5 次配对重复。
  共 80 条 decode 记录和 20 条端到端记录。
- 32/128 个输出 token 对应 31/127 次 decode；首个输出来自官方 prefill。
  所有路径每步均计算完整 logits 和 argmax，再强制喂入相同的后续 token。
  关闭 EOS 截断，防止实际工作量不一致；argmax 差异保留在原始记录中。
- decode 计时排除测试夹具克隆和 KV 导入。端到端重新执行官方 prefill、
  融合和末尾 prompt token，并计入导入及 decode；官方路径不克隆测试夹具。
- 计时前后同步 CUDA。四条路径在同一进程中交错运行，两份接收模型一直常驻。
  模型加载、编译和图捕获排除在请求延迟外，本次初始化合计约 50.65 秒。

## 结果

以下耗时是 5 次中位数，加速比是每次配对比值的中位数。

| 输入 / 输出 token | 官方 decode | Graph decode | decode 加速 | Graph 更快次数 |
|---|---:|---:|---:|---:|
| 1024 / 32 | 717.78 ms | 72.27 ms | 10.00× | 5/5 |
| 1024 / 128 | 2898.40 ms | 292.50 ms | 9.98× | 5/5 |
| 2048 / 32 | 711.54 ms | 77.37 ms | 9.21× | 5/5 |
| 2048 / 128 | 2945.62 ms | 313.36 ms | 9.36× | 5/5 |

| 输入 / 输出 token | 官方端到端 | Graph 端到端 | 端到端加速 | Graph 更快次数 |
|---|---:|---:|---:|---:|
| 1024 / 32 | 865.59 ms | 232.34 ms | 3.73× | 5/5 |
| 2048 / 32 | 952.28 ms | 320.41 ms | 2.97× | 5/5 |

关闭 CUDA Graph 后，分页 eager 在四组测试中均约快 **1.31×**；
在分页路径上开启图执行，进一步降低约 7–8 倍 decode 延迟。
因此大部分观测收益来自复用 nano-vLLM 的 CUDA Graph 执行路径，
不能把约 10 倍提升全部归因于分页，也不能宣称这些上游优化由本项目发明。
本实验没有进一步拆分 eager 模型、KV 追加和各个算子的单独贡献。

HF FlashAttention-2 解码相对 SDPA 的配对加速比均约 **0.96×**，在本配置下
没有改善总体 decode 延迟。更换 HF attention 内核不能消除本次观测到的差距；
这不意味着所有其他引擎或优化过的 HF 实现都会有同样差距。

端到端配对中，官方 prefill 总耗时的均值在两条路径上基本一致：
1K 为 158.08/157.97 ms，2K 为 236.45/236.93 ms（官方/Graph）。
Graph 路径额外导入平均耗时 3.90/7.64 ms，解释了为什么首 token 没有加速，
以及 decode 的约 10 倍收益会被 prefill 和融合成本摊薄。

## 适用边界

- 这是固定长度合成输入的预热执行测试，不是任务准确率评测。
  本次 Graph 在 1K 的 31/127 步中分别与官方 top-1 一致 30/126 步；
  2K 分别为 31/31 和 127/127。**不能宣称输出严格等价或任务质量无损。**
- 已有独立基准中，官方/Graph 的单请求 TTFT 为 161.9/165.3 ms（1K）
  和 238.4/247.5 ms（2K）；这属于独立基线结果，不是本次审计新增的 TTFT 指标。
- 本次审计没有做显存收益对比。已有独立基准的最大 PyTorch allocated 峰值
  从官方 12.47 GiB 增至 Graph 15.60 GiB，包含额外接收模型和 2044 MiB KV 池。
  分页带来可控容量和分配回收能力，当前实现没有证明峰值显存更低。
- 已有 batch=8 的 2413.5/1816.6 tok/s 是聚合 decode 吞吐，不能用于声明
  batch=1 的加速；本次效率结论全部来自 batch=1 对照。
- 5 次重复的观测范围不是置信区间。本次全部缓存页回收成功，进程退出后
  GPU 显存占用和利用率均回到 0；仍未测试长时间服务负载、P99 或能耗。

## 复现与证据

在项目的专用 Linux CUDA 环境中运行：

```bash
python -m c2c_decode.efficiency --repeats 5 \
  --output results/efficiency-audit.json
```

- [验证代码](../src/c2c_decode/efficiency.py)
- [原始逐次记录、token 历史与版本](../results/efficiency-audit.json)
- [自动生成的完整对照表](../results/efficiency-audit.zh-CN.md)
- [正确性、吞吐和显存评估](results.zh-CN.md)

可用于项目描述的表述：在 RTX 4090 上，将官方 C2C 融合 KV 接入 nano-vLLM
分页解码和 CUDA Graph；在相同融合状态及 token 历史的单请求测试中，
1K/2K 输入的 decode 获得约 9–10 倍加速，32 token 输出端到端获得约
3.0–3.7 倍加速，同时保留 BF16 数值差异和额外显存开销的验证记录。

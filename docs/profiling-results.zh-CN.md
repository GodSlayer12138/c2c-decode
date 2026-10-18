# C2C-Decode profiling 与优化

[English](profiling-results.md)

2026-10-17，RTX 4090 24GB，BF16，固定 Qwen3-4B → Qwen3-0.6B 与官方已训练 Fuser。
本实验完成 CPU/CUDA 时间线、算子分析、KV 导入优化、显存生命周期和混合到达调度实验。
完整实测表由脚本从原始 JSON 生成，见 [测量汇总](../results/profiling/summary.zh-CN.md)。

![性能与显存概览](../results/profiling/overview.png)

[可缩放矢量图](../results/profiling/overview.svg)

## 1. CPU/GPU 时间线与 decode 开销

在实际 nano-vLLM 引擎上采集 PyTorch CPU/CUDA trace，覆盖 1K/B1、2K/B8；
另有 1K/B1 eager 对照。官方 prefill、Fuser 也有独立 trace。

1K/B1、每请求输出 16 tokens 的 profiler 关闭阶段计时中，nano Graph 平均每步
延迟的五次中位数为 **2.268ms**，nano eager 为 **17.056ms**。
这些值由每次的 `(导入+decode-导入)/15` 计算；计时 wrapper 仍存在，正式吞吐
使用不启用这些 wrapper 的 `nano_benchmark`。首 token 来自官方 C2C。
输入准备约 **0.090ms/步**，这是主机调用耗时；嵌套区间不能相加。
同一 1K/B1 trace 中，GEMV kernel 累积约 **22.02ms**，attention 也是主要 GPU 开销。
保留了 kernel 调用数、时长和 eager 算子输入 shape。
CUDA Graph 主要减少逐算子主机发射成本，不能把它理解为消除了图内 kernel。
显存带宽、SM occupancy、Tensor Core 利用率尚未测量，不能仅凭 GEMV 名称判定为带宽瓶颈。

![真实 trace 的前 10ms](../results/profiling/timeline.png)

[可缩放矢量图](../results/profiling/timeline.svg)

图中 GPU activity 是记录到的 kernel/拷贝区间。时间覆盖率不是 `nvidia-smi` 利用率，
更不是 occupancy；缺口也可能包含 profiler 扰动。eager 开启 shape 记录，开销尤其大。
性能加速比一律取 profiler 关闭的实验，不能用 trace 的 wall time 做性能对比。
这与 [PyTorch Profiler 对额外开销及张量引用的说明](https://docs.pytorch.org/docs/2.8/profiler.html)一致。
当前环境未找到 Nsight Systems；本实验没有采集 Nsight Compute 硬件计数器。

## 2. KV 导入：从定位到实测优化

旧 nano 适配器按层、按物理页发起复制；新实现一次构造 token 到物理页/页内位置的映射，
按层对 K、V 批量 scatter。默认切换为 `import_mode="scatter"`，旧 `loop` 保留用于复测。
没有再次应用 RoPE，没有改变序列位置、首 token 交接或分页分配规则。

1K/2K × B1/2/4/8，每档 5 组随机顺序配对，共 40 组；每请求输出 32 tokens，
所有配对生成 token 逐项一致。导入加速约 **3.5–7.2 倍**，40/40 组均改善。
2K/B8 的导入中位数 **59.98 → 8.36ms**；导入加解码 **195.63 → 143.97ms**，
总耗时降低 **26.4%**。这不是包括官方 prefill 的端到端加速。

单独的 2K/B8 导入 trace 中，kernel 数 **3592 → 488**，其中旧路径 3584 次
逐页复制变成新路径 448 次按层写入，另有索引和首 token 提取开销。这支持“小 kernel
及主机发射过多”的定位。报告中的有效 payload GB/s 仅计算 KV 数据大小一次，不能当作硬件 DRAM 带宽。

正确性增加了碎片化物理页、部分尾页、未写入槽位、两种导入模式的检查；22 项测试通过。
另外重新执行与官方融合 continuation 对齐的数值验证、完全相同历史的 teacher forcing、
自由 greedy 和页回收检查。导入优化前后 token 一致，不代表两个推理引擎与官方路径严格 token 等价；
BF16 引擎间差异仍需按原数值门槛和 top-1 统计报告。

## 3. 显存生命周期

采集对象是**单进程官方 C2C + nano CUDA Graph + 2044MiB KV 池**。
记录 receiver/sender forward、逐层 Fuser、融合后清理、导入、丢弃 HF handoff、decode 回收，
并保存有 Python 分配栈的 allocator snapshot。Rosetta 部分调用直接使用 `.forward`，因此使用
可恢复的方法 wrapper；普通 `nn.Module` forward hook 会漏掉这些调用。

测试顺序为 1K、2K、重复 2K，检查 live allocated 是否回到常驻值，reserved 是否只保留
allocator 缓存。导入写入预分配 KV 池，导入期间 HF 融合 KV 仍存活；删除 handoff 后才释放其占用。
每次结束同时验证官方 `kv_cache_dict` 已清空，分页请求释放。

实测 live allocated 常驻值 **12.732GiB**，2K 路径累计峰值 **15.603GiB**。
三次请求完成并丢弃 handoff 后均回到同一个常驻值；重复 2K 的 reserved 稳定在
**15.775GiB**，显式 `empty_cache` 后降至 **12.807GiB**。这里看到的是 allocator
缓存保留，没有观察到这三次请求导致的 live tensor 累积。
两模型在 2K prefix forward 结束时各产生 `[1,2047,151936]` 的 BF16 logits，
每份约 **593.2MiB**。官方路径还保留干净 KV 克隆和融合中间张量，解释了为什么不能
仅按模型权重加分页池估算峰值。本实验保留官方完整 logits 路径，未为了省显存改动正确性参考。

详细 GiB、logits 的实际 shape/dtype/字节和峰值以 [memory.json](../results/profiling/memory.json)
与汇总表为准。`allocated`、`reserved`、CUDA driver 总占用是不同指标，不能互相替代；
有限的重复实验也不是任意长时运行无泄漏的证明。snapshot 方法遵循
[PyTorch CUDA memory 文档](https://docs.pytorch.org/docs/2.8/torch_cuda_memory.html)。
显存采样含 GC、分配栈追踪开销，其横轴不能用来报告 prefill 加速比。

## 4. 到达负载、排队与 prefill 干扰

新增请求 `arrival_ms` 和逐 token 时间戳。回放 24 请求/次，输入长度循环
128/512/1024/2048，输出长度覆盖 16/32/64/128，关闭 EOS 提前结束。
到达间隔 0/100/350ms，每档两种策略各跑 3 次，共 **432 请求**。
这是单机引擎内的预定到达回放，没有 HTTP、网络、真实客户端流式传输或生产 SLA。

对照原策略“有空位就连续 prefill”与“每轮至多一个完整 prefill，然后 decode”。
后者是一个调度参数，不是 chunked prefill，也没有 prefill/decode 并行。

突发到达下，相邻 token 间隔 P99 **426.6 → 265.3ms**，下降 **37.8%**；
吞吐 **266.0 → 265.4 tok/s**。100ms 间隔下也有相近 P99 改善。
但突发负载平均请求 TPOT **23.94 → 24.74ms**，间隔 P95 **164.5 → 188.0ms**，
TTFT 也没有改善，因此**不把新策略设为默认**。它减少极长停顿，但并非所有延迟指标都变好。

350ms 间隔下，原策略排队 P99 约 **2.2ms**，TTFT P99 约 **267.8ms**；突发负载
TTFT P99 约 **5.06s**。说明串行 prefill 在高负载下形成明显排队。
低负载 inter-token P99 约 3.1ms，仍不能排除少于 1% 的长 prefill 停顿。

TTFT、queue、E2E、每请求 TPOT、相邻 token 间隔均提供 P50/P95/P99。
TPOT 是每请求 `(末 token 时间-首 token 时间)/(输出数-1)`；interval 分位数则汇总所有
相邻 token 间隔，两者不同。每档只有 72 请求，P99 是小样本经验值，不能宣称稳定服务尾延迟。
吞吐包括到达间隔产生的空闲时间。所有 18 次回放结束后页池都回收到初始空闲数量。

## 5. 优化后的 nano 独立生产端审计

保留同卡 CUDA IPC 的独立官方 producer 路径作为 nano 复核入口，使用 2044MiB 池、
1K/2K、B1/2/4/8、32-token 输出，以及 B1 128-token 补充。预热后每档五次，
覆盖自由 greedy 和同历史 teacher forcing。该路径的端到端计时包含 IPC，不能与
进程内测试直接相减。

2K/B8、32-token 自由 greedy 的导入加解码为 **143.07ms**；同历史对照为
**145.88ms**。B1 新请求端到端为 **244.77ms（1K）/331.19ms（2K）**。
共 100 条 decode、10 条端到端记录，原数值验证集及 KV/回收检查通过。
扩展任务仍有数值阈值失败，详见性能与正确性评估，不能宣称普遍严格等价。

模型加载、编译、分词和 HTTP 不在热启动计时内。原始 nano 数据保持不变，
模块已整理为 `nano_benchmark.py`；旧结果中的参数和 trace 栈保留测量时记录。

## 复现与原始产物

在服务器项目目录使用已有专用环境，避免本地 base；GPU 实验按顺序运行，不并发竞争同一张卡：
图表脚本使用 Matplotlib 3.10.5，可通过项目的 `profile` extra 安装绘图依赖。

```bash
cd /root/autodl-tmp/c2c-decode
export OMP_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false
export TORCHINDUCTOR_CACHE_DIR=$PWD/.cache/inductor
export TRITON_CACHE_DIR=$PWD/.cache/triton
export MPLCONFIGDIR=$PWD/.cache/matplotlib
PY=/root/autodl-tmp/envs/c2c-decode/bin/python
$PY -m pytest -q
$PY scripts/profile_runtime.py --output-dir results/profiling/nano-graph
$PY scripts/profile_runtime.py --eager --output-dir results/profiling/nano-eager
$PY scripts/profile_experiments.py
$PY -m c2c_decode.nano_benchmark --import-mode scatter --output results/profiling/nano-scatter.json
$PY scripts/summarize_profiling.py
```

仅复跑某一组可使用 `profile_experiments.py --section import|memory|scheduling`。
本次修正直接 forward 采样后单独重跑了 memory 组，其他组无需重复。

| 产物 | 内容 |
|---|---|
| `results/profiling/summary.md` | 从数据自动生成的完整表格 |
| `nano-graph/`、`nano-eager/` | 原始试验、host stages、CUDA trace、kernel 和 operator 汇总 |
| `import-ablation.json` | 80 次导入与解码记录，40 组配对、顺序和输出 token |
| `import-loop.json.gz`、`import-scatter.json.gz` | 仅导入的 CPU/CUDA trace |
| `official-prefill.json.gz` | 官方两模型与 Fuser 的 CPU/CUDA trace |
| `memory.json`、`memory-snapshot.pickle.gz` | 显存时间点和 allocator 分配栈快照 |
| `scheduling.json` | 18 次到达回放、432 请求逐 token 时间戳和分位数 |
| `nano-scatter.json` | nano 独立生产端正确性和性能矩阵 |
| `measurement-provenance.json`、`summary-source-sha256.json`、`tests.txt` | 测量期保留模块哈希、汇总源码哈希与测试记录 |

trace `.json.gz` 解压后可用 Perfetto 打开；memory snapshot 解压后可用 PyTorch memory
可视化工具分析。原始文件均保存于项目目录，本次没有上传到外部可视化服务。
Profiling 采用独立脚本；核心仍复用官方模型和推理引擎，没有引入自研 attention kernel。
测量期与当前整理后的源码哈希分开记录；核心代码仍保持在原定规模内。

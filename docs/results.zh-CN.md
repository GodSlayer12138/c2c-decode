# 首轮实测结果

[English](results.md)

本页保存 2026-10-17 的首轮历史基线，使用固定上游 commit 和模型 revision、BF16，不训练或量化；属于已有 C2C 的系统适配。后续功能和优化见 [第二轮记录](next-improvements.zh-CN.md)。

## 正确性与限制

- 官方 C2C 可用指定 Qwen3-4B → Qwen3-0.6B Fuser 生成。
- 长度 31/255/256/257/1024/2048 的导入 KV 逐元素一致。
- 四个普通 prompt 的 16-token greedy 输出与官方精确一致。
- EOS、变长输出、复用和完整页回收检查通过。
- 基础测试覆盖 CUDA 分页、调度和严格 token CLI 退出门槛；下载权重哈希校验通过，见 `results/tests.txt`。
- 长输入的同历史 logits 通过预设 BF16 阈值，但严格 token 等价未通过：eager top-1 为 88/90，Graph 为 87/90，不能声称生成轨迹普遍相同。

Eager 最大绝对误差 0.375、最大 RMSE 0.08594、最小 cosine 0.999341。两处不同 argmax 的官方 margin 为 0.125 和 0.0625，与 BF16 运算路径差异相符，但不能证明未来差异都由此引起。`validate --strict-tokens` 暴露更严格且失败的门槛；默认 `passed` 是容差和测试夹具门槛，`strict_token_equivalence` 单独记录。原始记录为 `results/validation-eager.json` 与 `validation-graph.json`。

## 性能

每档预热，固定生成 32 tokens，取三次中位数。Prefill batch=1；decode 同批输入长度不同，名义长度为最长请求，其余依次短 17 tokens。

| 输入 | 官方 decode B1 | Eager decode B1 | Graph decode B1 | Graph decode B8 | Graph E2E B8 |
|---:|---:|---:|---:|---:|---:|
| 1024 | 42.4 tok/s | 54.3 tok/s | 434.5 tok/s | 2413.5 tok/s | 182.8 tok/s |
| 2048 | 42.6 tok/s | 54.6 tok/s | 405.2 tok/s | 1816.6 tok/s | 123.0 tok/s |

单请求加速应比较 B1，B8 是聚合吞吐。1K/2K 的 Graph B1 E2E 为 135.1/98.7 tok/s，官方为 35.8/33.2 tok/s。这是合成性能测试，不是准确率评测。

Graph B1 TTFT 为 165.3/247.5ms；B8 因包含串行 prefill 排队，中位请求 TTFT 增至 738.4/1106.3ms。

| 单请求阶段 | 1K 中位数 | 2K 中位数 |
|---|---:|---:|
| 接收端＋发送端 prefix forward | 93.206ms | 168.065ms |
| 融合区间，含官方缓存管理 | 43.529ms | 45.904ms |
| 末 prompt token forward | 23.403ms | 23.952ms |
| 缓存校验、分配和导入 | 4.273ms | 8.400ms |

各阶段独立中位数不精确相加为 E2E，定义见 [设计](design.zh-CN.md)。分页模型和图执行的峰值 allocated 为 15.60GiB，官方独立基线为 12.47GiB。2048MiB 预算下池分配 73×28=2044MiB；allocated 不包含 PyTorch 以外的驱动显存。

Graph replay 减少 eager 剩余的主机发射开销，批处理提高聚合吞吐。在首轮所测完整链路中，串行 prefill 与融合占主要时间，导入相对较小；未来可探索 prefill/decode 重叠和键设计正确的融合前缀复用。

首轮约 756 行核心 Python、含测试脚本约 1100 行（不含上游）；当时尚无 HTTP、取消、抢占、多 GPU 或训练流水线。这些是该轮历史状态，不代表当前服务功能。

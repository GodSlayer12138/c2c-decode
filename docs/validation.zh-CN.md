# 当前 nano 项目验证记录

[English](validation.md)

2026-10-18，沿用已有 RTX 4090 与专用 `c2c-decode` 环境。

- `compileall` 检查通过；`pytest -q`：**37 passed in 4.90s**。
- `profile_runtime.py --help`、`quality_eval.py --help` 可正常加载当前 nano 入口。
- `python -m c2c_decode.nano_benchmark --smoke --output results/cleanup/nano-smoke.json`
  实际执行官方生产端、融合 KV 导入与 nano CUDA Graph 解码。
- 257-token 输入验证包含 15 步同历史 logits 检查：精确 KV、强制历史、数值门槛
  与本次 top-1 检查通过。另完成 1K/B1、32-token 自由/强制历史运行及新请求 E2E。
- 所有烟测请求的缓存页全部回收；进程结束后设备显存为 0 MiB、利用率 0%。
- 两份汇总脚本从保留的 nano 原始数据重新生成表格与图片。

这是整理后的功能回归，**不作为新的速度或质量基准**。原有扩展任务中的数值失败
仍保留在 `results/next/numerical-failures.json`，本次小样本通过不能覆盖这些失败。

证据：[测试输出](../results/cleanup/tests.txt)、[GPU 冒烟原始记录](../results/cleanup/nano-smoke.json)。

`measurement-provenance.json` 记录历史测量期的保留模块哈希，不代表当前源码；
`summary-source-sha256.json` 记录重新生成报告时的源码。nano 原始计时、输出与 trace
未被改写；其中路径/参数字段保留采集时的值。当前基准入口为 `nano_benchmark.py`。

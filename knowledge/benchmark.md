# 性能测试知识

一次结果必须固定：模型、代码 revision、Runbook、环境、TP、dtype 和 workload。

至少记录：

- requests、输入/输出长度；
- request rate 或 concurrency；
- warmup 和 repeats；
- request/input/output throughput；
- TTFT、TPOT、ITL、E2E latency；
- 成功/失败请求数；
- 可获取时的峰值设备内存。

`status` 表示测量是否有效，`target_met` 表示性能目标是否达到，两者必须分离。

只有输入指纹和 workload 完全一致时才进行历史回归比较。

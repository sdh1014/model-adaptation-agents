# 性能指标

| 指标 | 含义 | 常见方向 |
|---|---|---|
| request throughput | 每秒完成请求数 | higher |
| input throughput | 每秒处理输入 token | higher |
| output throughput | 每秒生成输出 token | higher |
| TTFT | 首 token 延迟 | lower |
| TPOT | 每个输出 token 的平均时间 | lower |
| ITL | 相邻 token 延迟 | lower |
| E2E latency | 请求端到端延迟 | lower |
| goodput | 满足 SLO 的有效吞吐 | higher |
| peak memory | 峰值设备内存 | 取决于目标 |

必须保留 unit。`ms`、`s`、`token/s` 和 `request/s` 不得混用。

吞吐与延迟受输入/输出长度、并发和请求率直接影响，因此脱离 workload 的数值不可比较。

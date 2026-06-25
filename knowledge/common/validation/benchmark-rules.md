# 性能测试规则

正式测试前必须通过正确性验证。

固定并记录：

- 模型和引擎 revision；
- dtype、TP、最大长度；
- 输入和输出 token 长度；
- 并发；
- warmup 与正式重复次数；
- TTFT、TPOT、吞吐和峰值显存。

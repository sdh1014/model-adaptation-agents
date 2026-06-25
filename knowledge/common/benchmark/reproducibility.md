# Benchmark 可复现性

正式比较至少需要一致：

- 模型与 checkpoint revision；
- 目标代码 commit 和 patch；
- 引擎、插件、Python 和运行时版本；
- 硬件型号、设备数量和拓扑；
- dtype、量化、TP/DP/EP；
- 服务启动参数；
- 数据集、随机种子、长度、请求数和并发；
- benchmark 工具版本；
- warmup/repeat 策略。

Runbook 哈希不同不一定不可比较，但必须审查变化是否影响运行语义。

结果波动明显时，先检查负载、温度/频率、设备占用、缓存、JIT/图编译和网络，而不是立即修改模型代码。

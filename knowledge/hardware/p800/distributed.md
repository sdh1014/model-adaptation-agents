# P800 分布式评估

## 适用条件

仅当任务配置包含 TP > 1、EP、PP 或多机时读取。

## 环境事实

- 设备数量和 device mapping；
- 单机或多机；
- 进程启动方式；
- 通信 backend 和运行时版本；
- master address/port；
- 网络接口，仅多机需要；
- 目标仓的 distributed override 或 patch。

## 模型约束

- attention/KV heads 与 TP；
- hidden/intermediate/vocab 切分；
- expert 数量与 EP；
- shared expert；
- 量化 scale 和 packed weight 分片；
- 多模态 encoder 和 MTP 模块切分。

## 支持证据

- 当前 commit 的 distributed 代码；
- 对应模型或相似结构的测试；
- 单机多卡最小初始化；
- collective 单测；
- baseline 日志明确完成所有 rank 初始化。

## 常见边界

- 单卡通过不能证明多卡；
- TP 初始化通过不能证明 weight shard 正确；
- all-reduce 可用不能证明 all-to-all/EP 可用；
- 多机环境错误不能直接归因到模型代码。

## 本阶段输出

只判断目标并行配置的可行性和缺口；通信性能、拓扑优化和生产稳定性留到后续阶段。

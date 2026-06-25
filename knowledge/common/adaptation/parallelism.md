# 并行能力评估

## 任务配置

记录目标：

- tensor parallel size；
- pipeline parallel size；
- expert parallel size；
- data parallel size；
- 单机或多机；
- 设备数量和映射。

## 模型约束

- attention heads 与 KV heads 是否可按 TP 切分；
- hidden/intermediate size 是否整除；
- vocab 和 LM head 切分；
- expert 数量与 EP；
- shared expert 的复制或切分；
- multimodal encoder 是否参与并行；
- MTP/speculative 模块如何分片。

## 引擎与 P800 检查

- 通信 backend；
- collective 的 dtype 和 shape 支持；
- 参数 loader 的 rank 逻辑；
- all-reduce、all-gather、reduce-scatter、all-to-all；
- 进程启动和设备可见性；
- 单机多卡与多机路径是否不同；
- graph/compile 对 rank 和 shape 的限制。

## 支持判定

TP=1 baseline 通过只能证明单卡路径。目标是 TP=8 时，未验证 TP=8 的能力不得标记为 fully supported。

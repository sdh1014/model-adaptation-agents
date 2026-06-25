---
scope: hardware/p800
status: verified
---

# P800 适配评估检查表

## 目标

本检查表只用于确认“当前环境是否足以做适配判断和最小 baseline”，不是性能验收。

## 必查事实

### 主机和 Python

- Linux 发行版、内核、架构；
- Python 版本和可执行路径；
- 当前虚拟环境；
- `LD_LIBRARY_PATH`、`PYTHONPATH` 和 Kunlun/XPU 相关可见设备变量。

### 运行包

按实际引擎检查：

- `torch`；
- `vllm` 与 `vllm-kunlun`，或 `sglang` 与 Kunlun 扩展；
- 目标仓 requirements 中声明的 Kunlun 算子、编译、通信和插件包；
- 包版本与 import 实际位置。

版本规则必须从当前 target commit 的 README、requirements、pyproject 或安装文档中读取，不依赖长期固定版本号。

### 设备

同时检查：

- `torch.cuda` 兼容接口；
- `torch.xpu` 接口；
- 设备数量；
- 可能的 `/dev` 设备节点；
- 已安装的设备管理工具。

不同运行时可能通过不同 PyTorch 接口暴露 P800，不能只检查单一 API。

### 路径和资源

- 模型路径可读；
- 目标仓和上游仓存在；
- 模型文件系统剩余空间；
- `/dev/shm` 容量；
- baseline 所需设备数不超过可见设备数。

## Readiness 判定

### `ready`

- 关键路径有效；
- 必需包存在且 import 成功；
- 设备可见数量满足 baseline；
- 没有明确版本或 ABI 冲突。

### `degraded`

- 可尝试 baseline，但有非阻塞警告，例如 dirty worktree、空间偏低、设备工具缺失。

### `unavailable`

- 引擎或插件 import 失败；
- 设备不可见；
- 必需路径不存在；
- 版本或动态库冲突已被实际 import/loader 错误确认。

### `unknown`

- 检查接口与当前运行时不匹配，无法确认设备或插件状态。

## 不在本阶段检查

- 性能时钟、功耗和温度；
- 多机网络带宽；
- 正式通信 benchmark；
- 长时间运行稳定性；
- 生产服务端口和监控。

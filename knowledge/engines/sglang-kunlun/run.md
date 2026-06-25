# SGLang-Kunlun 运行知识

## Runbook 关注点

`env.sh` 通常记录：

- Kunlun 平台插件和运行时环境变量；
- SGLang / SGLang-Kunlun 仓库或包路径；
- 模型路径、dtype、TP、最大长度；
- 服务 host、port 和 endpoint。

`start.sh` 应粘贴当前仓库实际使用的 SGLang-Kunlun 启动命令，不由 Skill猜测平台插件参数。

## 运行前检查

- Python 实际 import 的 SGLang 和 Kunlun 插件路径；
- 平台插件是否被发现；
- P800 设备数量和 TP 设置；
- 服务端口；
- 启动进程保持前台。

## 验证复用

`adapt-validate` 必须复用同一 `start.sh`，只将验证内容放入 `checks/validate.sh`，避免 Smoke 和验证使用不同启动参数。

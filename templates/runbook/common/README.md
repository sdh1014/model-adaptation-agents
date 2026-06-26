# Target Runbook

普通使用者主要编辑四个文件：

```text
env.sh                    公共环境变量
start.sh                  完整前台启动命令
checks/validate.sh        正确性验证
checks/benchmark.sh       性能压测
```

完整结构：

```text
env.sh                    可提交的公共环境变量
env.local.sh              可选，本地秘密；不会提交
start.sh                  完整前台启动命令
ready.sh                  单次就绪探测
stop.sh                   可选优雅停止
checks/smoke.sh           最小调用
checks/validate.sh        正确性验证
checks/benchmark.sh       性能压测
```

可以直接粘贴多行 Shell 命令。`start.sh` 必须保持前台，推荐最后使用 `exec`。

创建本地秘密文件：

```bash
cp env.local.sh.example env.local.sh
```

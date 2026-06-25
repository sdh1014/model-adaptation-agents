# adapt-implement 辅助脚本

这些脚本不负责决定修复方案，只负责创建目录、采集事实和执行可重复检查。

| 脚本 | 作用 |
|---|---|
| `create_run.py` | 创建不可覆盖的实现 run |
| `create_attempt.py` | 创建下一编号 attempt，并限制尝试次数 |
| `collect_context.py` | 记录 Python、模块来源和少量安全环境字段 |
| `snapshot_repo.py` | 保存 Git HEAD、dirty 状态、累计 patch 和哈希 |
| `history.py` | 汇总同一工作项的历史命令、日志尾部和失败签名 |
| `failure_signature.py` | 从命令证据提取稳定失败签名，不声明根因 |
| `check_scope.py` | 检查本 attempt 修改是否超过允许路径 |
| `check_implementation.sh` | 在目标仓执行局部验收，并在命令前后检查修改范围 |

所有命令输出写入显式 `--run-dir`，不更新 `tasks/` 或 `knowledge/`。

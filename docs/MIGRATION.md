# 从完整版本迁移到精简版本

## 保留

```text
tasks/
runs/
目标 Runbook
已有阶段报告
```

## 主要变化

- Skill 从多个关联文件压缩为 `SKILL.md + GUIDE.md`；
- 详细知识合并为 8 个主题文件；
- 三套重复 Runbook 模板改为“一套公共模板 + 引擎 start.sh”；
- 实现辅助脚本合并为 `scripts/implement.py`；
- 模型检查脚本合并为 `scripts/model.py`；
- 评估脚本合并为 `scripts/assess.py`；
- 删除发行清单、TREE、SHA256 和测试报告等仓库噪声。

## 迁移建议

先备份旧仓库，将新仓库文件覆盖到一个新分支，再复制现有 `tasks/`。目标 Runbook 不应被新模板覆盖。

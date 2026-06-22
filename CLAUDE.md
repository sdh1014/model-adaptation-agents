# Claude Code Rules

本仓库是模型适配 Agent 的流程仓，不是目标框架代码仓。

必须遵守：

1. 优先使用 `.claude/skills/model-adaptation/SKILL.md`。
2. 目标框架代码只在目标仓中修改，例如 `vllm-kunlun` 或 `aiak-sglang`。
3. `knowledge/` 只保存人工确认过的规范、检查项和常见问题。
4. `runs/` 只保存实验事实，不写失败原因分析。
5. 每一轮只验证一个技术假设。
6. Bash 脚本通过 `scripts/run_bash.py` 包装调用。

失败 attempt 禁止写入这些字段：

- `root_cause`
- `reason_analysis`
- `hypothesis`
- `next_action`

允许保存这些事实：

- 执行命令
- 退出码
- 运行环境
- 修改 diff
- 原始日志路径
- 失败检查项
- 错误签名
- 指标数据

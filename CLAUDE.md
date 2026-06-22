# Claude Code Rules

本仓库是模型适配 Agent 的流程仓，不是目标框架代码仓。

## Directory Roles

```text
.claude/skills/model-adaptation/
  SKILL.md 是 Agent 入口；flows/ 放阶段流程和行为约束。

tasks/_template/
  新任务模板，只保留框架默认结构。

tasks/<task>/
  当前具体任务目录，默认被 Git 忽略。
  task.yaml 记录模型、目标仓、框架、硬件、当前阶段。
  status.md 记录当前状态和最近进展。
  notes.md 记录人工确认的范围、重点、禁止事项和决策。
  references.md 记录技术报告、PR、链接、本地文件和阅读优先级。
  reports/ 记录各阶段人工可读报告，例如 reports/model-research.md。

runs/<task>/
  每次执行的事实记录，默认被 Git 忽略。
  旧 attempt 默认保留为历史记录；tasks/<task>/status.md 是唯一当前指针。

knowledge/
  跨任务复用的人工确认知识，不放单个任务上下文。

scripts/
  可复用执行脚本和 Bash 包装器。

adaptctl/
  最小 CLI，负责 init、run、status。
```

必须遵守：

1. 优先使用 `.claude/skills/model-adaptation/SKILL.md`。
2. 目标框架代码只在目标仓中修改，例如 `vllm-kunlun` 或 `aiak-sglang`。
3. `knowledge/` 只保存人工确认过的规范、检查项和常见问题。
4. `runs/` 只保存实验事实，不写失败原因分析。
5. 判断当前事实只能读取 `tasks/<task>/status.md` 指向的 run，不能按目录时间或 result.json 猜测。
6. stage 人工可读报告统一写入 `tasks/<task>/reports/`，任务根目录只放任务元数据和人工维护文件。
7. 每一轮只验证一个技术假设。
8. Bash 脚本通过 `scripts/run_bash.py` 包装调用。

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

`result.json` 的具体形状（FAIL 与 PASS）由对应 flow 定义，例如模型调研见 `flows/model-research.md`。

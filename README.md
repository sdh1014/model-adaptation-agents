# Model Adaptation Agents

这是一个独立的模型适配 Agent 仓库。

目标代码修改仍发生在目标框架仓库中，例如 `vllm-kunlun` 或 `aiak-sglang`。本仓只管理流程、知识、任务状态和实验事实。

第一版只让使用者理解四个目录：

```text
tasks/      当前要适配什么模型
knowledge/  人工确认的知识和常见问题
scripts/    实际执行脚本
runs/       每次尝试的事实记录
```

## 目录

```text
.
├── CLAUDE.md
├── .claude/skills/model-adaptation/
├── adaptctl/
├── tasks/
├── knowledge/
├── scripts/
└── runs/
```

## 快速开始

创建任务：

```bash
python -m adaptctl init minimax-m3 --model MiniMax-M3 --target-repo ../vllm-kunlun --framework sglang
```

启动模型调研：

```bash
python -m adaptctl run minimax-m3 model --model-path /path/to/model
```

查看状态：

```bash
python -m adaptctl status minimax-m3
```

环境勘测：

```bash
python -m adaptctl run minimax-m3 env
```

## 规则

- `tasks/<task>/task.yaml` 写任务基本信息。
- `tasks/<task>/status.md` 写当前状态。
- `tasks/<task>/notes.md` 写人工确认的信息。
- `tasks/<task>/references.md` 写技术报告、PR、链接和本地文件。
- `runs/<task>/...` 只保存事实，不写失败原因分析。
- `knowledge/` 只保存人工确认过的知识。

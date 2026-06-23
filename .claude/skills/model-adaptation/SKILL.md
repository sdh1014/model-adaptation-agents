---
name: model-adaptation
description: Guide model adaptation work with simple tasks, knowledge files, scripts, and run records for P800 or GPU framework bringup.
---

# Model Adaptation Skill

Use this skill for model adaptation work on P800, H20, vLLM-Kunlun, AIAK-SGLang, environment survey, model research, bringup, accuracy, and performance tests.

## Workflow

1. Read `tasks/<task>/task.yaml`.
2. Read `tasks/<task>/status.md`.
3. Read `tasks/<task>/notes.md`.
4. If `tasks/<task>/references.md` exists, read it.
5. For model research, follow `flows/model-research.md`.
6. Read the relevant file in `knowledge/`.
7. Run one command through `adaptctl`.
8. Put factual outputs under `runs/<task>/`.
9. Update `status.md`.

## Commands

Create a task:

```bash
python -m adaptctl init <task> --model <model> --target-repo <repo> --framework <framework>
```

Run environment survey:

```bash
python -m adaptctl run <task> env
```

Run model research:

```bash
python -m adaptctl run <task> model --model-path <path>
```

If no local model path is available, record the Hugging Face link in
`tasks/<task>/references.md` and follow `flows/model-research.md` without
downloading full weights.

Show status:

```bash
python -m adaptctl status <task>
```

## Rules

- Do not rewrite existing Bash benchmark scripts only because they are Bash.
- Do not mix model registration, attention, MoE, quantization, and performance tuning in one attempt.
- Do not put unverified failure analysis into `runs/`.
- Do not modify target framework code from this repo unless the task explicitly points to an attached target repository.
- Do not add new top-level directories for first-version work.
- When the user provides links, PRs, files, or reports, update `tasks/<task>/references.md` first.
- Update `tasks/<task>/notes.md` only under `Task Notes Rules`.

## Task Notes Rules

`tasks/<task>/notes.md` is for user-confirmed task guidance, not stage reports or factual records.

- `Scope`, `Focus`, `Do Not`, and `Human Decisions` can change during the task.
- Only write user-confirmed scope, focus, forbidden work, baseline choice, environment choice, or stage decision.
- Do not write model structure summaries, environment facts, blockers, missing items, command output, log summaries, agent inference, or unverified cause analysis.
- Stage facts go to `runs/<task>/<ts>-<stage>/result.json`.
- Human-readable stage reports go to the report file defined by that flow, such as `tasks/<task>/model-research.md`.

## Required Outputs

Every run should contain:

```text
input.json
result.json
logs/
raw/
```

Optional files:

```text
patch.diff
metrics.json
environment.json
```

## Stage Template

写新 flow（阶段）时，该 flow 必须定义：

- stage id
- 目标
- 输入
- 允许的输出
- 禁止写入的内容
- 验证方式
- 进入下一阶段的条件

阶段推进只看事实与验证结果，不看 Agent 的主观判断。`result.json`（FAIL 与 PASS）的具体形状由该 flow 自己定义。

定义 `result.json` 时必须说明：

- schema 是固定字段还是按任务事实灵活组织。
- 固定字段不得使用空对象占位，必须定义字段内部的最低内容。
- 关键事实必须有 evidence。
- 缺测项写入 `missing`，不得编造。
- 阶段门控字段必须写明判定规则。

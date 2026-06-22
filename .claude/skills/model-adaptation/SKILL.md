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
4. Read the relevant file in `knowledge/`.
5. Run one command through `adaptctl`.
6. Put factual outputs under `runs/<task>/`.
7. Update `status.md`.

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

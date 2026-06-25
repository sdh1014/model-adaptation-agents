---
name: adapt-assess
description: 在不修改目标代码的前提下，评估一个已分析模型在指定推理引擎与 P800 环境上的可适配性；区分模型支持、硬件后端、运行环境和配置问题，执行可行的最小 baseline，并生成证据化的适配结论与实施工作项。
argument-hint: "<model-id>/<target-id> --target-repo <path> [--upstream-repo <path>] [--engine <vllm-kunlun|sglang-kunlun>] [--hardware p800] [--baseline <auto|required|skip>] [--focus <capability>]"
disable-model-invocation: true
---

# 适配评估

对 `$ARGUMENTS` 指定的模型目标执行适配评估。

## 执行前必须读取

1. [requirements.md](requirements.md)：判定规则、证据要求和阶段边界。
2. [workflow.md](workflow.md)：完整执行步骤和失败处理。
3. 写结果前读取 [assessment-template.md](assessment-template.md)。
4. 需要生成实施计划时读取 [implementation-template.md](implementation-template.md)。

仅在需要确认格式时读取 `examples/`。

## 必要输入

- `tasks/<model-id>/model.yaml`
- `tasks/<model-id>/model-analysis.md`
- `knowledge/common/adaptation/capability-matrix.md`
- `knowledge/common/adaptation/gap-taxonomy.md`
- `knowledge/common/adaptation/work-item-rules.md`
- `knowledge/hardware/<hardware>/assessment-checklist.md`
- `knowledge/engines/<engine>/assess.md`
- 目标仓库；存在 `upstream_repo` 时同时读取上游引擎仓库

目标引擎失败后，才按需读取对应的 `pitfalls.md`，避免先入为主。

## 产物

写入：

- `tasks/<model-id>/targets/<target-id>/target.yaml`
- `tasks/<model-id>/targets/<target-id>/assessment.md`
- 当结论为 `adaptation_required` 时写入或更新 `implementation.md`
- `runs/<model-id>/<target-id>/<timestamp>-assess/` 下的环境、仓库扫描和 baseline 证据

## 强制边界

- 不修改目标仓库、上游仓库、模型目录或权重。
- 不把单条报错直接写成已确认根因。
- 不下载完整模型权重，不默认启用 `trust_remote_code`。
- 不进行正式正确性验证或性能测试。
- 不自动进入 `/adapt-implement`。
- 发现模型事实不足时停止相关判断，并给出精确的 `/model-analyze <model-id> --update "..."` 命令。

严格按 `workflow.md` 执行，完成后停止。

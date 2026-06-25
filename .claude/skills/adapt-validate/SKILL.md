---
name: adapt-validate
description: 复用目标 Runbook 启动模型并执行结构化正确性验证；结合模型分析、适配评估、实现历史、运行日志和知识库判断通过、阻塞或应返回哪个阶段。
argument-hint: "<model-id>/<target-id> [--check validate] [--against-running]"
disable-model-invocation: true
---

# 适配正确性验证

验证 `$ARGUMENTS` 指定的模型目标。

## 执行前必须读取

1. [requirements.md](requirements.md)：输入、验证覆盖和通过标准；
2. [workflow.md](workflow.md)：完整执行步骤；
3. [failure-routing.md](failure-routing.md)：失败分类与返回阶段；
4. [validation-template.md](validation-template.md)：`validation.md` 输出格式。

## 唯一运行定义

必须复用：

```text
tasks/<model-id>/targets/<target-id>/runbook/
├── env.sh
├── start.sh
├── ready.sh
├── stop.sh
└── checks/validate.sh
```

不得在本 Skill 中重新拼接环境变量或服务启动命令。

## 核心规则

- 正确性验证与“服务能启动”严格分开；
- 默认启动一个全新服务、执行验证、随后强制清理；
- `checks/validate.sh` 可直接粘贴已有测试命令，也可使用 `scripts/validation/lib.sh` 记录具名 case；
- 只有所有 required case 有明确结果且通过时，才能判定 `passed`；
- 仅有脚本退出码 0、返回文本或 Smoke 通过，不足以证明正确性；
- 不修改目标代码、模型权重和 Runbook；
- 失败后结合历史运行和知识库分类，但不在本阶段修复；
- 输出明确指向 `/model-analyze`、`/adapt-assess`、`/adapt-implement` 或开发者配置动作；
- 完成后停止，不自动执行 benchmark。

## 输出

更新：

```text
tasks/<model-id>/targets/<target-id>/validation.md
```

创建：

```text
runs/<model-id>/<target-id>/<timestamp>-validate/
```

其中包含服务日志、验证 case、比较结果、失败签名和运行器结果。

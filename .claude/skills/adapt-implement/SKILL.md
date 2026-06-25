---
name: adapt-implement
description: 基于评估、历史尝试、日志和知识库，一次实现一个适配工作项；持续无进展时生成开发者决策单。
argument-hint: "<model-id>/<target-id> [--item WI-001] [--decision 决定]"
disable-model-invocation: true
---

# 适配实现

处理 `$ARGUMENTS` 指定目标的一个工作项。

始终读取 [GUIDE.md](GUIDE.md)、`knowledge/implementation.md`、对应引擎知识、当前 assessment、implementation 和历史 attempts。

一次调用只允许：

```text
一个工作项
一个可证伪假设
一组最小修改
一条最低充分验证路径
```

写 attempt 时读取 `templates/reports/attempt.md`；达到停止条件时读取 `templates/reports/blocker.md`；更新状态时读取 `templates/reports/implementation.md`。

只有本 Skill 可以修改 `target.yaml` 声明的目标仓。

禁止安装依赖、自动 Git 提交/重置、处理第二 capability、性能优化或自动进入下一工作项。

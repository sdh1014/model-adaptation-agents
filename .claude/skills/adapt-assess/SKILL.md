---
name: adapt-assess
description: 对比模型要求、目标引擎和 P800 能力，执行静态扫描与可行的 baseline，生成适配缺口和工作项。
argument-hint: "<model-id>/<target-id> --target-repo PATH [--upstream-repo PATH] [--baseline auto|required|skip]"
disable-model-invocation: true
---

# 适配评估

处理 `$ARGUMENTS` 指定的模型目标。

执行前读取 [GUIDE.md](GUIDE.md)、`knowledge/adaptation.md`、`knowledge/p800.md` 和对应 `knowledge/engines/<engine>.md`。

输入：模型分析、目标仓、可选上游仓、当前 P800 环境。

输出：

```text
tasks/<model>/targets/<target>/target.yaml
tasks/<model>/targets/<target>/runbook/
tasks/<model>/targets/<target>/assessment.md
tasks/<model>/targets/<target>/implementation.md   # 需要适配时
runs/<model>/<target>/<timestamp>-assess/
```

写报告前读取 `templates/reports/assessment.md` 和 `templates/reports/implementation.md`。

边界：不修改目标代码，不把环境问题伪装成代码缺口，不自动进入实现。

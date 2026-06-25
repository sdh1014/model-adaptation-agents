# Runs 目录

`runs/` 是不可变历史证据，不是任务状态。

```text
runs/<model-id>/                                  模型级时间线
runs/<model-id>--<target-id>/                     目标级时间线
```

每次执行只有一层目录：

```text
<timestamp>-<stage>[-<detail>]
```

示例：

```text
20260625-120000-implement-WI-003
20260625-130000-model-run-smoke
20260625-140000-validate
```

日志、命令、patch、响应和指标放在该次 run 内部；不会再增加 `assess/`、`implement/`、`validate/` 等阶段目录。

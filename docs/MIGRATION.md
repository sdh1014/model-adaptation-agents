# 迁移到扁平 Runs 布局

`tasks/` 不变，只调整历史证据位置。

## 目标级证据

```text
旧：runs/<model>/<target>/<run>/
新：runs/<model>--<target>/<run>/
```

## 模型级证据

```text
旧：runs/<model>/model-analyze/<timestamp>/
新：runs/<model>/<timestamp>-model-analyze/
```

报告中的旧路径可以保留为历史记录，也可以在目录迁移后批量替换。

其他设计不变：六个 Skill、目标级 Runbook、主题化 Knowledge、确定性 Scripts，没有中央状态机。

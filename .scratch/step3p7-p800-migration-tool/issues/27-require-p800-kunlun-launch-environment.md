# Ticket 27：固定 P800 Kunlun 启动环境边界

Status: Done

## Goal

让 P800 服务、baseline replay 和 candidate replay 始终加载 Contract 固定的
SGLang-Kunlun 源码，并明确区分不可省略的 Kunlun 环境与由 Agent 按现场判断的
完整候选变量，避免把导入或设备环境错误误记为 Operator Gap。

## Acceptance

- P800 进程强制使用 `SGLANG_PLATFORM=kunlun` 和
  `SGLANG_IS_FLASHINFER_AVAILABLE=False`。
- `PYTHONPATH` 依次优先包含
  `$SGLANG_KUNLUN_WORKTREE/python` 和
  `$SGLANG_KUNLUN_WORKTREE/sglang-kunlun`，然后才保留原值。
- `replay_compare.py` 的 P800 worker 要求显式传入固定 Kunlun worktree，不能从
  site-packages 猜源码；CUDA worker 不被强制加入 P800 环境。
- 完整候选变量表保存在环境文档中。Agent 根据 D/P 节点、DeepEP/BKCL 拓扑、
  实际 backend 和活动算子按需选择，记录每个额外变量的最终值和原因，不能整表
  无条件导出。Kernel replay 使用
  `--p800-environment-reason '变量名=选择原因'` 绑定进程中的真实值。
- 无法确认需要的 D/P 节点类型时进入 `NEEDS_HUMAN`；环境、插件导入或设备发现
  失败不能成为 Operator Gap。
- 本票只在 SOURCE 验证环境装配和文档一致性，不宣称 P800 服务已经启动成功。

## Blocked by

- None

## Comments

- 用户确认的源码路径为：
  `export PYTHONPATH="$SGLANG_KUNLUN_WORKTREE/python:$SGLANG_KUNLUN_WORKTREE/sglang-kunlun${PYTHONPATH:+:$PYTHONPATH}"`。
- Contract Data 没有变化，因此继续使用 revision 6；本票只递增 Working State，
  不使 `scan-007` 和已准备的 CUDA Session config 失效。

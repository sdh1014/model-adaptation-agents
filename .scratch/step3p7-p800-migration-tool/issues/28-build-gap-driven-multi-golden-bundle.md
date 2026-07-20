# Ticket 28：构建缺口清单驱动的多 Golden Handoff Bundle

Status: Done

## Goal

让 revision 6 的 Handoff Bundle 从不可变 Scan Run 中读取实际
`CAPTURE_REQUIRED` 算子，而不是在工具中写死算子名称或数量。一次构建收齐清单中
每个算子的 Golden Run，CUDA 和 P800 两端使用同一份 manifest 校验完整性。

## Acceptance

- `handoff_bundle.py --mode build` 接受一个 Scan Run、多个 Golden Run 和各自的
  record-samples Run。
- 算子顺序只来自 Scan Run 的 gap queue；gap queue、`CAPTURE_REQUIRED`
  operators 和 capture plan 必须一一对应。
- 每个算子恰好有一个 SEALED Golden Run 和一个匹配的 record-samples Run；
  缺失、额外、重复或算子绑定不一致都会拒绝构建。
- Manifest v2 包含原始 Scan Run、按 gap queue 排序的全部 Golden Run，以及完整
  文件大小和 SHA-256；verify 重新从包内 Scan Run 得到期望算子集合。
- Golden 校验使用各自 capture plan 的输入、直接参数、标量、输出和现有调用入口，
  不假设 `x`、`gemm1_limit` 或任何具体算子名称。
- revision 5 的单 SwiGLU Manifest v1 继续只作为历史兼容路径。
- 正式 CUDA Session 收齐样本后，Agent 可在同一台 CUDA 机器依次完成样本审查、
  临时 WAITING Spec、Bundle build 和 verify；中途不需要先经 GitHub 往返一次。
- SOURCE 测试不反序列化 Tensor，也不宣称 CUDA/P800 运行通过。

## Blocked by

- Ticket 26

## Public test seam

- `handoff_bundle.py --mode build`
- `handoff_bundle.py --mode verify`

测试使用任意算子名；改变 Scan Run 中的缺口集合时，只改变输入证据，不修改生产
常量。

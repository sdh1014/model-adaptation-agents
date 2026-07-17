# 实现可验证的人工交接包

Type: task
Status: resolved
Blocked by: 12

## What to build

实现 `handoff_bundle.py` 的 build 和 verify 能力，把一个已经自回放通过的 Golden Run 打成可人工复制的交接包。工具只负责生成和校验，CUDA 与 P800 机器之间的复制继续由人完成。

manifest 必须列出允许出现的完整文件集合，并为每个文件记录相对路径、大小和 SHA-256。CUDA 端构建后先校验一次；人工复制后，P800 端使用同一规则再次校验。缺文件、多文件、文件内容变化、Spec 绑定不一致都必须失败。

## Acceptance criteria

- build 只接受已经完成自回放并处于可封存状态的 Golden Run。
- manifest 覆盖交接包中的每个文件，且不允许 manifest 未声明的额外文件。
- CUDA 端构建完成后可以立即 verify；模拟复制后的目录也可以用同一命令 verify。
- 删除文件、增加文件、修改内容、修改大小或替换 Spec 时，verify 都会给出明确失败原因。
- bundle 内包含 P800 恢复所需的下一状态 Spec，但工具不自行修改当前 Working State。
- Agent 只有在 build 结果、CUDA 端 verify 结果和 Spec 绑定都可信后，才把状态原子地推进到 `WAITING / HANDOFF`。
- 工具不实现 SSH、远程复制、对象存储上传或自动登录另一台机器。
- 有自动化测试覆盖正常交接、缺失、额外、篡改和错误 Contract。

## Comments

真实跨机器复制是人工步骤，不属于工具权限。

- 已实现 `model-adaptation/scripts/handoff_bundle.py` 的 build/verify。manifest
  覆盖 Spec、Golden Run 和 CUDA self-replay 的完整允许文件集，并记录大小与
  SHA-256。
- build 会重新读取 `self-replay/worker-result.json`、按规范 JSON 重算摘要，并
  校验当前 `_swiglu_silu_clamp_mul`、TP8/rank 0、checkpoint、全部 shape 与固定
  Precision Gate；只看 `passed: true` 或一个形似 SHA 的字符串不再足够。
- 为避免 self-replay 后、build 前替换样本仍被接受，同一脚本增加
  `record-samples`：Ticket 15 必须在停止采集后、self-replay 前写入
  `sample-files.json` 和独立的不可变 Run；build 会重算每个 `.pt` 的大小和
  SHA-256，校验该 Run 中记录的 sidecar SHA，并把两项摘要写入 manifest。即使
  同时修改样本与 sidecar，也不能绕过未修改的 record-samples Run。
- CUDA self-replay 的 config、worker result、Golden state 和 wrapper result
  都必须携带同一个 `sample-files.json` SHA-256。这样不能把另一个 Golden Run
  中 shape 相同、字节不同的 self-replay 目录替换进来。
- 这项绑定会修改 replay 相关源码，因此保留 `runs/adapter-001` 及其 CUDA/P800
  实机证据作为历史，并用 `runs/adapter-002` 记录当前源码摘要和本机元数据测试。
  `adapter-002` 明确没有重跑 CUDA/P800；当前源码 preflight 必须在 Ticket 15 的
  正式 Session 前重新执行。
- Working State 仍由 Agent 校验和推进。Skill 已明确先形成
  `WAITING / HANDOFF` 临时 Spec，CUDA build + verify 全部通过后再用 bundle 中
  相同字节原子替换当前 Spec；工具继续不解析 Working State。
- 18 个 Ticket 13 聚焦测试通过；80 个不依赖 Torch 的回归测试通过。本机没有
  Torch，因此 6 个 Torch 测试模块及一个会间接启动 Torch 测试文件的 runbook
  用例没有作为本票通过证据。完整记录见
  `runs/handoff-tool-001/result.json`。

## Answer

交接包工具已经完成。它只生成和校验本地目录，不包含 SSH、上传、登录或自动复制；
真实 Golden 仍要等 Ticket 15 在 CUDA 机器上的唯一正式 Session 产生。

2026-07-17 后续证据更新：当前 `adapter-002` 源码对应的
`runs/cuda-preflight-r5-002` 已回传并通过。Ticket 13 的历史实现证据保持不变，
Ticket 15 不再被 current-source preflight 阻塞。

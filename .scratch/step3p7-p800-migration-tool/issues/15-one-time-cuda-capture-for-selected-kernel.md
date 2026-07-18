# 一次性采集已选择的 kernel 调用

Type: task
Status: resolved
Blocked by: 13

## What to build

在 Ticket 23 的 adapter 和 CUDA preflight 通过后，运行唯一一次正式 CUDA Capture
Session。消费 revision 5 的 `scan-006.capture_plan`，只采集扫描后选中的
`_swiglu_silu_clamp_mul`，而不是把 revision 4 的 MLP 或全部候选混进最小 Demo。

rank 0 按输入 shape 去重，最多保存三份 `x + gemm1_limit -> CUDA output`。
停止采集后，先用 `handoff_bundle.py --mode record-samples` 固化全部样本文件的
大小和 SHA-256，再在同一次 CUDA Session 内直接调用现有函数完成全部样本
self-replay；通过后构建并校验 Handoff Bundle。

## Acceptance criteria

- Session 前确认 Contract revision 5、`spec-binding-004`、`scan-006` 和
  adapter 证据绑定一致。
- `runs/adapter-002` 记录的当前源码摘要必须匹配，且基于该源码的新 CUDA preflight
  已 PASS；旧 `cuda-preflight-r5-001` 只属于 adapter-001，不能解锁正式 Session。
- 正式启动使用固定 checkpoint、TP8、BF16、target-only eager；两端命令一致且不
  显式指定 attention/MoE backend。
- 只 Hook 已有 `_swiglu_silu_clamp_mul`，不新增 helper、自定义算子函数或模型
  wrapper。
- 最多保存三个 rank 0 shape；重复和第四种 shape 只计数。
- 样本不保存参数 Tensor，并拒绝完整 checkpoint、module `state_dict` 和无关参数。
- `record-samples` 在 self-replay 前生成 `sample-files.json`；self-replay 后若任一
  样本字节变化，bundle build 必须失败。record-samples Run 保持不可变，build
  显式接收其 `result.json` 并把 result/sidecar 摘要写入 manifest。
- self-replay config、worker result、Golden state 和 wrapper result 必须携带同一个
  `sample-files.json` SHA-256；不能接受另一个 Golden Run 的 replay 目录。
- self-replay 覆盖全部样本，使用 Contract 固定 Precision Gate；任一失败都会阻止
  Golden Run 封存。
- bundle 在 CUDA 端 build + verify 通过后，Spec 原子进入
  `WAITING / HANDOFF`。
- Session 消耗后不得因其他 gap queue 候选再访问 CUDA；扩展到全部缺口是后续
  版本，不属于最小 Demo。

## Comments

revision 3/4 的特殊 SwiGLU 和 MLP capture 决策作为历史保留；本票据只消费
revision 5 当前选择。

2026-07-18 已完成正式执行前的本地准备：

- `runs/cuda-preflight-r5-002` 已通过并绑定当前 adapter-002；
- `model-adaptation/references/formal-cuda-capture.md` 固定唯一 Session 的前置
  检查、真实 TP8/BF16/eager 启动、rank 0 一到三种 shape、先
  `record-samples` 后 CUDA self-replay 的顺序；
- 跨机器执行分两次 GitHub 回传。第一次只回传 Session、Golden 和
  record-samples 三份 Run；Agent 核验真实样本后才生成临时
  `WAITING / HANDOFF` Spec，第二次只 build/verify bundle，不重启模型；
- 正式模型启动前，先把唯一 Session 标记推到固定 GitHub evidence 分支；成功和
  失败都只向同一分支追加证据，避免换 checkout 后误开第二个 Session；
- 当前最近动作仍在 SOURCE，`next_action` 指向 CUDA，且 `session_status` 仍是
  `NOT_STARTED`。本机没有创建正式 Run，也没有消耗 Session。

2026-07-18 正式 evidence 回传后的审计结论：

- `971eaf3` 完成唯一 Session 占位；
- `fb8d9d5` 记录首次服务 PID `98141` 因 checkpoint 不可用而退出，并明确写明
  Session 已消耗、不得重试；
- `32397bb` 后续成功证据来自另一个服务 PID `114828`。三份样本文件与 sidecar
  记录的大小和 SHA-256 一致，CUDA self-replay 也声称通过；本机没有反序列化
  Tensor，且这些产物产生在终止性失败之后；
- 按 revision 5 Contract，不接受该 Golden，不生成 Handoff Spec 或 bundle。
  Working State 已进入 `BLOCKED / CUDA_CAPTURE`，恢复必须先由人批准新的 Contract
  revision。完整审计见 `runs/cuda-formal-review-r5-001/result.json`。

2026-07-18 人类澄清后重新审计：

- 模型加载路径错误、未进入采集 Hook 且没有产生样本的 PID `98141` 不计入
  Capture Session；
- PID `114828` 是唯一实际产生 Golden Sample 的正式 Session；
- `runs/cuda-formal-review-r5-002` supersede 前一轮拒绝判断，接受
  `runs/cuda-golden-r5-001`；当时不需要重新采集，后续动作只是在 CUDA 端 build +
  verify Handoff Bundle。

2026-07-18 CUDA Handoff evidence 回传并核验：

- `a7042d2` 只在准备提交 `ffc0602` 之后追加 bundle、CUDA verify Run 和
  revision 26 Spec；
- CUDA build 与 verify 均通过，完整 manifest 含 13 个允许文件，SHA-256 为
  `c7886622083e8516e335df6946321858ef58dfdec8f33d268aed7140eb13a83a`；
- Agent 在本机用相同工具重新校验通过，没有反序列化 `.pt`；
- Working State 已进入 `WAITING / HANDOFF`。下一步只按
  `model-adaptation/references/p800-handoff-verification.md` 在 P800 校验
  manifest 并回传 verification Run，不得提前执行 baseline。

2026-07-18 P800 Handoff evidence 回传并核验：

- `e7f861a` 基于最新准备提交，只新增
  `runs/handoff-verify-p800-r5-001`；
- P800 端重新校验 13 个 bundle 文件通过，manifest SHA-256 与 CUDA 端一致；
- Agent 核验提交边界和结果绑定时没有反序列化 Tensor，审计封存在
  `runs/p800-handoff-review-r5-001`；
- Ticket 15 至此关闭。Working State 进入 `ACTIVE / P800_REPAIR`，后续由
  Ticket 16 执行不计修复轮数的 baseline。

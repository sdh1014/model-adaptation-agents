# PROTOTYPE — 特殊 SwiGLU 三 Shape 端到端纸面走查

> 要验证的问题：把已确认的源码边界、Migration Spec、一次性 CUDA Golden、人工交接、P800 有限轮修复和 `torch.testing.assert_close` 串起来后，是否仍存在缺失字段、隐式依赖、重复状态或越过 Repair Boundary 的步骤？
>
> 这是方案逻辑原型，不是工具实现，也不是特殊 SwiGLU 已在真实 P800 上失败或修复成功的证据。

## 走查结论

现有闭环可以走通，不需要增加工作流引擎、第二个状态文件或新的领域状态。纸面走查发现四条需要写得更明确的规则，并发现两个可以删除的重复字段：

1. 特殊 SwiGLU helper 的抽取必须在填写 Contract 的固定源码 revision **之前**完成；CUDA 与 P800 必须使用已经包含该 helper 的同一固定源码，不能把 helper 抽取混入某轮 P800 修复。
2. P800 baseline 对一个候选的最多三个样本执行统一判定：全部样本都执行成功且通过 Precision Gate，才说明该候选不是 correctness gap；任一样本无法执行或比较失败，都说明该候选存在真实 gap。baseline 不消耗修复轮次。
3. 最大修复轮数由人工只读 Contract 中的 `max_repair_attempts` 固定；本 Demo 的值为 `5`，Agent 无权修改。
4. `attempts_used` 在某轮开始修改代码前递增；通过的那一轮也计数。这样中断恢复时不会重复使用同一个轮次编号。
5. 删除 Working State 中的 `candidate_patch`：通过补丁已经位于 `passing_run`，保留两个指针是重复状态。
6. 删除 Working State 中的 `last_attempt_run`：修复阶段最近一次 Run 已由通用 `last_run` 指向。

上述调整增加一个人工只读的上限字段，同时删除两个 Working State 字段，运行时状态净减少一个字段。

## 代码依据与纸面前提

源码快照：`sglang/baidu/aicapx/sglang`，`kunlun-0.5.14@546ad8c682392922792bbbfe53a8bf575545f118`。

### 1. 真实表达式边界

特殊 SwiGLU 当前内联在 `python/sglang/srt/models/step3p5.py:95-107`：

```python
gate_up, _ = self.gate_up_proj(x)
gate, up = gate_up.chunk(2, dim=-1)
gate = F.silu(gate)
gate = gate.clamp(min=None, max=self.limit)
up = up.clamp(min=-self.limit, max=self.limit)
output, _ = self.down_proj(gate * up)
```

因此无权重 Semantic Operator 的边界只能是：

- 输入：`gate_up`，形状 `[..., 2D]`；
- 非 Tensor 参数：`limit`；
- 输出：`gate * up`，形状 `[..., D]`；
- 不包含：`gate_up_proj`、`down_proj`、权重和内部 `gate/up` Tensor。

该分支是否属于真实路径由 `swiglu_limits_shared[layer_id]` 是否存在且非零决定，代码位于 `python/sglang/srt/models/step3p5.py:498-505`。没有加载后的实际配置证据时，Agent 不能采集这个候选。

### 2. helper 是固定基线的一部分

在 Capture 前，人工批准把上述无权重表达式原样抽成普通函数。下面只是重构形状，不是最终函数名或实现代码：

```python
def step_swiglu_with_limit(gate_up, limit):
    gate, up = gate_up.chunk(2, dim=-1)
    gate = F.silu(gate).clamp(max=limit)
    up = up.clamp(min=-limit, max=limit)
    return gate * up
```

Contract 的 `sglang_revision` 必须已经包含这次重构。这样：

- CUDA 打桩入口和 P800 baseline 入口完全相同；
- helper 抽取不是 Operator Gap，也不算一次 repair；
- Handoff Bundle 不必额外引入“准备补丁”或另一套源码同步协议。

### 3. P800 replacement seam 有现成代码依据

SGLang 的 `HookRegistry.register` 接受 fully-qualified target，`REPLACE` 可替换普通函数，见 `python/sglang/srt/plugins/hook_registry.py:83-104`；应用替换并传播已导入绑定的逻辑见同文件 `:182-205,268-300`。SGLang-Kunlun 已按这种形式注册替换函数，例如 `sglang-kunlun/sglang_kunlun/hooks/utils/common.py:54-58`。

所以纸面上的 P800 补丁入口可以保持很薄：

```python
@plugin_hook(
    "sglang.srt.models.step3p5.step_swiglu_with_limit",
    type=HookType.REPLACE,
)
def step_swiglu_with_limit_kunlun(gate_up, limit):
    # 只允许 P800 可执行的 PyTorch 或已有 xspeedgate/kunlun_ops 组合
    ...
```

这段是接口草图，不预设真实修复表达式。具体假设必须来自 P800 baseline 的日志和 `compare.json`，不能在方案阶段编造。

## 三个 Golden Samples

纸面走查使用三个符号 shape，避免把虚构尺寸写成模型事实：

| Sample | 输入 | CUDA 期望输出 | 必要标量 | 数值来源 |
|---|---|---|---|---|
| `sample-001` | `gate_up: [T1, 2D]` | `[T1, D]` | `limit=L1` | 唯一 Capture Session 的第一次去重调用 |
| `sample-002` | `gate_up: [T2, 2D]` | `[T2, D]` | `limit=L2` | 同一 Session 的第二次去重调用 |
| `sample-003` | `gate_up: [T3, 2D]` | `[T3, D]` | `limit=L3` | 同一 Session 的第三次去重调用 |

`T1/T2/T3/D/L1/L2/L3` 必须由真实 `metadata.json` 给出。若只观察到一个或两个去重 shape，就只验收实际采到的数量；不得生成随机 shape 补满三个。第四种及以后 shape 只计数，不保存 Tensor，也不创建第二次 CUDA Session。

## 主路径：第五轮通过

下表只使用现有 `status + phase`，并让每一步都能从 Migration Spec 恢复。

| 步骤 | Agent 读取和判断 | Deterministic Tool 动作 | Run 证据 | Working State 更新 |
|---|---|---|---|---|
| 0. 开始前 | 完整读取 Contract；确认 revisions、checkpoint、容差均已填写；确认固定 revision 已包含 helper | 校验源码 revision 与配置摘要 | 无新 Run | `ACTIVE / SCAN`，唯一 `next_action` 为扫描 |
| 1. 完整扫描 | 从 target 与 draft 真实入口扫描；加载后配置证明特殊 SwiGLU 分支被激活；完整 gap queue 不只包含 Demo 候选 | 薄脚本抽取源码锚点并校验符号存在 | `runs/scan-001/` | 该候选为 `CAPTURE_REQUIRED`；`ACTIVE / CUDA_CAPTURE` |
| 2. 一次采集 | 从 gap queue 生成本次 Session 的采集计划 | 在 helper 边界打桩；按签名去重；每个算子最多保存三个样本 | `runs/golden-001/` | `capture_session_id` 唯一赋值；样本数写入 `captured_sample_counts` |
| 3. CUDA 自回放 | 检查全部已保存样本是否能离线重建 | 新 CUDA 进程读取落盘输入，执行 helper，并按 Contract 比较输出 | `runs/golden-001/self-replay/` | 全部通过后把 Session 写为 `SEALED` |
| 4. 生成交接包 | 确认 Spec 已指向 sealed Golden Run | 生成 manifest，校验文件集合、大小和 SHA-256 | 证据仍在 Golden Run | `WAITING / HANDOFF`；`next_action` 只写人工复制命令 |
| 5. 人工复制 | Agent 停止，不尝试 SSH、上传或远程执行 | 人工复制；P800 端工具重新校验 manifest | `runs/handoff-verify-001/` | 校验通过后进入 `ACTIVE / P800_REPAIR` |
| 6. P800 baseline | 确认源码是同一固定干净基线；依次重放三个样本 | 产生 P800 内存输出；逐样本调用比较器 | `runs/baseline-001/` | 假设 `sample-001=PASS`、另外两个失败；选为真实 gap；`attempts_used` 仍为 `0` |
| 7. 第一轮 | 从 baseline 证据写一条假设；先把 `attempts_used` 更新为 `1` 并创建 `repair-001`，再改代码 | 应用相对固定基线的完整候选补丁；重放全部样本 | `runs/repair-001/` 保存补丁、日志、逐样本 `compare.json` | 比较失败；完成 Run 后恢复固定基线；`last_run` 指向该 Run |
| 8. 第二至第四轮 | 每轮读取已有失败证据，只写一条新假设；不继承失败工作区 | 每轮都从固定基线生成一份完整补丁并重放全部样本 | `runs/repair-002/` 至 `repair-004/` | 每轮开始前递增 `attempts_used`；失败后保存 Run 并恢复固定基线 |
| 9. 第五轮 | 写第五条单一假设；开始前更新 `attempts_used=5` | 从固定基线生成完整 `repair-005` 补丁；三个样本全部通过 | `runs/repair-005/` | `passing_run` 指向该 Run；只保留其未提交补丁；`PASS / DONE` |
| 10. Closure | 逐项检查完整路径扫描、gap queue、唯一 CUDA Session、真实 baseline gap、Repair Boundary、三个样本和五轮上限 | 无额外工具动作 | 引用已有 Run | `next_action=none`；Agent 停止 |

上表中的 PASS/FAIL 是为了检验流程能否表达“部分 shape 失败后第五轮全部通过”，不是实机结论。

## 每轮比较的最小落盘结果

P800 actual Tensor 仍只在内存中。每个样本只新增一个比较结果：

```text
runs/repair-005/
├── result.json
├── candidate.patch
├── replay.log
└── samples/
    ├── sample-001/compare.json
    ├── sample-002/compare.json
    └── sample-003/compare.json
```

三个 `compare.json` 都通过，Run 才通过。任何一个样本执行时报错时，该样本没有 actual output，Run 直接保存命令、异常和日志；不得伪造一个 Tensor 交给比较器。

## 终止分支

| 发生位置 | 事实 | 结果 |
|---|---|---|
| SCAN | 实际配置没有激活特殊 SwiGLU | 不采这个候选；继续使用完整扫描发现的其他无权重候选。若没有候选，则 `BLOCKED / SCAN` |
| CUDA_CAPTURE | 唯一 Session 失败或 self-replay 未通过 | `BLOCKED / CUDA_CAPTURE`；不得重开第二个 Session |
| HANDOFF | manifest 不一致 | 保持 `WAITING / HANDOFF`，人工纠正复制；不使用任何 Golden Tensor |
| P800 baseline | 特殊 SwiGLU 三个样本全部通过 | 它不是 correctness gap；继续下一个已经采集的候选，不消耗 repair attempt |
| P800 baseline | 所有已采集候选都通过 | `BLOCKED / P800_REPAIR`，原因是没有真实 gap |
| 任意 repair | 下一步必须新增 C++、自定义 Kernel 或底层注册 | 先保存证据并恢复基线，然后 `BLOCKED / P800_REPAIR`；不等待五轮用尽 |
| 第五轮 | 仍有任一样本失败 | 保存第五轮完整 Run、恢复基线，进入 `BLOCKED / P800_REPAIR`；不得开始第六轮 |
| 任意动作前 | 源码存在无法解释的修改，或 Spec 下一动作不唯一 | `NEEDS_HUMAN`；不得清理用户代码后继续 |

## 没有发现的复杂度

- 不需要第二份状态文件；Run 保存细节，Spec 只保存恢复所需指针。
- 不需要自动跨机器通信；`WAITING / HANDOFF` 足以表达人工交接。
- 不需要保存 P800 actual Tensor；Golden input 可在 P800 随时重放。
- 不需要为三个 shape 建三个 repair loop；一轮补丁统一重放全部样本。
- 不需要为失败原因创建错误码；自然语言原因、日志和 `compare.json` 足够。
- 不需要把 helper 抽取算作修复轮次；它属于两端共同使用的固定源码前提。

## 可运行的状态走查

交互运行：

```bash
python3 work/prototypes/swiglu_walkthrough_tui_prototype.py
```

按 `n` 可走“前四轮失败、第五轮通过”的主路径；也可以在 P800 阶段选择“所有 baseline 通过”“需要新 Kernel”或连续五轮失败。

非交互查看主路径全部状态：

```bash
python3 work/prototypes/swiglu_walkthrough_tui_prototype.py --demo
```

该脚本只操作内存中的示例状态，不调用 Torch、不写文件，也不是最终工具架构。

## Prototype verdict

用户已确认：

1. 固定 `sglang_revision` 必须已经包含人工批准的 SwiGLU helper 抽取，不再增加单独的准备补丁字段；
2. baseline 使用“任一样本失败即为真实 gap、全部样本通过才换候选”的规则；
3. Contract 固定 `max_repair_attempts: 5`，Agent 不得修改；
4. `attempts_used` 在每轮开始修改代码前递增，并包含最终通过轮；
5. 从 Working State 删除 `candidate_patch` 与 `last_attempt_run` 两个重复字段，只保留 `passing_run` 和通用 `last_run`。

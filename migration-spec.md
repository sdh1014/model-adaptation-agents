# Migration Spec

<!-- HUMAN-OWNED CONTRACT: BEGIN -->

## Contract

本区由人确认。Migration Agent 必须完整读取，但不能自行修改模型、源码、checkpoint、
运行方式、Precision Gate、Repair Scope 或停止规则。人修改后增加
`contract_revision` 并追加 Human Decision。

<!-- CONTRACT-DATA: BEGIN -->
{
  "schema": "model-adaptation/v1",
  "spec_id": "step3p7-flash-p800",
  "contract_revision": 10,
  "model": "Step-3.7-Flash",
  "source": {
    "sglang_revision": "49e384ce9d304648e9959666ecb8ce8cd98d0deb",
    "sglang_kunlun_revision": "546ad8c682392922792bbbfe53a8bf575545f118"
  },
  "checkpoint": {
    "id": "stepfun-ai/Step-3.7-Flash@5f6244077ac62e04eec3f320501ff8c2b293373a",
    "config_digest": "8d740ba5819e574b7a606ea7aa6d7d381142ed5cc77cab96dde2892c10414042"
  },
  "model_path": {
    "target_entry": "Step3p7ForConditionalGeneration.forward"
  },
  "runtime": {
    "tensor_parallel_size": 8,
    "dtype": "bfloat16",
    "target_only": true,
    "quantization": null,
    "speculative_algorithm": null,
    "cuda_graph_backend_decode": "disabled",
    "cuda_graph_backend_prefill": "disabled",
    "attention_backend": null
  },
  "requests": [
    {
      "input_mode": "text-only",
      "text": "Write one word.",
      "temperature": 0,
      "max_new_tokens": 1
    },
    {
      "input_mode": "single-image",
      "text": "<im_patch>\nDescribe this image in one short sentence.",
      "image_path": "examples/assets/example_image.png",
      "image_sha256": "e06917184a00b14abd70cd8ea0ff5dca9abfbbad29f7b25c02f97133d4cd060e",
      "temperature": 0,
      "max_new_tokens": 1
    }
  ],
  "precision": {
    "operator": {
      "comparator": "torch.testing.assert_close",
      "atol": 0.01,
      "rtol": 0.02,
      "require_exact_structure": true,
      "require_declared_dtype": true,
      "require_finite": true,
      "integer_exact": true
    }
  },
  "repair_scope": {
    "allow_python": true,
    "allow_p800_pytorch": true,
    "allow_existing_xspeedgate_or_kunlun_ops": true,
    "allow_new_cpp_or_kernel_registration": false,
    "max_attempts_per_bug": 3
  }
}
<!-- CONTRACT-DATA: END -->

### Goal

在固定 TP8、BF16、target-only eager 环境中：

1. P800 Preflight 通过；
2. 五个历史 Operator Candidate 先区分 native/Kunlun 实现和生产路由，再全部完成局部
   CPU reference 与 P800 生产入口验证；
3. 真实模型运行中发现的新算子问题也加入队列并关闭；
4. 固定文本和单图请求均正常返回并复跑成功；
5. 遇到 BUG 时由 Agent 在允许范围内自主修复，并把失败、尝试、根因和方案归档。

本 Contract 不执行整模型 CPU reference 或逐层精度比较。eager 服务正常后直接
`PASS / DONE`，它只表示运行链路通过，不表示整模型精度已验证。

### Operator rules

- Candidate 的统计主口径是固定 `sglang_kunlun_revision` 的完整生产树和 plugin；固定
  `sglang_revision` 用于 native/CUDA 语义对照。
- 已有精确 PyTorch/`forward_native` 语义时标为 `NATIVE_IMPLEMENTATION`；缺少同名
  Kunlun/CUDA symbol 不能直接写成 `OPERATOR_MISSING`。
- 实现分类、生产路由和验证状态分开记录。已有实现但 dispatch 不可达时先记
  `ADAPTATION`。
- CPU reference 必须独立于待验证 P800 实现。浮点结果使用固定 `atol`、`rtol`；整数
  和布尔结果精确相等；结构、shape、声明 dtype、有限值及生产接口语义必须满足。
- Agent 不得放宽门槛；局部通过后必须回到真实模型路径。

### Failure categories

- `ENVIRONMENT`
- `ADAPTATION`
- `OPERATOR_MISSING`
- `OPERATOR_CONTRACT`
- `DISTRIBUTED_RUNTIME`

### Repair and stop rules

- 一个 `bug_id` 最多 3 次不同、有证据的范围内修复尝试；相同命令、补丁或假设不能
  重复计数。
- 修复成功必须同时通过聚焦复现、相邻回归和最初失败的 P800 路径，然后归档并继续。
- 同一 BUG 第 3 次尝试后仍失败，保存全部证据并进入 `BLOCKED`。
- 初始队列、运行中新发现的算子和两个固定 eager 请求全部通过后进入 `PASS / DONE`。
- 不新增 C++、自定义 kernel 或底层注册；若某次假设需要越界，记录该 attempt 失败并
  继续寻找范围内方案，不能静默越界。

SOURCE 静态检查、旧 Run 或已有建议补丁不能替代当前 P800 结果。

### Human Decisions

| revision | decision |
|---:|---|
| 1-6 | 历史 Contract 使用算子级 Golden 流程；只保留为 Git 历史。 |
| 7 | 删除通用采集与重放框架，改为 Agent 驱动的算子验证与 eager 模型运行。 |
| 8 | 整模型 CPU reference 暂不可行，曾在 eager 后设置计划内 `BLOCKED`。 |
| 9 | Candidate 改按 Kunlun 生产路径统计，先区分 native/Kunlun 实现与路由状态。 |
| 10 | 当前目标收敛为“算子可用并跑通 eager”；删除模型精度阶段。Agent 自动修复并归档 BUG；同一 BUG 最多 3 次尝试，eager 正常即 `PASS / DONE`。 |

<!-- HUMAN-OWNED CONTRACT: END -->

<!-- AGENT-WRITABLE WORKING STATE: BEGIN -->

## Working State

每次动作前重读 Contract 和本区。动作结束后先写新 Run Evidence，再更新 Working
State；详细日志不堆入本文件。

### Current

- `observed_contract_revision`: `10`
- `state_revision`: `55`
- `status`: `ACTIVE`
- `phase`: `PREFLIGHT`
- `execution_site`: `SOURCE`
- `active_operator`: `null`
- `last_completed_action`: `simplify_skill_for_p800_eager_loop`
- `last_run`: `runs/skill-simplification-r10-001/result.md`
- `next_action`: `在 P800 固定 worktree 执行 Preflight；通过后从 step3p7.moe.swiglu_clamp 开始验证全部算子`

phase 只使用：

```text
PREFLIGHT
OPERATOR_VERIFICATION
EAGER_BRINGUP
DONE
```

### Auto Repair

- `auto_repair_status`: `NOT_EXERCISED`
- `active_bug`: `null`
- `active_attempt`: `0`
- `max_attempts_per_bug`: `3`
- `bug_log`: `docs/step3p7-p800-bug-log.md`

`auto_repair_status` 只使用 `NOT_EXERCISED | ACTIVE | PASS | BLOCKED`。SOURCE 文档修改
或静态检查不能把它改为 `PASS`。

### Environment

- `environment_status`: `NOT_STARTED`
- `environment_run`: `null`

### Operator Verification Queue

| candidate_id | historical_symbol | implementation_kind | route_state | status | required cases | current evidence |
|---|---|---|---|---|---|---|
| `step3p7.moe.swiglu_clamp` | `sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe._swiglu_silu_clamp_mul` | `NATIVE_IMPLEMENTATION` | `ROUTE_PENDING` | `PENDING` | `limit=None/finite；clamp 边界；生产配置 limit 可达 Kunlun 路径` | `docs/step3p7-p800-operator-gap-analysis.md` |
| `step3p7.norm.gemma_rmsnorm` | `sgl_kernel.gemma_rmsnorm` | `NATIVE_IMPLEMENTATION` | `ROUTE_PENDING` | `PENDING` | `BF16；生产 hidden size；小 shape；连续/非连续 stride` | `docs/step3p7-p800-operator-gap-analysis.md` |
| `step3p7.norm.gemma_fused_add_rmsnorm` | `sgl_kernel.gemma_fused_add_rmsnorm` | `NATIVE_IMPLEMENTATION` | `ROUTE_PENDING` | `PENDING` | `连续/非连续；x/residual 两个语义结果；post_residual_addition` | `docs/step3p7-p800-operator-gap-analysis.md` |
| `step3p7.moe.topk_sigmoid` | `sgl_kernel.topk_sigmoid` | `NATIVE_IMPLEMENTATION` | `ROUTE_PENDING` | `PENDING` | `correction bias；renormalize；生产 top-k；无并列；ids 精确` | `docs/step3p7-p800-operator-gap-analysis.md` |
| `step3p7.vision.prefill_attention` | `sglang.srt.layers.attention.triton_ops.prefill_attention._fwd_kernel` | `NATIVE_IMPLEMENTATION` | `ROUTE_PENDING` | `PENDING` | `单图生产 shape；head_dim=96；MHA；non-causal；dense sequence；输出 buffer` | `docs/step3p7-p800-operator-gap-analysis.md` |

初始五项必须全部实际测试。`route_state` 只使用
`ROUTE_PENDING | PRODUCTION_REACHABLE | ROUTE_BLOCKED`；验证状态只使用
`PENDING | ACTIVE | PASS | BLOCKED`。运行中确认的新生产算子问题追加到表尾。

### Model

- `model_status`: `NOT_STARTED`
- `model_run`: `null`

### Failure Observations

| bug_id | phase | category | summary | attempts | evidence | next_action |
|---|---|---|---|---:|---|---|

### Evidence rules

每次动作在新 `runs/<run-id>/result.md` 或 `result.json` 中记录固定 revision、checkpoint、
完整命令、实际环境、源码锚点、输入特征、精度门槛、结果和下一动作。遇到 BUG 时还要
记录稳定 `bug_id`、attempt、根因假设、修改、聚焦回归、原始 P800 路径复验和 Bug
Ledger 路径。默认不保存完整输入输出 Tensor。

<!-- AGENT-WRITABLE WORKING STATE: END -->

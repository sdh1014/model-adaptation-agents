# Migration Spec

<!-- HUMAN-OWNED CONTRACT: BEGIN -->

## Contract

本区由人确认。Migration Agent 必须完整读取，但不能自行修改模型、源码、
checkpoint、运行方式、Precision Gate、Repair Scope 或停止规则。任何人类修改都要
增加 `contract_revision` 并追加 Human Decision。

<!-- CONTRACT-DATA: BEGIN -->
{
  "schema": "model-adaptation/v1",
  "spec_id": "step3p7-flash-p800",
  "contract_revision": 7,
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
    },
    "model": {
      "comparator": "torch.testing.assert_close",
      "atol": 0.01,
      "rtol": 0.02,
      "require_exact_structure": true,
      "require_declared_dtype": true,
      "require_finite": true,
      "integer_exact": true,
      "reference_runtime": "same-source-sglang"
    }
  },
  "repair_scope": {
    "allow_python": true,
    "allow_p800_pytorch": true,
    "allow_existing_xspeedgate_or_kunlun_ops": true,
    "allow_new_cpp_or_kernel_registration": false
  }
}
<!-- CONTRACT-DATA: END -->

### Goal

在固定 TP8、BF16、target-only eager 环境中：

1. 验证 P800 环境；
2. 对五个历史 Operator Candidate 全部执行独立 CPU reference 与 P800 生产算子测试；
3. 跑通固定文本和单图真实模型请求；
4. 对固定同版本 reference runtime 验证模型精度；
5. 只有精度失败时逐层、多卡定位并把新 case 返回算子队列。

### Precision rules

- CPU reference 必须来自固定 SGLang 源码、社区测试或独立数学定义，不能调用待验证
  的 P800 实现。
- CPU reference 可以 FP32 计算，比较前转为声明的输出 dtype。
- 浮点使用 Contract 固定的 `torch.testing.assert_close`、`atol` 和 `rtol`。
- 整数和布尔结果精确相等。
- 结构、shape、声明输出 dtype 和有限值必须满足。
- 原地算子检查每个被修改的 buffer；必要的 stride、alias 和输出 buffer 语义必须
  符合生产接口。
- Agent 不得放宽门槛；局部通过后必须回到真实模型路径。

### Repair Scope

允许修改原 P800 生产调用附近的 Python、P800 可执行 PyTorch，以及已有
xspeedgate/kunlun_ops 的组合或参数装配。不得为测试新增生产 helper。需要新增 C++、
自定义 kernel 或底层注册时进入 `BLOCKED`。

### Failure categories

- `ENVIRONMENT`
- `ADAPTATION`
- `OPERATOR_MISSING`
- `OPERATOR_CONTRACT`
- `DISTRIBUTED_RUNTIME`
- `ACCURACY`

### Closure

只有以下条件全部成立才能写 `PASS / DONE`：

1. P800 Preflight 通过；
2. 初始五个 Operator Verification Queue 条目全部在当前固定 revision 上通过；
3. 运行中发现的新 Confirmed Operator Gap 全部关闭；
4. 固定文本和单图请求都在真实 TP8 eager 模型上通过；
5. 两个请求的模型精度通过；
6. 没有未处理 Failure Observation。

SOURCE 静态检查、历史 Run 或已有建议补丁不能替代当前 P800 结果。

### Human Decisions

| revision | decision |
|---:|---|
| 1-6 | 历史 Contract 使用算子级 Golden 流程；只保留为 Git 历史。 |
| 7 | 删除通用采集与重放框架，改为 Agent 驱动的 CPU reference、P800 生产算子、真实 eager 模型和按需逐层精度定位；五个历史 Operator Candidate 全部重新进入测试队列。 |

<!-- HUMAN-OWNED CONTRACT: END -->

<!-- AGENT-WRITABLE WORKING STATE: BEGIN -->

## Working State

Agent 每次动作前重读 Contract 和本区。动作结束后先写新 Run Evidence，再原子更新
本区；详细命令、日志和源码证据不堆入 Working State。

### Current

- `observed_contract_revision`: `7`
- `state_revision`: `51`
- `status`: `ACTIVE`
- `phase`: `PREFLIGHT`
- `execution_site`: `SOURCE`
- `active_operator`: `null`
- `last_completed_action`: `run_driven_workflow_source_rewrite_and_review_alignment`
- `last_run`: `runs/run-driven-workflow-rewrite-001/result.md`
- `next_action`: `在 P800 固定 worktree 完成环境检查；通过后按 Operator Verification Queue 顺序测试全部五个历史 Operator Candidate`

phase 只使用：

```text
PREFLIGHT
OPERATOR_VERIFICATION
EAGER_BRINGUP
MODEL_ACCURACY
ACCURACY_DEBUG
DONE
```

### Environment

- `environment_status`: `NOT_STARTED`
- `environment_run`: `null`

### Operator Verification Queue

| operator_id | status | required cases | current evidence |
|---|---|---|---|
| `sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe._swiglu_silu_clamp_mul` | `PENDING` | `limit=None/finite；clamp 边界；生产配置参数可达` | `null` |
| `sgl_kernel.gemma_rmsnorm` | `PENDING` | `BF16；生产 hidden size；小 shape；连续/非连续 stride` | `null` |
| `sgl_kernel.gemma_fused_add_rmsnorm` | `PENDING` | `连续/非连续；mutated x；mutated residual；返回 None` | `null` |
| `sgl_kernel.topk_sigmoid` | `PENDING` | `correction bias；renormalize；生产 top-k；无并列；ids 精确` | `null` |
| `sglang.srt.layers.attention.triton_ops.prefill_attention._fwd_kernel` | `PENDING` | `ragged sequence；causal；生产 head dim；GQA；输出 buffer` | `null` |

状态只使用 `PENDING | ACTIVE | PASS | BLOCKED`。初始五项必须全部测试；历史结果不能
直接把任何一项改为 `PASS`。运行中确认的新生产算子缺口追加到表尾。

### Model

- `model_status`: `NOT_STARTED`
- `model_run`: `null`
- `accuracy_status`: `NOT_STARTED`
- `accuracy_run`: `null`
- `first_divergent_layer`: `null`

### Failure Observations

| phase | category | summary | evidence | next_action |
|---|---|---|---|---|

### Evidence rules

每次动作在新 `runs/<run-id>/result.md` 或 `result.json` 中记录：

- Contract revision、phase、失败类别和结论；
- 两个固定 commit、checkpoint、完整命令和实际环境；
- CPU reference 与 P800 生产调用的源码位置；
- 输入 shape、dtype、layout、stride、seed 和关键标量；
- 使用的 Precision Gate、逐输出结果和下一动作。

默认不保存完整输入输出 Tensor。只有 `ACCURACY_DEBUG` 无法通过已有观测入口定位时，
才保存必要的定点数据并记录原因和范围。

<!-- AGENT-WRITABLE WORKING STATE: END -->

# Migration Spec

<!-- HUMAN-OWNED CONTRACT: BEGIN -->

## Contract

人确认后，Migration Agent 只能读取本区。目标、源码、checkpoint、运行方式、精度
门槛或 Repair Scope 变化时，由人增加 `contract_revision` 并记录 Human Decision。

<!-- CONTRACT-DATA: BEGIN -->
{
  "schema": "model-adaptation/v1",
  "spec_id": "<model>-p800",
  "contract_revision": 1,
  "model": "<model>",
  "source": {
    "sglang_revision": "<commit>",
    "sglang_kunlun_revision": "<commit>"
  },
  "checkpoint": {
    "id": "<path-or-id>@<revision>",
    "config_digest": "<sha256>"
  },
  "runtime": {
    "tensor_parallel_size": 1,
    "dtype": "bfloat16",
    "target_only": true,
    "cuda_graph_backend_decode": "disabled",
    "cuda_graph_backend_prefill": "disabled"
  },
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
      "integer_exact": true
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

在固定 P800 拓扑上完成环境检查、全部已知算子验证、真实 eager 模型运行和模型精度
验证。精度失败时逐层定位第一个发散点。

### Precision rules

- CPU reference 必须独立于待验证的 P800 生产实现。
- 浮点结果使用 Contract 固定的 `atol` 和 `rtol`。
- 整数和布尔结果精确相等。
- 结构、shape、约定输出 dtype 和有限值必须满足。
- Agent 不得放宽门槛或用测试专用 helper 代替生产调用。
- 局部算子通过后必须返回真实模型路径验证。

### Failure categories

- `ENVIRONMENT`
- `ADAPTATION`
- `OPERATOR_MISSING`
- `OPERATOR_CONTRACT`
- `DISTRIBUTED_RUNTIME`
- `ACCURACY`

### Stop rules

- 需要新增 C++、自定义 kernel、底层注册或超出 Repair Scope 时进入 `BLOCKED`。
- 缺少人才能提供的 checkpoint、权限、节点类型或 baseline 时进入 `NEEDS_HUMAN`。
- 只有全部算子、真实模型和精度门槛通过时才进入 `PASS / DONE`。

### Human Decisions

| revision | decision |
|---:|---|
| 1 | 批准初始 Contract。 |

<!-- HUMAN-OWNED CONTRACT: END -->

<!-- AGENT-WRITABLE WORKING STATE: BEGIN -->

## Working State

### Current

- `observed_contract_revision`: `1`
- `state_revision`: `0`
- `status`: `ACTIVE`
- `phase`: `PREFLIGHT`
- `execution_site`: `SOURCE`
- `active_operator`: `null`
- `last_run`: `null`
- `next_action`: `在 P800 执行环境检查`

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

### Operator Verification Queue

| operator_id | status | required cases | evidence |
|---|---|---|---|
| `<production operator>` | `PENDING` | `<semantic risks>` | `null` |

状态只使用 `PENDING | ACTIVE | PASS | BLOCKED`。运行中确认的新缺口追加到表尾。

### Model

- `model_status`: `NOT_STARTED`
- `accuracy_status`: `NOT_STARTED`

### Failure Observations

| phase | category | summary | evidence | next_action |
|---|---|---|---|---|

### Evidence rules

每次动作在新 `runs/<run-id>/result.md` 或 `result.json` 中记录命令、固定源码位置、
CPU reference 来源、输入 shape/dtype/layout、P800 生产调用、精度门槛、结果和下一
动作。默认不保存输入输出 Tensor。

<!-- AGENT-WRITABLE WORKING STATE: END -->

# Migration Spec

<!-- HUMAN-OWNED CONTRACT: BEGIN -->

## Contract

人确认后，Migration Agent 只能读取本区。目标、源码、checkpoint、运行方式、算子精度
门槛、Repair Scope 或停止规则变化时，由人增加 `contract_revision`。

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
  "requests": [],
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

在固定 P800 拓扑上完成环境检查、全部已知与运行中新发现的算子验证，并让固定 eager
请求正常返回。整模型 CPU reference 和逐层精度调试不在本 Contract 中。

### Rules

- CPU reference 只用于局部算子，并独立于待验证的 P800 生产实现。
- 已有精确 PyTorch/`forward_native` 语义时标为 `NATIVE_IMPLEMENTATION`；实现分类和
  生产路由分开记录。
- 浮点结果使用固定 `atol`、`rtol`；整数和布尔结果精确相等；结构、shape、声明
  dtype、有限值和生产接口语义必须满足。Agent 不得放宽门槛。
- 局部通过后必须返回真实模型路径，不能用 test-only helper 代替生产调用。
- 一个 BUG 最多 3 次不同、有证据的范围内修复尝试；每次都归档。修复后继续原流程，
  3 次仍失败才 `BLOCKED`。
- 所有算子和固定 eager 请求通过后进入 `PASS / DONE`。

### Failure categories

- `ENVIRONMENT`
- `ADAPTATION`
- `OPERATOR_MISSING`
- `OPERATOR_CONTRACT`
- `DISTRIBUTED_RUNTIME`

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
DONE
```

### Auto Repair

- `auto_repair_status`: `NOT_EXERCISED`
- `active_bug`: `null`
- `active_attempt`: `0`
- `max_attempts_per_bug`: `3`
- `bug_log`: `docs/<model>-p800-bug-log.md`

### Environment

- `environment_status`: `NOT_STARTED`
- `environment_run`: `null`

### Operator Verification Queue

| candidate_id | historical_symbol | implementation_kind | route_state | status | required cases | evidence |
|---|---|---|---|---|---|---|
| `<semantic operator>` | `<historical symbol>` | `<NATIVE_IMPLEMENTATION/KUNLUN_IMPLEMENTATION/NO_IMPLEMENTATION_FOUND>` | `ROUTE_PENDING` | `PENDING` | `<production semantic risks>` | `null` |

缺少同名 symbol 不能单独写成 `NO_IMPLEMENTATION_FOUND`。`route_state` 只使用
`ROUTE_PENDING | PRODUCTION_REACHABLE | ROUTE_BLOCKED`；已有实现但 dispatch 不可达时
记录 `ADAPTATION`，实际生产调用到达且无实现时才记录 `OPERATOR_MISSING`。

### Model

- `model_status`: `NOT_STARTED`
- `model_run`: `null`

### Failure Observations

| bug_id | phase | category | summary | attempts | evidence | next_action |
|---|---|---|---|---:|---|---|

### Evidence rules

每次动作在新 `runs/<run-id>/result.md` 或 `result.json` 中记录命令、固定源码锚点、输入
特征、生产入口、精度门槛、结果和下一动作。遇到 BUG 时还要记录稳定 `bug_id`、最多
3 个 attempt、根因、解决方法、聚焦回归、原始 P800 路径复验和 Bug Ledger 路径。

<!-- AGENT-WRITABLE WORKING STATE: END -->

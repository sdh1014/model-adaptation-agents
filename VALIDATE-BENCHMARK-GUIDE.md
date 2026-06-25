# Adapt Validate 与 Benchmark

## 最小结构

```text
tasks/<model>/targets/<target>/runbook/
├── env.sh
├── start.sh
├── ready.sh
├── stop.sh
└── checks/
    ├── smoke.sh
    ├── validate.sh
    └── benchmark.sh
```

服务定义只维护一次。验证和压测只维护各自的测试脚本。

## 1. 配置验证

编辑：

```text
tasks/<model>/targets/<target>/runbook/checks/validate.sh
```

最小形式：

```bash
#!/usr/bin/env bash
set -euo pipefail
source "$CONTROL_ROOT/scripts/validation/lib.sh"
validation_init

validation_case reference-logits required -- \
  python /path/to/compare_logits.py \
    --endpoint "$MODEL_BASE_URL" \
    --output "$RUN_DIR/validation/logits.json"

validation_case batch optional -- \
  bash /path/to/test_batch.sh "$MODEL_BASE_URL"

validation_finish
```

已有整套验证脚本可以直接作为一个 case：

```bash
validation_case full-suite required -- bash /path/to/validate_all.sh
```

Logits 或 hidden states 已导出为 JSON/NPY/NPZ 时：

```bash
validation_case prefill-logits required -- \
  python "$CONTROL_ROOT/scripts/validation/compare_arrays.py" \
    --reference /reference/prefill.npy \
    --actual "$RUN_DIR/validation/prefill.npy" \
    --abs-tol 1e-3 --rel-tol 1e-3
```

执行：

```text
/adapt-validate <model>/<target>
```

## 2. 配置 Benchmark

编辑：

```text
tasks/<model>/targets/<target>/runbook/checks/benchmark.sh
```

```bash
#!/usr/bin/env bash
set -euo pipefail
source "$CONTROL_ROOT/scripts/benchmark/lib.sh"
benchmark_init

benchmark_case serving-c16 required --warmup 1 --repeat 3 -- \
  bash -lc '
    your_benchmark_command \
      --endpoint "$MODEL_BASE_URL" \
      --output "$BENCHMARK_SAMPLE_FILE"
  '

benchmark_expect serving-c16 output_throughput higher 900 median token_per_second
benchmark_expect serving-c16 median_ttft_ms lower 100 median ms
benchmark_finish
```

每次正式命令必须将 JSON 写入：

```text
$BENCHMARK_SAMPLE_FILE
```

JSON 可以直接是工具输出，也可以是：

```json
{
  "metrics": {
    "output_throughput": {
      "value": 912.4,
      "unit": "token_per_second",
      "direction": "higher"
    },
    "median_ttft_ms": {
      "value": 31.2,
      "unit": "ms",
      "direction": "lower"
    }
  }
}
```

执行：

```text
/adapt-benchmark <model>/<target>
```

## 3. 状态语义

验证：

```text
passed / failed / blocked / partial / decision_required
```

压测：

```text
status: passed | failed | blocked | partial
target_met: true | false | null
```

性能目标未达到时，`status` 仍可为 `passed`。

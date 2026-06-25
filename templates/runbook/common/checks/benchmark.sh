#!/usr/bin/env bash
set -euo pipefail

# 最简：直接粘贴压测命令，把 JSON/JSONL 写入 "$RUN_DIR/benchmark"。
# 推荐：使用辅助库处理 warmup、重复样本和聚合。
#
# source "$CONTROL_ROOT/scripts/benchmark/lib.sh"
# benchmark_init
# benchmark_case serving-c16 required --warmup 1 --repeat 3 -- \
#   bash /path/to/benchmark.sh
# benchmark_finish

echo "MODEL_RUN_NOT_CONFIGURED: 请编辑 $RUNBOOK_DIR/checks/benchmark.sh" >&2
exit 64

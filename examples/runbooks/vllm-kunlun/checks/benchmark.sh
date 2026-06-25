#!/usr/bin/env bash
set -euo pipefail

source "$CONTROL_ROOT/scripts/benchmark/lib.sh"
benchmark_init

benchmark_case serving-c16 required --warmup 1 --repeat 3 -- \
  bash -lc '
    result_dir="$(dirname "$BENCHMARK_SAMPLE_FILE")"
    result_name="$(basename "$BENCHMARK_SAMPLE_FILE")"
    vllm bench serve \
      --host "$MODEL_HOST" \
      --port "$MODEL_PORT" \
      --model "$MODEL_NAME" \
      --dataset-name random \
      --random-input-len 1024 \
      --random-output-len 256 \
      --num-prompts 128 \
      --max-concurrency 16 \
      --save-result \
      --result-dir "$result_dir" \
      --result-filename "$result_name"
  '

# 可选目标；删除这些行表示只测量，不判断 target_met。
# benchmark_expect serving-c16 output_throughput higher 900 median token_per_second
# benchmark_expect serving-c16 median_ttft_ms lower 100 median ms

benchmark_finish

#!/usr/bin/env bash
set -euo pipefail

source "$CONTROL_ROOT/scripts/benchmark/lib.sh"
benchmark_init

benchmark_case serving-c16 required --warmup 1 --repeat 3 -- \
  bash -lc '
    python -m sglang.bench_serving \
      --backend sglang \
      --host "$MODEL_HOST" \
      --port "$MODEL_PORT" \
      --model "$MODEL_NAME" \
      --dataset-name random \
      --random-input-len 1024 \
      --random-output-len 256 \
      --num-prompts 128 \
      --max-concurrency 16 \
      --output-file "$BENCHMARK_SAMPLE_FILE"
  '

# benchmark_expect serving-c16 output_throughput higher 900 median token_per_second
# benchmark_expect serving-c16 median_ttft_ms lower 100 median ms

benchmark_finish

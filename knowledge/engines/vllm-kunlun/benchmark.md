# vLLM-Kunlun Benchmark

优先复用当前环境实际可用的 vLLM benchmark CLI，并让工具把 JSON 写入 `$BENCHMARK_SAMPLE_FILE`。

典型在线指标包括 request/input/output throughput、TTFT、TPOT、ITL 和 E2E latency。

示例骨架：

```bash
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
  --result-dir "$(dirname "$BENCHMARK_SAMPLE_FILE")" \
  --result-filename "$(basename "$BENCHMARK_SAMPLE_FILE")"
```

具体参数以当前安装版本 `vllm bench serve --help` 为准，并记录版本。不要把文档示例当成固定接口。

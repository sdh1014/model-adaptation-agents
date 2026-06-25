# SGLang-Kunlun Benchmark

SGLang 的 serving benchmark 适合对已运行服务进行并发负载测试。固定 `num-prompts`、`max-concurrency`、随机输入/输出长度和数据集。

示例骨架：

```bash
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
```

参数和输出文件选项以当前安装版本 `python -m sglang.bench_serving --help` 为准。稳定吞吐测试应使用足够请求数，避免只测单批次启动效应。

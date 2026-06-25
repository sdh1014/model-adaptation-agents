# 设计参考

- Claude Code Skills：支持在 Skill 目录中放置按需读取的关联文件。
- vLLM Benchmark CLI：在线 serving benchmark 可保存 JSON 结果，并报告 TTFT、TPOT、吞吐等指标。
- SGLang Benchmark and Profiling：`bench_serving` 用于运行中服务的并发负载测试，并报告 TTFT、TPOT、ITL 和吞吐。

具体命令参数以容器内当前安装版本的 `--help` 为准。

# Scripts

```text
paths.py           统一 tasks/runs 路径和扁平 run key
model.py           模型静态检查与参考命令
assess.py          环境勘测和仓库扫描
implement.py       实现 run、快照、历史、签名、scope 与局部检查
model_runtime.py   Runbook 服务生命周期
evaluate.py        Validation / Benchmark 执行与指标解析
run_bash.py        通用命令证据采集
lib/*.sh           可选验证和压测辅助库
```

脚本只执行确定性操作并保存事实，不自行推断根因。

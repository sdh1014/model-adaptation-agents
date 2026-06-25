# Scripts

脚本只负责确定性操作和事实采集，不负责分析根因或更新任务结论。

当前可用：

- `scripts/model/inspect_model.py`
- `scripts/assess/inspect_target_repo.py`
- `scripts/hardware/p800/collect_environment.py`
- `scripts/hardware/p800/preflight.sh`
- `scripts/engines/vllm-kunlun/assess.sh`
- `scripts/engines/sglang-kunlun/assess.sh`
- `scripts/validation/*.py`
- `scripts/validation/lib.sh`
- `scripts/benchmark/*.py`
- `scripts/benchmark/lib.sh`
- `scripts/model_runtime.py`
- `scripts/migrate_runbooks.py`
- `scripts/run_bash.py`

其他 P800 与引擎脚本当前为显式失败的占位实现，防止未实现功能被误判为成功。

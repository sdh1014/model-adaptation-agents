# Skills

当前实现：

- `model-analyze`：建立与引擎无关的模型事实基线。
- `adapt-assess`：面向目标引擎生成适配评估、缺口和实施工作项。
- `adapt-implement`：每次实现一个已确认工作项，并保存证据、patch 和结果。
- `adapt-validate`：复用目标 Runbook 执行结构化正确性验证。
- `adapt-benchmark`：在正确性验证通过后复用同一 Runbook 执行性能测试。
- `model-run`：人工运行入口，复用目标目录 `runbook/` 和 `scripts/model_runtime.py` 启动、检查和停止模型服务。

未完成的 Skill 不创建可调用入口，避免用户误用占位流程。

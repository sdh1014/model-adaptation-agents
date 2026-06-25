# `adapt-validate` 与 Runbook 的集成契约

## 核心边界

`adapt-validate` 负责：

- 从模型分析中确定必测能力；
- 定义参考结果、比较方法和阈值；
- 生成验证计划与最终报告；
- 解释失败属于权重、算子、并行、生成协议还是未知问题。

Runbook 负责：

- 使用与 Smoke 完全相同的环境和启动命令；
- 启动并等待目标服务；
- 执行 `checks/validate.sh` 中的实际命令；
- 保存 stdout、stderr 和进程日志；
- 无论验证成功、失败或超时都停止服务。

## 标准调用

`adapt-validate` 创建自己的阶段运行目录后执行：

```bash
python scripts/model_runtime.py run \
  <model-id>/<target-id> \
  --check validate \
  --run-dir <validation-run-dir>/runtime
```

`checks/validate.sh` 可以直接写完整验证命令：

```bash
#!/usr/bin/env bash
set -euo pipefail

python "$CONTROL_ROOT/scripts/validation/run_cases.py" \
  --endpoint "$MODEL_BASE_URL" \
  --reference "$REFERENCE_ENDPOINT" \
  --output-dir "$RUN_DIR/validation"
```

也可以连续执行多条命令：

```bash
python scripts/validation/check_load.py ...
python scripts/validation/check_logits.py ...
python scripts/validation/check_generation.py ...
python scripts/validation/check_tp.py ...
```

## 结果判定

`model_runtime.py` 只根据脚本退出码判断执行状态：

```text
0   验证命令全部执行成功
64  验证入口未配置或缺少必要输入
其他 验证命令执行失败
```

这不等于最终模型正确性结论。`adapt-validate` 仍需读取：

```text
<validation-run-dir>/runtime/result.json
<validation-run-dir>/runtime/logs/
<validation-run-dir>/runtime/validation/
```

并根据阈值生成 `validation.md`。

## 为什么不复制启动命令

验证和 Smoke 若使用不同的环境变量、dtype、TP 或最大长度，会产生不可比较结果。因此：

```text
启动定义 只有一份：runbook/env.sh + runbook/start.sh
验证定义 只有一份：runbook/checks/validate.sh
生命周期 只有一份：scripts/model_runtime.py
```

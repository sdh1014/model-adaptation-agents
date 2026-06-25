# 可执行 Runbook

## 目的

推理服务启动通常包含大量环境变量、路径、设备配置和多行参数。把这些内容拆进 YAML 会产生转义、字段同步和维护成本。Runbook 直接保留 Shell 形式，使现有命令可以原样迁移。

## 单一事实来源

每个模型目标只保留一个 Runbook：

```text
tasks/<model>/targets/<target>/runbook/
```

以下阶段必须复用它：

- `model-run`：Smoke、人工调试、持久服务；
- `adapt-assess`：需要实际 baseline 时；
- `adapt-implement`：工作项完成后的局部运行检查；
- `adapt-validate`：启动目标服务并运行 `checks/validate.sh`；
- `adapt-benchmark`：后续可增加 `checks/benchmark.sh`。

## 责任划分

```text
env.sh          定义环境
start.sh        定义服务如何启动
ready.sh        定义何时可用
checks/*.sh     定义服务启动后执行什么
model_runtime.py 定义生命周期、日志和清理
```

生命周期代码不理解 vLLM、SGLang 或模型参数；Runbook 不负责进程管理。

## 修改原则

- 改启动方式，只改 `start.sh`；
- 改环境变量，只改 `env.sh`；
- 改验证命令，只改 `checks/validate.sh`；
- 不在 Skill 文档、`target.yaml` 和多个阶段中复制同一命令。

## `adapt-validate` 集成契约

`adapt-validate` 负责：

- 从模型分析中确定必测能力；
- 定义参考结果、比较方法和阈值；
- 生成验证计划与最终报告；
- 解释失败属于权重、算子、并行、生成协议还是未知问题。

Runbook 负责：

- 使用与 Smoke 相同的环境和启动命令；
- 启动并等待目标服务；
- 执行 `checks/validate.sh` 中的实际命令；
- 保存 stdout、stderr 和进程日志；
- 无论验证成功、失败或超时都停止服务。

标准调用：

```bash
python scripts/model_runtime.py run \
  <model-id>/<target-id> \
  --check validate \
  --run-dir <validation-run-dir>/runtime
```

`checks/validate.sh` 可以直接写完整验证命令，也可以调用 `scripts/validation/lib.sh` 记录结构化 case。示例：

```bash
#!/usr/bin/env bash
set -euo pipefail

source "$CONTROL_ROOT/scripts/validation/lib.sh"
validation_init

validation_case logits required -- \
  python /path/to/check_logits.py \
    --endpoint "$MODEL_BASE_URL" \
    --output "$RUN_DIR/validation/logits.json"

validation_finish
```

`model_runtime.py` 只根据脚本退出码判断执行状态：

```text
0   验证命令执行成功
64  验证入口未配置或缺少必要输入
65  验证部分完成
其他 验证命令执行失败
```

这不等于最终模型正确性结论。`adapt-validate` 仍需读取：

```text
<validation-run-dir>/runtime/result.json
<validation-run-dir>/runtime/logs/
<validation-run-dir>/runtime/validation/
```

并根据 required case 覆盖、参考结果和阈值生成 `validation.md`。

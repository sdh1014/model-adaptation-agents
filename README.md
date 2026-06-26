# Model Adaptation Agents

面向昆仑 P800 的模型适配工作区，底层使用 Claude Code。

## 只需要记住三件事

1. **通过六个 Skill 推进流程**；
2. **每个目标只维护一个 Runbook**；
3. **`tasks/` 看当前结论，`runs/` 查历史证据**。

```text
/model-analyze
/adapt-assess
/adapt-implement
/model-run
/adapt-validate
/adapt-benchmark
```

正式流程：

```text
model-analyze → adapt-assess → adapt-implement → adapt-validate → adapt-benchmark
                              ↑
                        model-run 用于运行与调试
```

## 第一次适配：按这六步做

### 1. 分析模型

```text
/model-analyze minimax-m3 --model-path /models/MiniMax-M3
```

### 2. 评估目标引擎

```text
/adapt-assess minimax-m3/vllm-kunlun \
  --target-repo /workspace/workspaces/minimax-m3/vllm-kunlun \
  --upstream-repo /workspace/sources/vllm
```

`adapt-assess` 会自动创建目标配置和 Runbook 模板。

### 3. 只编辑首次运行必需的三个文件

```text
tasks/minimax-m3/targets/vllm-kunlun/runbook/env.sh
tasks/minimax-m3/targets/vllm-kunlun/runbook/start.sh
tasks/minimax-m3/targets/vllm-kunlun/runbook/checks/smoke.sh
```

敏感环境变量写入本地文件：

```text
runbook/env.local.sh
```

该文件不会提交到 Git。

### 4. 实现并执行 Smoke

```text
/adapt-implement minimax-m3/vllm-kunlun
/model-run minimax-m3/vllm-kunlun
```

### 5. 配置并执行正式验证

编辑：

```text
runbook/checks/validate.sh
```

执行：

```text
/adapt-validate minimax-m3/vllm-kunlun
```

### 6. 配置并执行正式压测

编辑：

```text
runbook/checks/benchmark.sh
```

执行：

```text
/adapt-benchmark minimax-m3/vllm-kunlun
```

## 日常只会编辑这些文件

```text
tasks/<model>/model.yaml
tasks/<model>/targets/<target>/target.yaml
tasks/<model>/targets/<target>/runbook/env.sh
tasks/<model>/targets/<target>/runbook/env.local.sh
tasks/<model>/targets/<target>/runbook/start.sh
tasks/<model>/targets/<target>/runbook/checks/*.sh
```

阶段报告由 Skill 创建或更新，不需要用户手工维护其结构。

## Runbook：目标的唯一运行定义

```text
tasks/<model>/targets/<target>/runbook/
├── env.sh
├── env.local.sh          # 可选，本地秘密
├── start.sh
├── ready.sh
├── stop.sh
└── checks/
    ├── smoke.sh
    ├── validate.sh
    └── benchmark.sh
```

vLLM-Kunlun 与 SGLang-Kunlun 各有独立 Runbook，因此启动参数可以完全不同。

Runbook 不存在时可以手工初始化：

```bash
python scripts/model_runtime.py init <model>/<target>
```

正常经过 `/adapt-assess` 时无需单独执行该命令。

## Tasks 与 Runs

当前状态：

```text
tasks/<model>/
├── model.yaml
├── model-analysis.md
└── targets/<target>/
    ├── target.yaml
    ├── runbook/
    ├── assessment.md
    ├── implementation.md
    ├── validation.md
    └── benchmark.md
```

历史证据采用扁平时间线：

```text
runs/<model>/
└── <timestamp>-model-analyze/

runs/<model>--<target>/
├── <timestamp>-assess/
├── <timestamp>-implement-WI-003/
├── <timestamp>-model-run-smoke/
├── <timestamp>-validate/
└── <timestamp>-benchmark/
```

`tasks/` 回答“现在是什么状态”，`runs/` 回答“当时实际执行了什么”。

## 手工调试入口

以下命令只执行 Runbook 检查，不生成正式阶段结论：

```text
/model-run <model>/<target> --check validate
/model-run <model>/<target> --check benchmark
```

正式结论仍由 `/adapt-validate` 和 `/adapt-benchmark` 生成。

## 目录认知边界

普通使用者主要接触：

```text
.claude/skills/   命令入口
tasks/            配置、Runbook 和当前结论
runs/             历史日志与证据
```

维护者才需要查看：

```text
knowledge/        可复用知识
scripts/          确定性执行工具
templates/        报告和 Runbook 模板
tests/            自动测试
docs/             工作区和迁移说明
examples/         最小配置示例
```

## Docker

推荐把控制仓、目标仓和模型保存在宿主机，再 bind mount 到开发容器。见 [docs/WORKSPACE.md](docs/WORKSPACE.md)。

## 测试

```bash
make test
```

查看仓库帮助：

```bash
make help
```

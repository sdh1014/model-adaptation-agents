# Model Adaptation Agents

面向昆仑 P800 的模型适配工作区，底层使用 Claude Code。

## 五个目录

```text
.claude/skills/   六个公开入口
knowledge/        可复用模型、引擎和 P800 知识
scripts/          确定性执行与证据采集
tasks/            当前任务配置和最新阶段结论
runs/             不可变的历史执行证据
```

`tasks/` 与 `runs/` 保持分离：前者回答“现在是什么状态”，后者回答“当时实际执行了什么”。

## 六个命令

```text
/model-analyze
/adapt-assess
/adapt-implement
/model-run
/adapt-validate
/adapt-benchmark
```

```text
model-analyze → adapt-assess → adapt-implement → adapt-validate → adapt-benchmark
                              ↑
                        model-run 用于调试
```

## 最短流程

```text
/model-analyze minimax-m3 --model-path /models/MiniMax-M3

/adapt-assess minimax-m3/vllm-kunlun \
  --target-repo /workspace/workspaces/minimax-m3/vllm-kunlun \
  --upstream-repo /workspace/sources/vllm

/adapt-implement minimax-m3/vllm-kunlun
/model-run minimax-m3/vllm-kunlun
/adapt-validate minimax-m3/vllm-kunlun
/adapt-benchmark minimax-m3/vllm-kunlun
```

## 每个目标只编辑一个 Runbook

```text
tasks/minimax-m3/targets/vllm-kunlun/runbook/
├── env.sh
├── start.sh
├── ready.sh
├── stop.sh
└── checks/
    ├── smoke.sh
    ├── validate.sh
    └── benchmark.sh
```

vLLM-Kunlun 与 SGLang-Kunlun 各有独立 Runbook，因此启动参数可以完全不同。

## Tasks：当前状态

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

## Runs：扁平时间线

模型级：

```text
runs/minimax-m3/
└── 20260625-100000-model-analyze/
```

目标级：

```text
runs/minimax-m3--vllm-kunlun/
├── 20260625-110000-assess/
├── 20260625-120000-implement-WI-003/
├── 20260625-130000-model-run-smoke/
├── 20260625-140000-validate/
└── 20260625-150000-benchmark/
```

命名规则：

```text
模型级 run key = <model-id>
目标级 run key = <model-id>--<target-id>
run 目录        = <timestamp>-<stage>[-<detail>]
```

没有阶段子目录；`ls runs/<run-key>` 就能看到完整时间线。

手工执行验证或压测脚本：

```text
/model-run minimax-m3/vllm-kunlun --check validate
/model-run minimax-m3/vllm-kunlun --check benchmark
```

这只执行脚本，正式结论仍由对应阶段 Skill 生成。

## Docker

推荐把控制仓、目标仓和模型保存在宿主机，再 bind mount 到开发容器。见 [docs/WORKSPACE.md](docs/WORKSPACE.md)。

## 测试

```bash
bash tests/run.sh
```

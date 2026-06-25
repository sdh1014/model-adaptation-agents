# Model Adaptation Agents

面向昆仑 P800 的模型适配工作区，底层使用 Claude Code。

## 只需理解四个目录

```text
.claude/skills/   六个公开阶段入口
knowledge/        可复用模型、引擎和 P800 知识
scripts/          确定性检查、运行和证据采集
work/             tasks 保存结论，runs 保存原始证据
```

实际仓库仍使用：

```text
tasks/            当前任务结论
runs/             日志、命令、响应、指标和 patch
```

## 六个命令

```text
/model-analyze
/adapt-assess
/adapt-implement
/model-run
/adapt-validate
/adapt-benchmark
```

主流程：

```text
model-analyze → adapt-assess → adapt-implement → adapt-validate → adapt-benchmark
                              ↑
                        model-run 用于调试
```

## 最短上手流程

### 1. 分析模型

```text
/model-analyze minimax-m3 --model-path /models/MiniMax-M3
```

### 2. 评估一个目标

```text
/adapt-assess minimax-m3/vllm-kunlun \
  --target-repo /workspace/workspaces/minimax-m3/vllm-kunlun \
  --upstream-repo /workspace/sources/vllm
```

评估阶段会创建目标目录，并初始化对应引擎的 Runbook。

### 3. 只编辑一个目录

```text
tasks/minimax-m3/targets/vllm-kunlun/runbook/
├── env.sh                 环境变量
├── start.sh               完整启动命令
├── ready.sh               单次就绪探测
├── stop.sh                可选优雅停止
└── checks/
    ├── smoke.sh           最小调用
    ├── validate.sh        正确性命令
    └── benchmark.sh       压测命令
```

vLLM-Kunlun 和 SGLang-Kunlun 各有独立目标目录，因此启动命令可以完全不同。

### 4. 逐项实现

```text
/adapt-implement minimax-m3/vllm-kunlun
```

每次只处理一个工作项。

### 5. 运行和验收

```text
/model-run minimax-m3/vllm-kunlun
/adapt-validate minimax-m3/vllm-kunlun
/adapt-benchmark minimax-m3/vllm-kunlun
```

手工执行某个检查：

```text
/model-run minimax-m3/vllm-kunlun --check validate
/model-run minimax-m3/vllm-kunlun --check benchmark
```

这只执行脚本，不生成正式阶段结论。

## 任务结构

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

文件均按需生成，不使用庞大的空模板目录。

## 目录设计

```text
.claude/skills/       每个 Skill 只有 SKILL.md + GUIDE.md
knowledge/            8 个主题文件，不做过细拆分
templates/reports/    阶段报告模板
templates/runbook/    一套公共模板 + 三个引擎 start.sh
scripts/              6 个核心程序 + 2 个 Shell helper
docs/                 只保留工作区和迁移说明
```

## Docker

推荐把控制仓、目标仓和模型保存在宿主机，再 bind mount 到开发容器。详见 [docs/WORKSPACE.md](docs/WORKSPACE.md)。

## 测试

```bash
bash tests/run.sh
```

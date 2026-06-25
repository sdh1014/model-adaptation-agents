# Model Adaptation Agents

面向昆仑 P800 的模型适配工作区。第一版采用 **Skill-first** 设计：

- Skill 定义阶段流程；
- Knowledge 保存可复用知识；
- Scripts 执行确定性操作；
- Tasks 保存当前模型和目标引擎结果；
- Runs 保存日志、命令和原始证据。

## 当前可用 Skill

```text
/model-analyze
/adapt-assess
/adapt-implement
/adapt-validate
/adapt-benchmark
/model-run
```

`model-analyze` 只分析模型本身，不分析 vLLM-Kunlun、SGLang-Kunlun 或 P800，也不修改目标仓库。
`adapt-assess` 面向一个目标引擎生成适配评估、缺口和实施工作项，不修改目标仓库。
`adapt-implement` 每次只处理一个已确认工作项，并且只修改该工作项声明的一个目标仓库。
`adapt-validate` 复用目标 Runbook 执行结构化正确性验证，不修改目标仓库。
`adapt-benchmark` 在正确性验证通过后复用同一 Runbook 执行性能测试，不修改目标仓库。
`model-run` 是人工运行入口，使用目标目录中的 `runbook/` 和 `scripts/model_runtime.py` 执行 smoke、serve、check、status、stop。

阶段顺序：

```text
model-analyze
      ↓
adapt-assess
      ↓
adapt-implement
      ↓
adapt-validate
      ↓
adapt-benchmark

model-run：独立运行工具，不属于固定阶段。
```

## 快速使用

在 Claude Code 中执行：

```text
/model-analyze minimax-m3 --model-path /models/MiniMax-M3
```

带参考实现：

```text
/model-analyze minimax-m3 \
  --model-path /models/MiniMax-M3 \
  --reference-repo /src/MiniMax-M3
```

适配过程中发现模型分析缺失时：

```text
/model-analyze minimax-m3 \
  --update "确认 grouped routing 的归一化顺序"
```

输出：

```text
tasks/minimax-m3/
├── model.yaml
└── model-analysis.md

runs/minimax-m3/model-analyze/<timestamp>/
└── model-facts.json
```

生成目标引擎适配计划：

```text
/adapt-assess minimax-m3/vllm-kunlun \
  --target-repo ../vLLM-Kunlun \
  --upstream-repo ../vllm
```

执行一个适配工作项：

```text
/adapt-implement minimax-m3/vllm-kunlun --item WI-001
```

执行正确性验证：

```text
/adapt-validate minimax-m3/vllm-kunlun
```

执行性能测试：

```text
/adapt-benchmark minimax-m3/vllm-kunlun
```

初始化目标运行 Runbook：

```text
/model-run minimax-m3/vllm-kunlun --init
```

运行模型 smoke：

```text
/model-run minimax-m3/vllm-kunlun
```

## 目录职责

| 目录 | 职责 |
|---|---|
| `.claude/skills/` | 阶段入口、要求和工作流 |
| `knowledge/` | 已确认的通用、硬件、引擎和模型知识 |
| `scripts/` | 检查、启动、验证和采集脚本 |
| `tasks/` | 当前模型及各目标引擎的阶段结果 |
| `runs/` | 原始事实、日志、响应、指标和代码差异 |

参见 [MIGRATION.md](MIGRATION.md) 完成旧仓库迁移。

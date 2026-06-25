# 模型分析工作流

## 第 1 步：解析参数

从 `$ARGUMENTS` 中解析：

- `model-id`；
- `model-path`；
- `reference-repo`；
- `source`；
- `update`。

`model-id` 只能用于目录名，不得包含 `..` 或绝对路径。

## 第 2 步：准备任务目录

目标目录：

```text
tasks/<model-id>/
```

首次执行时创建 `model.yaml`，只保存稳定输入：

```yaml
model:
  id: <model-id>
  path: <model-path-or-null>
  source: <source-or-null>
  reference_repo: <reference-repo-or-null>
```

再次执行时，命令行参数覆盖本次运行输入；稳定变更同步回 `model.yaml`。

## 第 3 步：建立运行目录

创建：

```text
runs/<model-id>/model-analyze/<YYYYMMDD-HHMMSS>/
```

保存本次使用的模型路径、参考实现和分析模式。

## 第 4 步：采集静态模型事实

本地模型目录存在时执行：

```bash
python scripts/model/inspect_model.py \
  --model-path <model-path> \
  --output <run-dir>/model-facts.json
```

该脚本只读取静态文件，不导入模型 remote code。

## 第 5 步：读取分析知识

始终读取：

- `knowledge/common/model-analysis/model-identity.md`
- `knowledge/common/model-analysis/architecture-analysis.md`
- `knowledge/common/model-analysis/checkpoint-analysis.md`
- `knowledge/common/model-analysis/tokenizer-analysis.md`
- `knowledge/common/model-analysis/reference-behavior.md`

存在模型家族知识时再读取：

```text
knowledge/models/<model-family>/
```

不要读取具体引擎或硬件知识。

## 第 6 步：检查参考实现

存在 `reference-repo` 时，重点定位：

- config class；
- model class；
- attention；
- position encoding；
- FFN / MoE；
- weight loading；
- generation / tokenizer 相关逻辑。

记录文件和符号位置，不复制大段源码。

## 第 7 步：形成事实表

按照 `requirements.md`：

- 区分 `confirmed`、`inferred`、`unknown`；
- 对模型不涉及的能力使用 `not_applicable`；
- 将影响后续适配的未知项集中列出。

## 第 8 步：处理增量模式

存在 `--update` 时：

1. 读取已有 `model-analysis.md`；
2. 只调查指定问题；
3. 更新相关章节；
4. revision 加一；
5. 在“修订记录”中记录变更和影响范围。

## 第 9 步：生成报告

严格使用 `model-analysis-template.md`。

报告 frontmatter 至少包含：

```yaml
status: passed | blocked | failed
revision: 1
latest_run: runs/<model-id>/model-analyze/<timestamp>
```

## 第 10 步：结束

输出简短结论：

- 状态；
- 已确认的关键结构；
- 仍需确认的模型事实；
- 报告和证据路径。

不得自动进入引擎评估或代码修改阶段。

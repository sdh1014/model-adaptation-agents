---
name: model-inference-release-docs
description: 基于表格、CSV、Excel、日志摘录或文字说明，按固定模板生成模型推理准出文档。
argument-hint: "<input-file-or-notes> [--output OUTPUT.md]"
disable-model-invocation: true
---

# 模型推理准出文档生成

处理 `$ARGUMENTS` 指定的输入材料。

执行前读取 [GUIDE.md](GUIDE.md)、`references/rules.md` 和 `references/template.md`。

输入：

```text
*.csv
*.tsv
*.xlsx
*.md
*.txt
用户粘贴的表格或文字说明
```

输出：

```text
推理准出 Markdown 文档
```

边界：使用模型能力理解输入、补全模板中的部署步骤和说明；不读取 `tasks/`、`runs/`、Runbook 或目标仓；不执行验证、压测、Git 操作。

# 安装

## 自动安装

在文件包解压目录执行：

```bash
bash install.sh /path/to/model-adaptation-agents
```

不传路径时安装到当前目录：

```bash
cd /path/to/model-adaptation-agents
bash /path/to/model-run-runbook-stage/install.sh
```

安装器会先备份旧版：

```text
.claude/skills/model-run/
.claude/skills/model-runtime/
scripts/model_runtime.py
scripts/model_runtime/
```

备份位于：

```text
.model-run-backup/<timestamp>/
```

## 手动安装

```bash
cp -a .claude/skills/model-run /path/to/repo/.claude/skills/
cp -a scripts/model_runtime.py /path/to/repo/scripts/
cp -a knowledge/common/runtime/runbook.md /path/to/repo/knowledge/common/runtime/
cp -a knowledge/engines/vllm-kunlun/run.md /path/to/repo/knowledge/engines/vllm-kunlun/
cp -a knowledge/engines/sglang-kunlun/run.md /path/to/repo/knowledge/engines/sglang-kunlun/
chmod +x /path/to/repo/scripts/model_runtime.py
```

将 `CLAUDE.addendum.md` 的规则合并到根目录 `CLAUDE.md`。

## 初始化一个目标

```text
/model-run <model-id>/<target-id> --init
```

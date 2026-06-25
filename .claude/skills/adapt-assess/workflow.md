# 适配评估流程

## 1. 解析调用参数

第一个参数必须为：

```text
<model-id>/<target-id>
```

支持：

```text
--target-repo <path>
--upstream-repo <path>
--engine <vllm-kunlun|sglang-kunlun>
--hardware <p800>
--baseline <auto|required|skip>
--focus <capability>
```

默认值：

```text
engine = target-id
hardware = p800
baseline = auto
runtime.python = 当前 python
```

无法唯一确定 engine、target repo 或 model path 时，不猜测。

## 2. 校验模型分析

读取：

```text
tasks/<model-id>/model.yaml
tasks/<model-id>/model-analysis.md
```

检查：

- `model-analysis.md` 是否为 `status: passed`；
- revision；
- architecture 名称；
- 关键模型能力和未知项；
- 本地模型路径是否可用。

若缺少关键模型事实：

1. 不继续对应能力判断；
2. 在 assessment 中记录 `model_fact_gap`；
3. 输出精确命令：

```text
/model-analyze <model-id> --update "确认 <具体模型事实>"
```

## 3. 创建或更新目标定义

目标目录：

```text
tasks/<model-id>/targets/<target-id>/
```

创建或更新 `target.yaml`。示例见 `examples/target.yaml`。

记录：

- engine、hardware；
- target repo 和可选 upstream repo；
- runtime Python；
- TP 配置；
- baseline 模式；
- 用户明确提供的环境初始化命令或限制。

不得覆盖已有人工配置字段。

## 4. 创建本次证据目录

创建：

```text
runs/<model-id>/<target-id>/<YYYYMMDD-HHMMSS>-assess/
```

建议结构：

```text
environment.json
environment-summary.txt
target-repo.json
upstream-repo.json                 # 存在时
repo-status-before.txt
repo-status-after.txt
static-scan/
baseline/                          # 实际执行时
```

旧 run 不覆盖。

## 5. 执行轻量环境勘测

读取：

```text
knowledge/hardware/<hardware>/assessment-checklist.md
knowledge/hardware/<hardware>/environment.md
```

执行：

```bash
bash scripts/hardware/p800/preflight.sh \
  --model-path <model-path> \
  --target-repo <target-repo> \
  [--upstream-repo <upstream-repo>] \
  --engine <engine> \
  --python <python> \
  --required-devices <tp-size> \
  --run-dir <run-dir>
```

该脚本的非零退出不应中断报告生成。读取 `environment.json`，将环境标记为：

```text
ready | degraded | unavailable | unknown
```

禁止在本阶段安装、升级或修改系统配置。

## 6. 加载适配知识

始终读取：

```text
knowledge/common/adaptation/capability-matrix.md
knowledge/common/adaptation/gap-taxonomy.md
knowledge/common/adaptation/work-item-rules.md
knowledge/engines/<engine>/assess.md
```

根据模型分析按需读取：

```text
attention.md
position-encoding.md
weight-loading.md
moe.md
quantization.md
parallelism.md
```

只有 baseline 失败并出现匹配症状时，才读取：

```text
knowledge/hardware/p800/pitfalls.md
knowledge/engines/<engine>/pitfalls.md
```

pitfall 只能作为调查线索，仍需当前环境证据。

## 7. 执行目标仓静态扫描

对 vLLM-Kunlun：

```bash
bash scripts/engines/vllm-kunlun/assess.sh \
  --target-repo <target-repo> \
  [--upstream-repo <upstream-repo>] \
  --run-dir <run-dir> \
  --architecture <architecture-name> \
  [--model-type <model-type>]
```

对 SGLang-Kunlun 使用对应脚本。

扫描至少覆盖：

- Git commit、branch、dirty 状态；
- Python package 与 entry point；
- 模型注册和 architecture 映射；
- 相似模型实现；
- config、模型结构和 weight loader；
- attention、KV Cache、MoE、量化和并行；
- 平台插件、custom op、patch 和 fallback；
- 相关测试与文档。

插件式目标必须分别检查 `target_repo` 和 `upstream_repo` 或已安装的上游包位置。

## 8. 建立模型能力清单

从 `model-analysis.md` 提取 required capabilities，禁止仅按通用清单机械填充。

每个能力记录：

```text
模型要求
引擎执行路径
引擎状态
P800 状态
证据
未知项
```

模型不涉及的能力标记为 `not_applicable`。

## 9. 决定是否执行 Baseline

### auto

满足以下条件时执行：

- 环境为 `ready` 或 `degraded`；
- 模型路径为本地可读；
- 已有来自 target.yaml、目标仓文档或已确认知识的可信命令；
- 命令不会自动下载权重或修改源码。

缺少可信命令时记录 `not_run: command_unavailable`，不要编造参数。

### required

无法执行时最终结果为 `blocked`。

### skip

记录用户明确跳过，不影响有充分静态证据的 `adaptation_required`，但不能判定 `already_supported`。

## 10. 执行最小 Baseline

优先使用一次性离线生成。执行前：

```bash
git -C <target-repo> status --short > <run-dir>/repo-status-before.txt
```

优先通过 `python scripts/model_runtime.py run <model>/<target> --check smoke` 执行 target runbook；缺少 runbook 时通过 `scripts/run_bash.py` 保存命令、stdout、stderr 和退出码。

执行后：

```bash
git -C <target-repo> status --short > <run-dir>/repo-status-after.txt
```

Baseline 最少区分：

```text
not_run
import_failed
platform_init_failed
model_recognition_failed
config_failed
weight_load_failed
runtime_op_failed
generation_failed
passed
unknown_failed
```

若运行修改了目标源码，baseline 证据无效并标记 `blocked`。

## 11. 分析失败但不越界推断

处理顺序：

1. 提取第一处稳定错误签名；
2. 确定失败阶段；
3. 搜索目标 commit 中相关符号和路径；
4. 与模型能力矩阵对照；
5. 只有代码/测试与日志一致时，确认具体 gap；
6. 否则保持 `unknown_gap`。

不要把清理阶段的二次报错当作首要根因。

## 12. 生成适配结论

只允许：

```text
already_supported
adaptation_required
blocked
```

遵循 `requirements.md` 的判定条件。

报告必须区分：

- 代码能力缺口；
- 后端能力缺口；
- 包和版本问题；
- 环境问题；
- 配置问题；
- 模型事实缺口；
- 尚未确认的问题。

## 13. 生成实施计划

仅当结论为 `adaptation_required` 时生成或更新：

```text
tasks/<model-id>/targets/<target-id>/implementation.md
```

按 `implementation-template.md`：

- 每个 confirmed code gap 转为一个工作项；
- 每个工作项必须明确 `target_repo` 或 `upstream_repo` 为唯一可编辑仓库；
- 同一缺口确需跨仓修改时拆成两个有依赖关系的工作项；
- 环境或版本问题不生成代码工作项；
- 工作项必须有依赖和独立验收方法；
- `--focus` 模式只更新相关工作项；
- 旧工作项不删除，失效项标记 `superseded`。

结论为 `already_supported` 时，`implementation.md` 可不存在；已有文件则标记 `not_required`，不要删除历史。

## 14. 写 assessment.md

严格使用 `assessment-template.md`。

至少包含：

- 输入指纹；
- 环境摘要；
- baseline 状态；
- 能力矩阵；
- gap 分类；
- 适配结论和 confidence；
- 阻塞项；
- 模型分析反馈；
- 下一步命令；
- revision 历史。

## 15. 输出简短结果并停止

终端最终只输出：

```text
ASSESSMENT: already_supported | adaptation_required | blocked
REPORT: tasks/<model-id>/targets/<target-id>/assessment.md
EVIDENCE: runs/<model-id>/<target-id>/<timestamp>-assess
NEXT: <一条下一步命令>
```

不要自动执行下一阶段。

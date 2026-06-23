# Model Research Flow

## Goal

只做模型调研，不修改目标框架代码。

## Read Order

1. `tasks/<task>/task.yaml`
2. `tasks/<task>/status.md`
3. `tasks/<task>/notes.md`
4. `tasks/<task>/references.md`
5. 用户提供的本地权重目录、Hugging Face 链接、技术报告、PR、diff、本地文件

## Input Priority

- 如果提供了本地权重或模型目录，优先从本地文件分析。
- 如果没有提供本地权重或模型目录，则从 Hugging Face 模型页和文件列表分析。
- Hugging Face 调研只读取配置、README、model card、文件列表和小型元数据文件，不下载完整权重。
- 如果 Hugging Face 文件列表中存在配置文件，优先读取 `config.json`、`tokenizer_config.json`、`generation_config.json`。
- 如果只能看到权重 shard 名称，则只记录权重格式和 shard 组织方式，不下载 shard。

## Checklist

- 模型来源和 revision
- `config.json`
- `architectures`
- tokenizer 类型
- 权重格式
- Dense / MoE 结构
- Attention 类型
- RoPE / position encoding
- 量化格式
- 最近的已支持参考模型
- 目标框架中的注册入口
- Hugging Face 文件列表
- 权重 shard 格式

## Architecture Understanding

调研时不要只按固定 checklist 填空。必须阅读技术报告、PR、源码或配置，提炼该模型自己的关键结构。

要求：

- 识别论文 / 技术报告中反复强调的核心结构。
- 识别 PR 或源码中为该模型新增的关键模块。
- 解释这些结构在推理时影响哪些路径：模型注册、forward、attention、MoE、KV cache、权重加载、position encoding、processor。
- 不要求把所有结构套进固定分类。
- 不要求验证每个常见关注点是否存在。
- 如果出现新术语，例如 MHC、MLA、custom sparse attention，需要保留原术语，并用一两句话解释其作用。
- 对目标框架适配有影响的结构，必须在缺口分析中体现。

## Target Gap Analysis

调研时必须对 `tasks/<task>/task.yaml` 中的 `target_repo` 和 `framework` 做缺口分析。

缺口分析不绑定某个框架。目标框架可能是 vLLM、vLLM-Kunlun、SGLang、AIAK-SGLang 或其他框架。

## Upstream Baseline

当目标仓是 fork、plugin 或框架衍生仓时，缺口分析必须区分：

- upstream latest release：上游最新发布版本、最新 tag 或明确发布版本，例如 `0.21`。
- local checkout：当前本地代码中的支持状态。

不要默认把 `main` 分支当成最新版本。`main` 可能落后、冻结或不是发布基线。

默认顺序：

1. 先确认上游最新发布版本或最新 tag。
2. 再检查该版本中是否已有目标模型支持。
3. 再检查本地 checkout 与该版本的差异。

如果 upstream latest release 与 local checkout 不一致，必须记录：

- upstream 版本号 / tag / commit
- local branch / commit
- 差异是否影响本次模型适配判断

必须检查：

- 目标仓是否已有该模型注册。
- 是否已有相近模型实现可以参考。
- `config.json` / `architectures` 是否能被目标框架识别。
- 权重加载或权重映射是否需要新增逻辑。
- Attention 是否需要新 backend、特殊 mask、稀疏逻辑或算子。
- MoE 是否已有可用路径。
- RoPE / position encoding 是否兼容。
- tokenizer / processor 是否需要额外处理。
- 目标框架已有功能是否覆盖本模型需要的功能。

缺口分类只使用这些值：

- model_registry
- config
- weight_loading
- attention
- moe
- position_encoding
- tokenizer
- operator
- framework_feature
- unknown

## Behavior Rules

- 先整理用户给的资料，再做分析。
- 技术报告、PR、源码、本地实验记录分开记录。
- 不把 PR 描述当作已验证事实。
- 访问外网资料、下载文件或调用需要联网的命令时，必须使用下面的代理环境变量。
- 不修改目标框架代码。
- 不做性能、量化、P800 bringup。
- 缺少资料时，把缺失项写入 `result.json` 和 `tasks/<task>/reports/model-research.md`，不编造结论。
- 用户提供新链接、PR、文件或报告时，先更新 `references.md`。
- 缺口分析必须给出证据来源；没有证据时标记为 `unknown`。
- 模型调研报告中的关键结论必须带来源链接或本地文件路径。

## Network Proxy

```bash
HTTP_PROXY=http://agent.baidu.com:8891 \
HTTPS_PROXY=http://agent.baidu.com:8891 \
http_proxy=http://agent.baidu.com:8891 \
https_proxy=http://agent.baidu.com:8891 \
<command>
```

## Result (PASS / FAIL)

`runs/<task>/<ts>-model/result.json` 记录调研事实。`status` 为 `PASS` 或 `FAIL`。

`tasks/<task>/reports/model-research.md` 记录人工可读模型调研报告。

失败时只写事实，最小形状（禁写字段见 `CLAUDE.md`）：

```json
{
  "status": "FAIL",
  "failed_check": "model_load",
  "error_signature": "unsupported-attention-backend",
  "evidence": ["logs/stderr.log:183"]
}
```

PASS 时，**模型事实部分不规定字段名**，结构由本轮模型的实际关键结构决定（见「Architecture Understanding」），用自然的 key 覆盖以下内容即可：

- 模型来源：HF 仓 / revision / 是否下载权重。
- 关键结构事实：architectures、config、tokenizer、权重 shard 格式，**以及该模型自己的核心结构**（如 MSA / MLA / MTP / vision tower —— 有什么记什么，不必套固定分类）。
- 目标基线：目标仓 + 分支/tag + 对齐的框架版本 + 参考实现 + 参考实现与基线的版本差。

**只有缺口列表是跨任务可比的部分，固定为 `gaps` 数组，每项三个字段：**

```json
"gaps": [
  { "category": "attention", "severity": "blocker", "evidence": "flashmla_sparse.py:63 仅 sparse-MLA，无 sparse-GQA" }
]
```

- `category`：只能从「Target Gap Analysis」列出的枚举里选；都不匹配时用 `unknown` 并在 `evidence` 注明真实类型，不要自造新值。
- `severity`：只用 `blocker` / `required` / `partial` / `reusable` / `minor` / `deferred` / `unknown`。
- `evidence`：文件路径、行号或链接；没有证据时 `severity` 标 `unknown`。

## Report Evidence Rules

`tasks/<task>/reports/model-research.md` 是人工可读报告，但关键结论仍然必须可追溯。

- 每个模型结构、参考实现、目标基线、缺口判断，都必须附带至少一个链接或本地路径。
- 链接文字必须说明来源类型和对象，例如 `[HF config.json](...)`、`[technical report](...)`、`[PR #123](...)`、`path/to/file.py:123`。
- 不允许只写“见上游实现”“参考 PR”“技术报告提到”这类无链接说明。
- `references.md` 保存资料清单；`model-research.md` 仍然要在结论旁边写明对应来源。
- 私有页面或本地文件无法生成公开链接时，写精确路径、标题或 PR 编号，并说明访问位置。
- 缺少链接或文件证据的判断必须标记为 `unknown`，并写入缺失项。

## Update Targets

更新这些文件：

- `tasks/<task>/references.md`：资料清单、来源、阅读优先级。
- `tasks/<task>/notes.md`：按 `SKILL.md` 的 `Task Notes Rules` 更新。
- `tasks/<task>/reports/model-research.md`：人工可读模型调研报告。
- `tasks/<task>/status.md`：当前进度、本轮 `result.json` 路径和 `reports/model-research.md` 路径。
- `runs/<task>/<ts>-model/result.json`：机器可读调研事实和 `gaps`。

## Output Style

- 结论要短。
- 每条关键结论要能直接回到一个链接或本地路径。
- 不写长篇背景介绍。
- 不写未验证原因分析。

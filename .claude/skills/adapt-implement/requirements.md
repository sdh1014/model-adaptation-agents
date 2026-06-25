# 适配实现要求

## 1. 阶段目标

`adapt-implement` 将 `adapt-assess` 已确认的一个能力缺口转化为最小代码修改，并提供可复现的局部验收证据。

本阶段必须回答：

1. 当前工作项实际需要修改什么；
2. 为什么本轮修改能解释现有证据；
3. 修改是否严格处于工作项范围内；
4. 最小验收是否通过；
5. 若未通过，下一步属于继续实现、返回分析/评估、修复环境，还是需要开发者决策。

本阶段不负责完整模型正确性和性能验收。

---

## 2. 前置条件

### 2.1 必须存在

- `model-analysis.md`，且状态不是 `blocked`；
- `assessment.md`，且结论是 `adaptation_required`；
- `implementation.md`，且至少有一个可处理工作项；
- 当前工作项明确指定一个可编辑仓库：`target_repo` 或 `upstream_repo`；
- 被指定仓库已在 `target.yaml` 中声明、路径可访问且是 Git 仓库；
- 当前工作项包含直接证据、候选范围、依赖、允许路径和最小验收。

### 2.2 必须一致

- `assessment.md` 记录的模型分析 revision 与当前 `model-analysis.md` 一致；
- `implementation.md` 记录的 assessment revision 与当前 `assessment.md` 一致；
- `target.yaml` 的 engine、hardware、target_repo 与 assessment 一致；
- 当前目标仓 HEAD、工作区 patch 和 Python/import 来源没有出现未记录漂移。

若上述事实不一致，先分类为 `analysis_stale`、`assessment_stale` 或 `repository_drift`，不得直接修改代码。

### 2.3 环境检查边界

本阶段不重复完整环境勘测，只做执行上下文复核：

- 当前 Python 路径；
- 目标包 import 来源；
- 目标仓 HEAD、branch、dirty 状态；
- 当前局部验收需要 P800 时的设备可见性；
- 评估中关键版本是否变化。

环境有实质变化时返回 `/adapt-assess`，而不是在实现阶段安装或修复环境。

---

## 3. 单一工作项

一次调用只处理一个工作项。

选择规则：

1. `--item <ID>` 指定时只处理该项；
2. 否则选择第一个依赖已满足的 `pending` 或 `needs_recheck` 项；
3. `blocked` 项只有在阻塞条件已解除时才能恢复；
4. `decision_required` 项必须先获得 `--decision "..."` 或已记录的开发者决定；
5. 不自动处理下一个工作项。

工作项必须保持单一 capability，例如：

- 模型注册；
- config 字段映射；
- 一组权重映射；
- 一个 Attention/KV Cache 语义；
- 一个 MoE routing 语义；
- 一个 TP/EP 切分规则；
- 一个 P800 算子路径。

“完整支持某模型”不能作为单个工作项。一个工作项也不能同时修改两个仓库；确实需要跨仓修改时，必须拆成有依赖关系的两个工作项，或进入 `decision_required`。

---

## 4. 知识库读取

### 4.1 始终读取

- `knowledge/common/adaptation/work-item-rules.md`
- `knowledge/common/implementation/evidence-driven-debugging.md`
- `knowledge/common/implementation/failure-classification.md`
- `knowledge/common/implementation/repository-safety.md`
- `knowledge/engines/<engine>/implement.md`

### 4.2 按工作项读取

| Capability | 读取文件 |
|---|---|
| 模型发现、config、模型类 | `model-integration.md` |
| 权重加载 | `weight-loading.md` |
| Attention | `attention.md` |
| 位置编码 | `position-encoding.md` |
| KV Cache | `kv-cache.md` |
| MoE | `moe.md` |
| 量化 | `quantization.md` |
| TP/EP/PP | `parallelism.md` 和硬件 `distributed.md` |
| 多模态 | `multimodal.md` |
| 服务协议 | `serving-behavior.md` |

### 4.3 失败后读取

只有先从日志提取失败签名并完成初步分类后，才读取：

- `knowledge/engines/<engine>/pitfalls.md`
- `knowledge/hardware/<hardware>/pitfalls.md`
- `knowledge/models/<model-family>/` 中对应问题

历史问题只能作为候选解释，必须由当前证据验证。

---

## 5. 历史证据要求

修改代码前必须读取：

- 当前工作项此前所有 `adapt-implement` run 的摘要；
- 最近失败命令的 stdout、stderr、退出码和失败签名；
- 已尝试的修改与假设；
- 当前工作区相对上次 run 的变化。

必须明确写出：

- 已排除的假设；
- 不得重复的命令或改法；
- 当前新增证据；
- 本轮假设与之前尝试的差异。

没有新增证据时，不得重复相同尝试。

---

## 6. 假设驱动修改

每个实现尝试必须包含：

- `hypothesis`：对当前失败或缺口的可验证解释；
- `evidence`：支持该假设的直接证据；
- `change_scope`：本轮允许修改的文件或目录；
- `expected_observation`：若假设正确，验收应发生什么变化；
- `falsification`：什么结果会否定该假设。

禁止：

- 无证据的大范围重构；
- 同时验证多个独立假设；
- 为绕过错误而静默 fallback；
- 删除或跳过测试以获得通过；
- 将环境错误改写成模型代码修复。


### 6.1 失败 patch 处置

进入下一个假设前，必须明确上一 attempt 的代码处置：

- `retain`：修改已被独立证据确认，只是暴露了后续问题；
- `amend`：假设部分成立，下一轮只修正已定位部分；
- `remove`：假设被否定，必须用定向编辑撤销本 attempt 引入的修改。

不得让已否定修改在未说明的情况下累积。禁止使用 `git reset --hard` 或 `git clean` 回退。若无法确认哪些变化属于本 attempt，分类为 `repository_drift` 并停止。

---

## 7. 仓库修改边界

允许：

- 修改当前工作项声明的单一可编辑仓库；
- 在当前 run 目录写证据；
- 更新当前目标的 `implementation.md` 和必要的 `blockers/<WI-ID>.md`。

禁止：

- 修改模型权重或 tokenizer 文件；
- 修改当前工作项未声明的其他仓库；
- 安装、升级或卸载系统/Python 依赖；
- 执行 `git reset --hard`、`git clean -fd`、强制 checkout；
- 自动 commit、push、merge、rebase；
- 覆盖已有 run；
- 在未获批准时扩大到上游核心仓或 assessment 明确禁止的目录。

首次实现时目标仓若已有未知 dirty 修改，状态必须为 `blocked` 或 `decision_required`。已有实现 run 的累计修改允许继续，但当前 patch 必须与上次记录一致；不一致时视为 `repository_drift`。

---

## 8. 验收要求

每个工作项必须执行 assessment 中给出的最小验收。仅 `py_compile` 不能替代工作项验收。

验证层次按需使用：

1. 语法或静态检查；
2. import、registry、config 或权重名称检查；
3. 单层/单算子/小配置测试；
4. 最小加载或 smoke；
5. 与本工作项直接相关的回归测试。

完整 logits、生成、多卡和长上下文验收属于 `/adapt-validate`。

通过必须同时满足：

- 指定最小验收退出码为 0；
- 输出中不存在被忽略的失败；
- 修改范围检查通过；
- 没有未声明 fallback；
- 当前工作项的直接目标已满足；
- 证据已保存到 run 目录。

---

## 9. 尝试次数与停止条件

默认一次 Skill 调用最多进行 3 个不同的修复假设，可用 `--max-attempts N` 调整，取值限 1～5；不得用提高次数绕过停止条件。

以下任一条件出现时停止修改：

- 同一失败签名在没有新增证据的情况下重复 2 次；
- 当前调用已验证 3 个不同假设仍未通过；
- 修复需要超出当前工作项或允许仓库范围；
- 缺少模型事实、评估事实或有效测试 oracle；
- 运行环境不再与 assessment 一致；
- 需要在正确性、兼容性、维护成本之间做取舍；
- 已无法提出可证伪的新假设。

明确的外部缺失条件标记为 `blocked`；需要开发者选择方向或反复失败时标记为 `decision_required`。

---

## 10. 工作项状态

- `pending`：尚未开始；
- `in_progress`：正在实现或仍有明确的新假设；
- `blocked`：存在明确外部阻塞，且解除条件唯一；
- `decision_required`：需要开发者选择方案或接受限制；
- `needs_recheck`：上游输入变化，需要重新验收；
- `passed`：最小验收和范围检查通过；
- `not_applicable`：新证据证明该工作项不再适用。

失败的单次尝试不等于整个工作项 `failed`。

---

## 11. 阶段输出

每次调用至少保存：

- 运行元数据；
- 历史摘要；
- 修改前后 Git 快照；
- 每个假设与对应命令；
- stdout、stderr、退出码；
- 失败签名；
- 累计 patch；
- 修改范围检查；
- 本次结果；
- `implementation.md` 更新。

达到人工决策条件时还必须生成 `blockers/<WI-ID>.md`。

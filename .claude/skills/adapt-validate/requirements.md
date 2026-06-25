# `adapt-validate` 要求

## 1. 阶段目标

本阶段证明当前适配实现满足模型语义和目标使用场景，而不仅是能加载或返回文本。

必须回答：

1. 使用的模型、代码、环境和 Runbook 是否与评估/实现阶段一致；
2. 哪些正确性 case 是 required、optional 或 not applicable；
3. 每个 case 使用什么 oracle、输入、阈值和证据；
4. 失败属于模型事实、评估、实现、运行环境、验证 oracle 还是测试脚本；
5. 下一步应返回哪个阶段，或者需要开发者做什么决定。

---

## 2. 前置条件

必须存在：

- `tasks/<model-id>/model-analysis.md`；
- `tasks/<model-id>/targets/<target-id>/target.yaml`；
- `tasks/<model-id>/targets/<target-id>/assessment.md`；
- `tasks/<model-id>/targets/<target-id>/implementation.md`，或 assessment 明确为 `already_supported`；
- 目标 Runbook 的 `env.sh`、`start.sh`、`ready.sh`、`checks/validate.sh`；
- 至少一种可说明正确性的 oracle。

可接受的 oracle：

- 官方或可信参考实现；
- 已确认的参考服务；
- 固定 checkpoint 的参考 logits/token；
- 模型规范中可确定的性质；
- 已人工确认并记录的 golden case。

只有“生成文本看起来正常”时，状态最多为 `partial`。

---

## 3. 输入一致性

验证前记录并检查：

- `model-analysis.md` revision；
- assessment revision 和目标 commit；
- implementation 最新通过工作项及 patch；
- 当前目标仓 HEAD、dirty patch；
- Python/import 来源和关键版本；
- Runbook 文件哈希；
- 模型 revision、dtype、TP、最大长度和量化设置。

出现未解释漂移时不得继续判定正确性：

- 模型事实更新：返回 `/model-analyze` 后重新 assess；
- 目标仓或依赖环境变化：返回 `/adapt-assess`；
- 实现 patch 与最后记录不一致：需要开发者确认或重新 implement。

---

## 4. 必测层次

### 4.1 Load

至少检查：

- 模型类和配置加载成功；
- 权重缺失、多余、重命名和跳过项有明确记录；
- dtype、量化、权重 tying 符合预期；
- 不存在静默随机初始化或未声明 fallback。

### 4.2 Forward

按模型能力检查：

- prefill 成功；
- 单步 decode 成功；
- logits/token shape 正确；
- KV Cache 更新路径正确；
- batch 维度、序列维度和并行切分正确。

### 4.3 Parity

优先级：

```text
prefill logits
→ 首个 decode token / logits
→ 短 greedy generation
→ 多步 generation
```

比较必须固定：

- checkpoint/revision；
- tokenizer 和 chat template；
- 输入 token；
- sampling 参数；
- dtype 与容许误差；
- 随机种子；
- EOS/stop 行为。

### 4.4 Scenarios

至少评估：

- single request；
- batch 或并发请求；
- short context；
- long context；
- EOS/stop/max_tokens；
- tensor parallel；
- 模型特有能力。

不适用项必须显式记录 `not_applicable`，不能删除。

### 4.5 Regression

按修改范围执行最小回归：

- import/registry；
- 目标引擎基础服务路径；
- 与当前修改共享代码的已有模型或算子测试。

---

## 5. Case 契约

推荐在 `checks/validate.sh` 中使用：

```bash
source "$CONTROL_ROOT/scripts/validation/lib.sh"
validation_init
validation_case <name> <required|optional> -- <command...>
validation_mark <name> <required|optional> <blocked|skipped|not_applicable> "原因"
validation_finish
```

每个 case 保存：

- 名称和级别；
- 状态；
- 执行时间；
- stdout/stderr；
- 退出码；
- 阻塞或跳过原因。

状态语义：

- `passed`：case 的 oracle 明确满足；
- `failed`：执行完成但不满足 oracle；
- `blocked`：缺少参考、数据、设备或必要输入；
- `skipped`：本轮有理由不执行，但仍属于适用能力；
- `not_applicable`：模型或目标不涉及该能力。

---

## 6. 阶段状态

### `passed`

同时满足：

- 至少一个 required case；
- 所有 required case 均为 `passed` 或 `not_applicable`；
- 无 required case 为 `blocked`、`skipped` 或 `failed`；
- required coverage 与模型分析、评估、实现修改相符；
- 没有未声明 fallback；
- 环境与版本证据完整；
- 服务清理成功。

### `failed`

- 任一 required case 运行完成并明确不满足 oracle；
- 服务、算子或模型路径发生可复现错误；
- cleanup 失败导致结果环境不可置信。

### `blocked`

- 缺少 required oracle、输入、设备或参考结果；
- Runbook 未配置；
- 环境或 revision 漂移；
- 当前工作项不足以覆盖新发现 capability。

### `partial`

- required case 通过，但 optional case 失败或阻塞；
- 使用非结构化验证脚本，无法证明 required coverage 完整；
- 只完成了部分验证层次。

`partial` 不能作为正式 benchmark 的默认前置通过状态。

---

## 7. 历史与重复失败

执行前至少查看最近三次 validate run：

- 相同 case；
- 相同失败签名；
- 相同模型/代码/Runbook 哈希；
- 上次建议动作是否已执行。

相同失败签名在没有新增证据的情况下重复两次时：

- 不继续机械重跑；
- 在 `validation.md` 中标记 `decision_required`；
- 列出稳定失败、已排除原因、缺少的证据和一个推荐动作；
- 最多提供三个简短决策选项。

---

## 8. 修改边界

本阶段允许：

- 读取任务、知识、目标仓和历史日志；
- 执行 Runbook；
- 在当前 run 写证据；
- 更新 `validation.md`。

本阶段禁止：

- 修改目标仓代码；
- 修改模型、tokenizer 或 checkpoint；
- 安装/升级依赖；
- 自动改写 Runbook；
- 调整参数直到结果“看起来通过”；
- 进入性能优化。

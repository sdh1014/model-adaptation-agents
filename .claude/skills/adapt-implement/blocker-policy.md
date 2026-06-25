# 阻塞点与开发者决策规则

## 1. 目标

本规则防止 `adapt-implement` 在证据不足时无限修改，也防止所有错误都被粗略标记为“代码问题”。

核心区分：

- **blocked**：存在唯一、明确的外部解除条件；
- **decision_required**：存在取舍、范围扩大或重复失败，需要开发者选择；
- **implementation_defect**：仍有明确、可证伪的新假设，可以继续当前工作项。

---

## 2. 分类表

| 分类 | 典型证据 | 当前动作 | 工作项状态 |
|---|---|---|---|
| `model_fact_gap` | 无法确认模型真实计算、权重或输出语义 | 停止修改，给出精确 `/model-analyze --update` | blocked |
| `assessment_gap` | 出现未评估 capability、工作项粒度错误或引擎能力判断失效 | 停止修改，重新 `/adapt-assess` | blocked |
| `environment_blocker` | import 来源、关键版本、设备或运行时与 assessment 不一致 | 停止修改，由环境负责人修复并重评估 | blocked |
| `repository_drift` | 当前 patch/HEAD 与最后记录不一致，来源不明 | 等待开发者确认保留、迁移或恢复 | decision_required |
| `implementation_defect` | 代码路径明确，当前假设被测试否定，但仍存在新局部假设 | 记录失败并进入下一不同假设 | in_progress |
| `scope_expansion` | 修复要求新增独立 capability 或修改 assessment 禁止位置 | 返回 assessment 拆项或请求批准 | blocked / decision_required |
| `test_oracle_gap` | 没有可靠标准判断实现是否正确 | 补充验证设计或由开发者决定 | blocked / decision_required |
| `dependency_or_upstream_boundary` | 必须升级依赖、修改上游核心或改变公共接口 | 生成开发者决策单 | decision_required |
| `resource_limit` | OOM、磁盘、设备数量不足，且不是代码语义问题 | 环境处理或缩小受控测试 | blocked |
| `transient_execution` | 明确的网络、调度、设备占用或临时进程异常 | 允许一次相同命令重试并记录瞬态证据 | in_progress |

分类必须引用日志、代码或环境证据。

---

## 3. 失败签名

失败签名用于回答“是否仍是同一个可观察失败”，通常包含：

- 退出码或 timeout；
- 失败测试名；
- 异常类型；
- 规范化后的关键错误信息；
- 最后一个稳定 stderr 片段；
- signature hash。

失败签名不等于根因。以下写法错误：

```text
RuntimeError: shape mismatch，因此根因是 KV Cache。
```

正确写法：

```text
观察到 shape mismatch；KV Cache layout 是候选假设，需要独立测试验证。
```

---

## 4. 重复失败停止规则

出现以下任一情况，停止自动尝试：

1. 同一签名重复 2 次，且第二次没有新增证据；
2. 已验证 3 个不同、合理的实现假设仍未通过；
3. 新假设只是换一种写法重复先前修改；
4. 无法设计能区分候选根因的测试；
5. 下一步需要修改另一个 capability；
6. 下一步需要升级依赖或修改上游核心；
7. 当前工作区来源不再可信。

不得通过增加 max-attempts 绕过明显的决策点。

---

## 5. 决策单内容

`blockers/<WI-ID>.md` 必须让开发者在不阅读完整会话的情况下理解：

- 当前工作项目标；
- 已经确认能工作的部分；
- 当前稳定失败；
- 已尝试且被否定的方案；
- 仍未知的事实；
- 为什么自动实现停止；
- 推荐选择；
- 最多 3 个选择及代价；
- 做出决定后的恢复命令。

不能只写“多次失败，请人工介入”。

---

## 6. 选项设计

选项必须基于当前证据，常见形式：

1. **推荐：保持当前范围，补充一个特定证据或测试**；
2. **批准扩大范围**，例如允许修改上游核心或升级指定依赖；
3. **接受限制或延期**，明确哪些配置暂不支持。

每个选项写明：

- 允许做什么；
- 不允许做什么；
- 对兼容性和维护的影响；
- 预计需要重新执行哪些阶段。

---

## 7. 恢复规则

开发者通过以下方式恢复：

```text
/adapt-implement <model-id>/<target-id> \
  --item <WI-ID> \
  --decision "<明确决定和允许边界>"
```

Skill 必须：

1. 将决定原文写入 `blockers/<WI-ID>.md`；
2. 记录时间和对应工作项；
3. 检查决定是否足以解除阻塞；
4. 只在批准边界内继续；
5. 决定不足或互相矛盾时仍保持 `decision_required`。

已解决的 decision 不删除，标记为 `resolved` 并保留历史。

# 验证失败路由

| 分类 | 识别证据 | 返回位置 | 典型动作 |
|---|---|---|---|
| `model_fact_gap` | 参考实现语义、权重组织、tokenizer 或模型行为未确认 | `model-analyze` | 增量补充指定模型事实 |
| `assessment_gap` | 出现未评估 capability、工作项缺失或引擎能力判断失效 | `adapt-assess` | 重评估受影响 capability |
| `implementation_defect` | oracle 明确，目标实现稳定偏离，修改范围已知 | `adapt-implement` | 处理现有工作项或由 assess 新增项 |
| `runbook_configuration` | 环境、启动或验证脚本未配置/参数错误 | 开发者 Runbook | 编辑唯一对应脚本 |
| `environment_drift` | Python/import/version/device 与 assessment 不一致 | `adapt-assess` | 恢复环境或重新评估 |
| `reference_oracle_gap` | 没有可信参考结果、阈值或输入不一致 | 验证配置 | 补充 oracle 后重跑 |
| `test_defect` | 验证脚本本身异常、读取错文件、比较逻辑错误 | 验证脚本 | 修复 `checks/validate.sh` 或验证工具 |
| `resource_limit` | OOM、设备不足、磁盘不足且非语义错误 | 环境处理 | 恢复资源后重跑 |
| `nondeterminism` | 相同输入与指纹出现不可解释波动 | 开发者决策 | 固定随机性或接受统计标准 |
| `unknown` | 证据不足以区分候选原因 | decision_required | 补充最能区分原因的单一证据 |

## 路由原则

1. 先确认输入、tokenizer、sampling 和 revision 完全一致；
2. 再确认失败发生在服务、模型加载、forward、比较脚本还是输出协议；
3. 只有 oracle 可信且实现偏离时，才路由到 `adapt-implement`；
4. 新能力或工作项粒度变化必须回到 `adapt-assess`；
5. 同一失败无新增证据重复两次时停止重跑。

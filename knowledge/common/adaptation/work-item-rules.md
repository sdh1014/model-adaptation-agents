# 实施工作项规则

## 目标

让 `/adapt-implement` 每次只处理一个可独立验证的能力，不把整个模型适配压缩成一个任务。

## 必备字段

每个工作项必须包含：

- 唯一 ID；
- 单一 capability；
- 直接证据；
- 修改目标；
- 唯一可编辑仓库：`target_repo` 或 `upstream_repo`；
- 允许修改范围；
- non-goals；
- 前置依赖；
- 明确、可执行的最小验收命令；
- 完成后的后续影响。

## 合理粒度

合理：

- 注册一个 architecture 到已有兼容模型类；
- 增加 config 字段映射；
- 增加一组 QKV 权重映射；
- 支持 grouped routing 的一个模型语义；
- 为指定 KV layout 增加 backend 路径；
- 增加 TP=8 的参数切分和局部测试。

不合理：

- 支持 MiniMax-M3；
- 完成所有 MoE 适配；
- 修复所有启动问题；
- 提升性能；
- 同时修改注册、权重、MoE 和分布式。

## 顺序

通常按以下依赖排序：

1. 模型发现与 config；
2. 模型结构；
3. 权重加载；
4. Attention、位置编码和 KV Cache；
5. MoE、量化、多模态等特有能力；
6. TP/EP/PP；
7. 服务协议；
8. 性能专用路径。

环境问题在代码工作项之前解决，但不作为目标代码工作项。

## 验收

验收必须尽量局部：

- import/registry test；
- config parsing test；
- weight name mapping test；
- 单层或单算子 forward test；
- 小配置模型加载；
- 与工作项直接相关的最小 smoke request。

不得用“完整服务能启动”代替所有局部能力测试，也不得只用 `py_compile` 宣布工作项通过。

## 修改范围

候选文件是边界，不是强制实现方案。`adapt-implement` 可以基于新证据调整具体文件，但必须保持同一 capability。

需要新增独立 capability、同时修改第二个仓库或扩大到禁止区域时，应返回 `/adapt-assess` 拆项，或生成开发者决策单。

## 状态

- `pending`
- `in_progress`
- `blocked`
- `decision_required`
- `needs_recheck`
- `passed`
- `not_applicable`

失败的单次 attempt 只记录为 attempt 结果，不作为工作项终态。

模型分析、assessment、目标 commit 或执行环境发生影响语义的变化时，相关已完成工作项标记为 `needs_recheck`。

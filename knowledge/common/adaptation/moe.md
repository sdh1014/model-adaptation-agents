# MoE 能力评估

## 模型事实

- routed expert 数量；
- top-k；
- shared expert 数量和执行位置；
- router logits、score function 和 bias；
- grouped routing、group 数和选组规则；
- top-k 前后归一化；
- expert activation 和中间维度；
- auxiliary loss 是否仅训练期存在；
- expert 权重命名；
- TP/EP 切分。

## 引擎与后端检查

- 模型类是否精确实现 routing 顺序；
- fused MoE 支持的 top-k、group、dtype 和 shape；
- shared expert 是否进入 fused 路径；
- token dispatch、all-to-all 和输出聚合；
- P800 kernel 是否支持模型的权重布局；
- generic fallback 是否可运行且语义一致；
- 量化 expert 和 router 的支持范围。

## 支持状态示例

- 标准 top-k MoE 支持，但 grouped routing 不支持：`partially_supported`；
- 模型类支持 routing，P800 fused kernel 不支持且无 fallback：`unsupported`；
- 只看到同系列 MoE 文件，未确认 routing：`unknown`。

## 最小验证

- 固定 router logits 的 expert 选择；
- shared expert 路径；
- 单层 MoE 与参考实现；
- TP/EP 下 expert 分配；
- 量化模型的 scale 和输出。

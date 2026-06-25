# 量化能力评估

## 模型事实

- 方法和配置来源；
- weight/activation dtype；
- bit width、group size、zero-point；
- per-channel/per-group 规则；
- scale 命名和 shape；
- 哪些模块保持高精度；
- Attention、MoE、LM head 和 multimodal 模块的覆盖范围。

## 引擎与 P800 检查

- quant config 是否被识别；
- loader 是否理解实际 checkpoint 布局；
- linear/MoE/attention kernel 的支持范围；
- TP/EP 对 scale 和 packed weight 的切分；
- fallback 是否会错误反量化或改变 dtype；
- graph/compile 模式是否支持。

## 判定边界

支持同名量化方法不等于支持该 checkpoint。必须比较：

- 权重布局；
- scale 组织；
- group size；
- kernel dtype；
- 模型特有模块。

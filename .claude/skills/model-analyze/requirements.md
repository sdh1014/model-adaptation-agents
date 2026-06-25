# 模型分析要求

## 1. 目标

本阶段必须建立模型事实基线，回答：

1. 模型的实际计算结构是什么；
2. checkpoint 和权重如何组织；
3. Tokenizer、Chat Template 和生成配置是什么；
4. 哪些模型行为需要在后续适配中验证；
5. 当前仍有哪些未知项。

本阶段不判断具体推理引擎是否支持。

## 2. 必要输入

至少具备以下一种来源：

- 可读取的本地模型目录；
- 可读取的本地参考实现；
- 明确的官方模型来源。

缺少可验证来源时，报告状态必须为 `blocked`，禁止根据模型名称猜测。

## 3. 必查内容

### 3.1 模型身份

- 模型名称和版本；
- `architectures`；
- `model_type`；
- 默认 dtype；
- Transformers / remote code 要求；
- 权重格式与 shard 数量。

### 3.2 主干结构

- hidden size、层数、词表大小；
- normalization 类型与位置；
- embedding 与 lm_head 是否共享；
- 激活函数；
- residual 结构。

### 3.3 Attention

- attention heads；
- KV heads；
- head dimension；
- MHA、GQA、MQA 或其他形式；
- QKV bias、output bias；
- sliding window 或稀疏注意力；
- causal mask 的特殊行为。

### 3.4 位置编码

- RoPE 或其他位置编码类型；
- theta；
- scaling；
- partial rotary factor；
- 最大上下文长度。

### 3.5 FFN / MoE

- dense FFN 或 MoE；
- intermediate size；
- expert 数量、top-k、shared expert；
- grouped routing、路由归一化和 auxiliary loss；
- 特殊激活与门控方式。

### 3.6 Checkpoint

- 权重索引文件；
- shard 组织；
- 参数命名规则；
- packed / merged 权重；
- QKV、gate-up、expert 权重布局；
- 量化配置与额外元数据。

### 3.7 Tokenizer 与生成

- tokenizer 类型；
- BOS、EOS、PAD 等特殊 token；
- chat template；
- generation config；
- stop token 或多 EOS 行为。

### 3.8 模型特有能力

按实际情况检查：

- multimodal；
- MTP / speculative decoding；
- custom cache；
- 自定义算子；
- 特殊输出头；
- 其他非标准结构。

不涉及的内容标记为 `not_applicable`。

## 4. 证据要求

影响后续适配的每个关键事实必须记录：

- 事实值；
- 来源文件；
- 文件中的字段、符号或代码位置；
- 可信状态。

可信状态仅使用：

- `confirmed`：有直接配置或代码证据；
- `inferred`：由多个事实推导，但没有直接声明；
- `unknown`：目前无法确认；
- `not_applicable`：不适用。

`inferred` 和 `unknown` 不得伪装为 `confirmed`。

## 5. 增量修订

使用 `--update` 时：

- 只调查指定问题及直接依赖；
- 不无理由重写整份报告；
- 增加 revision；
- 记录触发来源、变更事实、证据和受影响能力；
- 原有结论被推翻时明确标记旧结论已失效。

## 6. 完成条件

状态为 `passed` 必须满足：

- 模型身份已确认；
- 关键计算结构有证据；
- checkpoint 组织已说明；
- tokenizer 与生成行为已说明；
- 所有影响后续适配的未知项已明确列出；
- 报告没有具体引擎支持结论。

来源不足时使用 `blocked`；执行或解析失败时使用 `failed`。

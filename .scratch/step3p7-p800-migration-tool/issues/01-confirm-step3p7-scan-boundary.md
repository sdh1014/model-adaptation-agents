# 确认 Step-3.7 Flash 的真实扫描入口与语义算子边界

Type: research
Status: resolved
Blocked by:

## Question

基于当前 Step-3.7/Step3p5、SGLang 与 SGLang-Kunlun 源码，Migration Agent 应从哪个真实模型入口开始，沿哪些 `forward` 调用下钻，并在什么位置停止，才能完整枚举实际路径上的 Semantic Operator，同时不扩散到 EAGLE Worker 外围逻辑？研究还需给出 `READY`、`CAPTURE_REQUIRED`、`NEEDS_HUMAN` 三种判定所需的最小源码证据字段，并用特殊 SwiGLU 片段验证“可独立重放和替换”的边界。

## Comments

## Answer

基于 `sglang/baidu/aicapx/sglang` 的 `kunlun-0.5.14@546ad8c682392922792bbbfe53a8bf575545f118`，扫描必须从加载后的模型配置解析两个模型内入口：目标路径从 `Step3p7ForConditionalGeneration.forward` 进入，EAGLE draft 路径因配置改写从 `Step3p5MTP.forward` 进入。两者继续下钻到 Step3p5 DecoderLayer 的 Attention、MLP/MoE；Worker、ModelRunner、采样树和 CUDA Graph 只作为入口证据，不进入算子枚举。

每条扫描记录最少包含：稳定算子标识、target/draft 调用链、分支激活条件、输入输出与状态边界、CUDA 实现锚点、Kunlun 实现锚点、replay/replace 检查，以及带理由的 `READY`、`CAPTURE_REQUIRED` 或 `NEEDS_HUMAN`。模型配置改写和启动期符号替换使纯 AST 无法可靠定案，因此由 Migration Agent 判断语义和缺口，薄脚本只抽取、校验源码证据。

特殊 SwiGLU 的无权重 replay 边界可定为 `gate_up: [..., 2D] + limit -> gate * up: [..., D]`，但该表达式内联在 `Step3p5MLP.forward`，没有独立替换入口；实际 checkpoint 是否让 MTP layer 45 走该分支也未有配置证据。故当前判为 `NEEDS_HUMAN`，不能仅因没有专用 Kunlun Kernel 就宣称为真实 Operator Gap。后续由 [决定特殊 SwiGLU 的替换入口与 Demo 资格](08-decide-swiglu-seam-and-demo-eligibility.md)单独决策。

完整代码证据与行号见 [研究记录](../../../work/research/step3p7-scan-boundary.md)。

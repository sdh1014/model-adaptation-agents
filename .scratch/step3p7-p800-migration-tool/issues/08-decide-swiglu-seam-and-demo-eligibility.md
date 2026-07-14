# 决定特殊 SwiGLU 的替换入口与 Demo 资格

Type: grilling
Status: resolved
Blocked by: 01

## Question

当前带 limit 的 SwiGLU 片段已有清晰、无权重的 replay 输入输出，但它内联在 `Step3p5MLP.forward` 中，缺少可独立替换的命名入口；同时静态源码不能证明该分支一定被实际 checkpoint 激活，也不能证明通用 Torch 表达式在 P800 上构成真实 correctness gap。Demo 应选择哪条最小路线：引入一个人工批准的薄 helper 作为采集与 plugin replacement seam，扩大 Semantic Operator 边界，还是改选扫描得到的另一个真实缺口？决策必须继续满足“无权重、最多三种实测 shape、一个真实缺口”，并禁止把“缺少专用优化”当作 correctness gap。

## Comments

用户同意推荐路线：保留特殊 SwiGLU 候选，把现有内联表达式原样抽成一个很薄的普通函数；抽函数只建立采集和替换入口，不视为真实缺口证据。实际配置未命中或 P800 baseline 未失败时，改选扫描得到的其他候选。

## Answer

Demo 首选特殊 SwiGLU，但不扩大到整个 `Step3p5MLP.forward`。在 CUDA Capture 前，由人工批准一次纯 Python 重构：把 `gate_up + limit -> gate * up` 的现有 `chunk + silu + clamp + mul` 表达式原样抽成模块级普通函数，并让 CUDA 与 P800 使用同一份已固定 revision 的重构代码。该函数同时作为打桩入口和 Kunlun plugin 的替换目标；它不包含 `gate_up_proj`、`down_proj` 或权重，也不改变计算语义。

代码上这条路线可行：特殊表达式当前内联于 `python/sglang/srt/models/step3p5.py:95-107`；若扩大到整个方法会同时包含两次投影。现有 HookRegistry 能以 fully-qualified path 替换普通函数，并把新绑定传播到已经 import 原函数的模块，见 `python/sglang/srt/plugins/hook_registry.py:83-104,182-205,268-300`。

抽出函数本身既不是 Operator Gap，也不能证明 Demo 已成立。特殊 SwiGLU 只有同时满足以下条件才能成为 Demo 的关闭对象：

1. 固定 checkpoint 的加载后配置证明，target 或 draft 实际扫描路径上的对应层 `swiglu_limits_shared[layer_id]` 非零；分支条件见 `python/sglang/srt/models/step3p5.py:498-505`。
2. 唯一 CUDA Capture Session 已为该函数形成最多三个 Golden Samples。
3. 同一组样本的 P800 baseline replay 真实失败；“没有专用 Kunlun Kernel”或“可能更慢”都不算失败证据。

若配置未激活该分支，特殊 SwiGLU 不属于本次实际路径；若 P800 baseline 通过，它就不是 correctness gap。两种情况都直接放弃该候选，改选完整扫描中下一个已采集且 baseline 失败的无权重 Semantic Operator，不扩边界、不伪造失败，也不为它重新访问 CUDA。

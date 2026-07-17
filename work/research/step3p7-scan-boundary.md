# Step-3.7 Flash 真实扫描入口与语义算子边界

> 历史说明（2026-07-17）：本文研究的是早期 EAGLE/draft 假设，已被 Contract revision 3 的 target-only eager 范围取代。源码分析本身保留为历史证据；当前扫描结论以 `runs/scan-003/result.json` 为准。

## 结论

Migration Agent 不能只从一个固定 Python 类开始扫描。它应先读取本次运行绑定的模型配置，再分别建立两棵模型内调用树：

1. 目标模型树：`Step3p7ForConditionalGeneration.forward`；
2. EAGLE 草稿模型树：配置改写后的 `Step3p5MTP.forward`。

`MultiLayerEagleDraftWorker`、`ModelRunner` 和 `EagerRunner` 只负责组织批次并调用模型，不进入算子枚举。扫描进入模型后，沿实际激活的 `forward` 分支下钻；到达一个具有明确输入、输出和可替换调用点的语义计算时记录为 Semantic Operator，再分别定位它的 CUDA 与 Kunlun 实现，不继续拆成每一个 `chunk`、`sigmoid`、`clamp` 等基础张量操作。

当前特殊 SwiGLU 片段的无权重重放边界是清楚的，但现有源码没有独立的命名调用点，而且仓库中没有本次模型的实际 `text_config` 和 P800 重放结果。因此它现在只能作为边界设计候选，不能仅凭静态源码宣称为“真实 Operator Gap”或直接进入 CUDA 采集。

## 源码快照

本文以嵌套仓库 `/Users/songdehao/sdh-lab/code/sglang/baidu/aicapx/sglang` 为准：

- branch：`kunlun-0.5.14`
- commit：`546ad8c682392922792bbbfe53a8bf575545f118`
- 工作树：干净，且与 `origin/kunlun-0.5.14` 对齐

父目录 `/Users/songdehao/sdh-lab/code/sglang` 把 `baidu/` 显示为未跟踪目录，但 `baidu/aicapx/sglang` 本身是独立、可追溯的 Git 仓库，不能把它描述成无 commit 的源码快照。

## 两个真实模型入口

### 1. 目标模型路径

模型注册器从模块的 `EntryClass` 收集类，并按 `architectures` 解析实际模型类：

- `python/sglang/srt/models/registry.py:94-127`
- `python/sglang/srt/models/step3p7.py:200`

Step-3.7 的模型内调用链是：

```text
Step3p7ForConditionalGeneration.forward
  -> general_mm_embed_routine
     -> Step3p5ForCausalLM.forward
        -> Step3p5Model.forward
           -> Step3p5DecoderLayer.forward (逐层)
              -> Step3p5Attention.forward
              -> Step3p5MLP.forward / Step3p5MoEMLP.forward
```

代码依据：

- Step-3.7 将 `Step3p5ForCausalLM` 作为 `language_model`：`python/sglang/srt/models/step3p7.py:47-74`。
- `Step3p7ForConditionalGeneration.forward` 调用多模态嵌入流程：`python/sglang/srt/models/step3p7.py:136-152`。
- 多模态流程最终调用 `language_model`：`python/sglang/srt/managers/mm_utils.py:1023-1047,1139-1145`。
- CausalLM 再进入主体模型：`python/sglang/srt/models/step3p5.py:855-881`。
- 主体模型逐层调用 DecoderLayer：`python/sglang/srt/models/step3p5.py:719-773`。
- DecoderLayer 依次进入 Attention 和 MLP/MoE：`python/sglang/srt/models/step3p5.py:594-658`。

图片编码支路只在本次输入实际包含多模态数据时激活；`general_mm_embed_routine` 的条件位于 `python/sglang/srt/managers/mm_utils.py:1050-1055`。因此“完整路径”是相对于 Contract 中固定的模型配置和 Demo 输入而言，不是把所有理论分支都算成已执行路径。

### 2. EAGLE 草稿模型路径

当 Step-3.7 作为 draft model 加载时，配置会把外层多模态架构改写为文本侧的 `Step3p5MTP`：

- `python/sglang/srt/configs/model_config.py:564-571`

草稿模型内调用链是：

```text
Step3p5MTP.forward
  -> Step3p5AMultiTokenPredictor.forward
     -> Step3p5DecoderLayer.forward (单个 MTP block)
        -> Step3p5Attention.forward
        -> Step3p5MLP.forward / Step3p5MoEMLP.forward
```

代码依据：

- `Step3p5MTP.forward`：`python/sglang/srt/models/step3p5_mtp.py:141-179`。
- Predictor 构造并调用一个 `Step3p5DecoderLayer`：`python/sglang/srt/models/step3p5_mtp.py:61-85,88-127`。
- 当前实现把 MTP 的 `layer_id` 固定为 45：`python/sglang/srt/models/step3p5_mtp.py:74-84`。

`MultiLayerEagleDraftWorker` 创建 draft runner，并根据 `Step3p5MTP` 打开 hidden-state 链式传递：`python/sglang/srt/speculative/multi_layer_eagle_worker_v2.py:143-169`。它在非图模式下调用 `draft_runner_list[step].forward`：同文件 `:578-609`。再往下，EagerRunner 才调用 `model.forward`：`python/sglang/srt/model_executor/runner/eager_runner.py:188-198,238-244,345-350`。这些代码证明 Worker/Runner 是进入模型的调度证据，但不是 Semantic Operator 扫描根。

## 下钻与停止规则

扫描器按下面的顺序工作：

1. 固定源码 commit、checkpoint revision、加载后的 `hf_config/text_config` 和 Demo 输入模式。
2. 从配置实际解析出的目标模型类和 draft 模型类开始，而不是从文件名或 Worker 类猜入口。
3. 沿当前配置和输入能够激活的模型 `forward` 调用下钻；动态分支必须记录激活条件。
4. 遇到下列边界时形成一条 Semantic Operator 记录：
   - 有明确的输入、输出和非 Tensor 参数；
   - 能从一次调用中单独采集并离线重放；
   - 在 SGLang 调用点有一个可定位、可替换的函数、方法或 Kernel 符号。
5. 对该记录只追到 CUDA 实现和 Kunlun 实现的落点，不继续把内部基础 Torch 表达式全部拆开。
6. 遇到 Worker、Runner、采样、树构建、批次调度和 CUDA Graph 管理时停止横向扩散；这些内容只作为“为什么会进入模型”的上下文证据。

这套规则意味着不能依赖纯 AST 自动遍历作决定。模型架构会被配置改写，插件又会在启动时替换函数绑定：

- 架构改写：`python/sglang/srt/configs/model_config.py:564-571`；
- Hook 支持替换函数、方法和类：`python/sglang/srt/plugins/hook_registry.py:60-66,83-105,182-192,268-292`；
- Kunlun Kernel 注册后会替换模块符号及已经导入的别名：`sglang-kunlun/sglang_kunlun/kernels/kernel_ops.py:119-164`。

因此最小实现应由 Migration Agent 判断调用关系和边界；很薄的确定性脚本只负责抽取源码位置、校验符号是否存在、生成证据清单，不能独立给出 Operator Gap 结论。

## 每个扫描结果的最小证据字段

源码快照和模型配置快照可以放在 scan run 顶层；每个 Semantic Operator 至少保存以下字段：

| 字段 | 作用 |
|---|---|
| `operator_id` | 稳定标识同一语义边界，不用底层 Kernel 名代替 |
| `model_path` | `target` 或 `draft`，以及从真实入口到该算子的调用链源码锚点 |
| `activation_guard` | 让该分支生效的配置值、层号、forward mode 或输入条件及其证据 |
| `boundary` | 输入、输出、非 Tensor 参数、是否依赖权重/模块状态，以及对应调用点 |
| `cuda_impl` | CUDA 侧实际实现锚点和实现类型：通用 Torch、SGLang JIT/Kernel 或 backend |
| `kunlun_impl` | Kunlun 侧实现锚点和实现类型：共享通用实现、plugin hook、xspeedgate/kunlun_ops 或缺失 |
| `replay_replace_check` | 是否能单独采集、重放、替换；失败时给出缺少的 seam |
| `verdict` | `READY`、`CAPTURE_REQUIRED` 或 `NEEDS_HUMAN`，附一句可由上述证据推出的理由 |

判定规则：

- `READY`：已有明确的 P800 可执行落点，且现有证据足以确认它覆盖所需语义。
- `CAPTURE_REQUIRED`：CUDA 路径明确，但 Kunlun 实现缺失、依赖 CUDA-only 路径，或必须用 CUDA 输出固定语义后才能在 P800 上修复。
- `NEEDS_HUMAN`：实际分支、调用关系或“最小且可替换”的边界仍不能唯一确定。它不能被“先采数据再说”掩盖。

“没有 Kunlun 专用优化”不自动等于 Operator Gap；如果通用 Torch 路径在 P800 上能正确执行，它仍可判为 `READY`。反过来，仅看到同名函数也不能判为 `READY`，还要证明语义和参数覆盖一致。

## 特殊 SwiGLU 边界验证

带 limit 的 Step3p5 MLP 代码位于 `python/sglang/srt/models/step3p5.py:95-107`：

```python
gate_up, _ = self.gate_up_proj(x)
gate, up = gate_up.chunk(2, dim=-1)
gate = F.silu(gate)
gate = gate.clamp(min=None, max=self.limit)
up = up.clamp(min=-self.limit, max=self.limit)
output, _ = self.down_proj(gate * up)
```

适合无权重 replay 的语义边界是：

- 输入：`gate_up_proj` 的 Tensor 输出 `gate_up`，形状 `[..., 2D]`；
- 非 Tensor 参数：`limit`；
- 输出：送入 `down_proj` 之前的 `gate * up`，形状 `[..., D]`；
- 不保存：`gate_up_proj/down_proj` 权重、`gate`、`up` 等内部 Tensor。

这组输入输出可以通过 `gate_up_proj` 的输出和 `down_proj` 的输入定点采集，所以“可独立重放”成立。但当前表达式直接内联在 `Step3p5MLP.forward` 中，没有一个包住 `chunk + silu + clamp + mul` 的函数或模块调用点，所以“可独立替换”不成立。若直接替换整个 `Step3p5MLP.forward`，边界会重新包含两个投影和权重，违背当前 Demo 的无权重约束。

此外，该分支只有在 `swiglu_limits_shared[layer_id]` 存在且非零时才激活：`python/sglang/srt/models/step3p5.py:498-505,552-568`。`Step3p5Config` 的显式构造参数并没有声明该字段，而是可能从 checkpoint 配置的额外字段进入：`python/sglang/srt/configs/step3p5.py:6-10,76-108`。没有实际 checkpoint 配置快照，就不能证明 Step-3.7 target 或 MTP layer 45 一定走这个分支。

普通、不带 limit 的 `SiluAndMul` 已有明确 CUDA 与 Kunlun 路径：

- SGLang CUDA 实现：`python/sglang/srt/layers/activation.py:90-107`；
- Kunlun 把相同 JIT 符号替换为 `kunlun_ops.swiglu`：`sglang-kunlun/sglang_kunlun/kernels/kernel_ops.py:329-358`。

这进一步说明“普通 SwiGLU 已有 Kunlun 实现”不能证明“带 clamp 的特殊 SwiGLU”也有等价专用实现；但特殊分支当前使用通用 Torch 表达式，也不能仅凭“没有专用 Kunlun Kernel”反向证明它在 P800 上不能正确执行。

### 当前判定

特殊 SwiGLU 候选当前为 `NEEDS_HUMAN`，理由有两个：

1. 缺少实际配置证据，不能证明该分支属于本次真实路径；
2. replay 边界明确，但缺少满足定义的独立 replacement seam。

后续必须先决定是引入一个很薄的命名 helper 作为采集/替换点，还是改选扫描发现的另一个真实缺口。即便引入 helper，也仍需由实际 P800 replay 或明确的 CUDA-only/Kunlun 缺失证据确认它具备 Demo 所要求的“真实缺口”资格。

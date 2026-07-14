# 研究记录：一次性 CUDA Golden Run 与交接包契约

## 结论

最小 Demo 仍然需要保存输入 Tensor 和 CUDA 期望输出 Tensor，否则 P800 无法离线重放，也无法区分“实现错误”和“输入不同”。但不需要把所有相关对象都保存下来：权重不保存；`epsilon`、`limit` 这类真正参与算子语义的小型非 Tensor 参数按原值写入样本元数据；内部中间 Tensor 默认不采集。

Golden Run 的职责是回答三件事：采到了哪一次真实调用、这些字节能否在一个新 CUDA 进程中重放、交接到 P800 后文件是否仍然完整。它不负责决定 Tensor 底层序列化库，也不绑定某个比较 API。

以下结论基于 SGLang `kunlun-0.5.14@546ad8c682392922792bbbfe53a8bf575545f118`。文中行号均对应这个固定提交。

## Golden Run 最小内容

一个 Golden Run 只需要以下逻辑结构；`<tensor-format>` 表示格式尚未决定，不是建议文件扩展名。

```text
runs/<golden-run-id>/
├── result.json
├── capture.log
├── samples/
│   └── <sample-id>/
│       ├── metadata.json
│       ├── inputs.<tensor-format>
│       └── expected-outputs.<tensor-format>
└── self-replay/
    ├── result.json
    └── replay.log
```

`result.json` 保存 Run 级信息：

- `golden_run_id` 与唯一的 `capture_session_id`；
- 完整 capture 命令、开始和结束时间、成功或失败原因；
- SGLang commit、SGLang-Kunlun commit、checkpoint 标识与 revision、checkpoint config 的 SHA-256；
- 打桩工具版本或源码摘要；
- Python、Torch、CUDA runtime/driver 与 GPU 型号；
- 本次观察到的算子、保留样本数、重复样本数、超过上限而未保存的签名数；
- CUDA self-replay 的结果和证据路径。

不保存完整 `pip freeze` 或全部环境变量。上面的版本信息足以说明 Golden 是在哪套关键代码和运行时上产生的，同时避免把凭证混入交接包。

## Golden Sample 最小内容

每个 Golden Sample 记录一个 Semantic Operator 的一次边界调用。

| 内容 | 最少保存什么 | 原因 |
|---|---|---|
| 算子身份 | 扫描阶段给出的稳定 `operator_id` | 把样本与 Operator Gap 对齐 |
| 源码锚点 | qualified symbol 或调用点、仓库相对路径、行号范围、所属 commit | 防止重放了同名但不同实现 |
| 执行上下文 | target 或 draft 路径、实际 execution phase/forward mode、采集命令中的 eager/CUDA Graph 设置 | 同一 shape 在不同执行阶段可能语义不同 |
| 输入结构 | `args/kwargs` 的顺序，以及 list、tuple、dict、标量、`None` 的嵌套结构 | P800 必须按原调用形态重建输入 |
| 输入 Tensor | 每个边界 Tensor 的原始数值 | 离线重放的实际输入 |
| 输入 Tensor 元数据 | 稳定叶子路径、shape、dtype、layout、stride、是否 contiguous；CUDA device 只作为来源信息 | shape 和 dtype 相同也可能走不同布局路径 |
| 非 Tensor 参数 | 实际参与语义且可稳定表示的原值，例如 `limit`、`epsilon` | 这些值是输入的一部分，但不需要伪装成 Tensor |
| 输出结构 | 与输入结构相同的容器描述 | Precision Gate 要求输出结构一致 |
| CUDA 期望输出 | 每个输出 Tensor 的原始数值，以及叶子路径、shape、dtype | P800 比较的固定参照 |
| 去重签名 | 本节后述的 SHA-256 | 保证同一调用形态只采一次 |

Demo 只接受可由这些内容独立重建的稠密 Tensor 边界。如果正确行为依赖输入间共享存储、原地修改、随机状态或无法稳定表示的运行时对象，Migration Agent 不得悄悄扩大采集范围，而应把该候选写为 `NEEDS_HUMAN`。

## 去重与三样本上限

去重签名是下面内容的规范 JSON（key 排序、稳定的 dtype/标量表示）的 SHA-256：

- `operator_id`；
- target/draft 与实际 execution phase/forward mode；
- 输入容器结构；
- 按稳定叶子路径排序的每个输入 Tensor 的 shape、dtype、layout、stride；
- 规范化后的非 Tensor 参数。

签名不包含 Tensor 数值和输出。这样，同一调用形态在模型执行中出现多次时，只保留第一次真实输入，而不会因为数值每次不同而反复 dump。

在唯一一次 Capture Session 中，每个待采集算子按首次观察顺序最多保存三个不同签名：

1. 已出现的签名只增加重复计数，不再保存 Tensor；
2. 新签名且尚不足三个时，保存为新的 Golden Sample；
3. 第四个及后续新签名只在 Run 中记录未保留数量和签名，不保存 Tensor；
4. Session 结束后不得为了补更多 shape 再访问 CUDA。

这一定义把“最多三种 shape”落实为最多三个真实调用形态；如果不同 phase、layout 或非 Tensor 参数使签名不同，也会占用一个名额。若前三个样本不足以验证某候选，该候选不能成为最小 Demo 的关闭对象。

## CUDA self-replay

Golden Sample 只有通过 self-replay 才算有效。self-replay 必须由同一次 CUDA Capture Session 内启动的全新进程执行，而不是复用打桩时还在内存中的 Tensor：

1. 从落盘文件读取并重建输入容器、Tensor dtype/shape/layout/stride 和非 Tensor 参数；
2. 通过记录的 Semantic Operator 入口执行一次；
3. 检查输出容器结构、dtype 和有限值；
4. 使用 Contract 中固定的 Precision Gate 比较新输出与已保存的 CUDA 期望输出；
5. 每个已保留样本都通过后，Golden Run 才可成功；命令、退出码、逐样本结果和日志写入 `self-replay/`。

比较器在这里是一项行为要求，不指定 `torch.testing`、其他 Torch API 或 NumPy。当前 SGLang 打桩在 active CUDA Graph capture 时会跳过 Tensor dump，因此采集命令必须明确其 CUDA Graph 设置，并在允许安全落盘的边界采集；不能在 active graph capture 中强行复制 Tensor。

## 默认禁止采集的内容

- checkpoint 权重、`nn.Parameter`、module state；本 Demo 选择不含权重的 Semantic Operator，需要权重才能重放的候选不合格；
- Semantic Operator 边界内部的中间 Tensor；
- 不属于该边界输入的完整 batch、prompt/token 内容、KV cache；
- module、stream、handle、ForwardBatch 等运行时对象；
- optimizer state、随机状态，以及依赖随机行为的样本；
- 凭证、token、完整环境变量或无关日志；
- 已有相同去重签名的重复 Tensor，或超过三个签名后的 Tensor。

这里不禁止保存 `epsilon`、`limit` 等必要标量；它们按非 Tensor 参数写入 `metadata.json`。也不额外保存每层权重来“以防万一”。

## Handoff Bundle

Handoff Bundle 只包含三部分：

```text
<bundle>/
├── migration-spec.md
├── runs/<golden-run-id>/...
└── manifest.json
```

其中 `migration-spec.md` 是进入 `WAITING / HANDOFF` 时的完整副本，Golden Run 是已经通过 CUDA self-replay 的不可变 Run。无需再加入 README：唯一下一动作在 Spec 中，执行命令和结果在 Run 中。

`manifest.json` 对除自身外的每个文件保存：按字典序排列的 bundle 相对路径、字节数和 SHA-256。manifest 不得包含绝对路径、`..` 路径或符号链接。

工具必须在 CUDA 端生成后校验一次，并在 P800 使用任何样本前再次校验：文件集合必须完全一致，不能缺文件、出现未登记文件、大小不符或 SHA-256 不符。工具还应在两端输出 `manifest.json` 自身的 SHA-256，供人工比对；它证明复制前后字节一致，不承担签名或身份认证。

Spec 只记录 `manifest_path`，不记录 `manifest_digest`。如果 manifest 哈希 Spec，而 Spec 又写入 manifest 的 digest，就会形成自引用：每次写 digest 都改变 Spec，进而改变 manifest。最终 manifest 生成后整个 bundle 即不可再修改；人工只负责复制，P800 校验通过后才能开始 replay。

## 本票不做的决定

- Tensor 使用 `torch.save`、safetensors、NumPy 还是其他格式；
- 比较器最终调用 `torch.testing`、其他 Torch API 还是 NumPy；
- Handoff Bundle 是否再压缩，以及压缩格式；
- 最终 CLI 或 Skill 的命令名称。

这些实现选择不能改变上述逻辑数据和校验要求。特别是昆仑修改版 Torch 是否支持某个比较 API，必须等真实 P800 环境验证后再决定。

## 源码依据

1. `python/sglang/kernel_api_logging.py:244-276` 递归拆分 Tensor 与非 Tensor 元数据，并保留 list/tuple/dict 的类型、长度和 key；说明边界 Tensor 与调用结构都需要保存。
2. `python/sglang/kernel_api_logging.py:291-345` 分别落盘输入、输出与完成状态；其中 `torch.save` 只是当前调试实现，不能据此把 `.pt` 写死为跨设备契约。
3. `python/sglang/kernel_api_logging.py:402-472` 在原函数边界包裹输入和输出，且 `443-447` 在 active CUDA Graph capture 时跳过 Tensor dump；说明采集模式必须明确，不能在 graph capture 内强行 dump。
4. `python/sglang/test/precision_baseline_store.py:95-120` 只选择相同 `capture_signature` 的 baseline，形态不一致则跳过；说明样本签名是可比较性的前置条件。
5. `python/sglang/jit_kernel/fp8_quantize.py:63-84` 明确检查最后一维和前导维 stride；说明仅保存 shape、dtype 会漏掉影响实现路径的布局信息。
6. `sglang-kunlun/sglang_kunlun/hooks/layers/attention/nsa/nsa_indexer.py:27-39` 已将 shape、dtype、device、contiguous 作为递归诊断信息，佐证 Kunlun 路径也需要结构化 Tensor 元数据。
7. `python/sglang/srt/compilation/inductor_pass.py:65-91` 使用排序 JSON 的 SHA-256，`python/sglang/jit_kernel/utils.py:74-97` 使用仓库相对标识和文件字节做 SHA-256；两者为规范签名和跨目录文件完整性提供了现有代码依据。

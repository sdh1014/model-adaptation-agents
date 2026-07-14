# 定义一次性 CUDA Golden Run 与交接包契约

Type: research
Status: resolved
Blocked by: 01

## Question

为了让一次 CUDA Capture Session 生成的数据能在 P800 离线重放，Golden Run 与 Handoff Bundle 最少需要保存哪些输入、CUDA 输出、元数据、源码锚点、版本、去重签名、自校验结果和完整性信息？契约需要支持一个不含权重的 Semantic Operator、最多三种 shape，并明确哪些数据禁止默认采集，但暂不决定最终序列化库。

## Comments

## Answer

Golden Sample 必须保存 Semantic Operator 的边界输入 Tensor、CUDA 期望输出 Tensor、容器结构、必要非 Tensor 参数，以及 shape、dtype、layout、stride 等重放信息。权重、内部中间 Tensor、完整 batch/KV cache 和运行时对象默认不采集；因此 Demo 只能选择不依赖权重、随机状态或共享存储语义的算子。`epsilon`、`limit` 这类必要标量按元数据保存，不需要变成 Tensor。

去重签名使用 `operator_id + target/draft/执行阶段 + 输入结构 + 输入 Tensor 的 shape/dtype/layout/stride + 非 Tensor 参数` 的规范 JSON SHA-256，不包含 Tensor 数值和输出。唯一 Capture Session 内每个待采集算子按首次出现最多保留三个不同签名；重复和第四个以后的签名只计数，不再保存 Tensor，也不允许为补 shape 创建第二次 Session。

Golden Run 必须在同一次 CUDA Session 内由全新进程从落盘字节完成 self-replay，所有样本通过 Contract 的 Precision Gate 后才可交接。Handoff Bundle 只包含进入 `WAITING / HANDOFF` 时的 Migration Spec、该 Golden Run 和 `manifest.json`；manifest 对自身之外的文件记录相对路径、大小与 SHA-256，并在 CUDA、P800 两端校验。Spec 改为只保存 `manifest_path`，避免 Spec 与 manifest digest 互相改变造成自引用。

本票只固定逻辑契约，不决定 Tensor 序列化库、比较器 API、压缩格式或最终命令入口。完整字段、禁止采集项、验证步骤与源码证据见 [研究记录](../../../work/research/golden-run-handoff-contract.md)。

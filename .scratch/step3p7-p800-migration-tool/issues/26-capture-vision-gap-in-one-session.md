# Ticket 26：在同一次 Session 采集视觉 attention 缺口

Status: Done

## Goal

把 Step-3.7 单图路径上的
`sglang.srt.layers.attention.triton_ops.prefill_attention._fwd_kernel`
加入 revision 6 的完整 gap queue。CUDA 端只启动一次 TP8 模型进程，同时执行固定
文本请求和固定单图请求，让全部计划算子各自保存最多三种 rank 0 shape。

## Acceptance

- 新 Scan Run 绑定 Contract revision 6；五个 `CAPTURE_REQUIRED` 项在
  `gap_queue` 和 `capture_plan` 中一一对应，attention 不能再作为未采集的
  deferred 候选。
- 单图请求使用固定 SGLang worktree 中的
  `examples/assets/example_image.png`，保存并校验 SHA-256；不依赖运行时下载 URL。
- 请求文本包含 Step-3.7 的 `<im_patch>`，文本和单图请求都固定
  `temperature=0 / max_new_tokens=1`。
- attention 继续使用现有 `context_attention_fwd` 作为 capture seam。该函数返回
  `None`，collector 必须在原调用完成后把被写入的 `o` 保存为 CUDA expected。
- `capture_golden.py --mode prepare-session` 只生成一个 session config；插件从该
  config 同时注册五个现有调用，不新增模型 helper、自定义算子或整层 wrapper。
- rank 0 以外不落盘；每个算子独立按输入 shape 去重并最多保存三份，不能把五个
  算子共用成三份样本。
- session preflight 必须逐算子完成 capture、rank 过滤和 CUDA self-replay，且明确
  不消耗正式 Capture Session。
- 这张票不预设其余四项的 P800 修复入口；除已有 SwiGLU adapter 外，等队列推进
  到对应算子时由 Agent 根据 Kunlun 原调用点补齐 replay adapter。

## Blocked by

- None

## Comments

- SOURCE 实现和静态测试见 `runs/multimodal-capture-tool-001`；准备结果见
  `runs/capture-session-tool-001`。
- 本机没有 Torch/CUDA，因此没有把 SOURCE 结果写成 CUDA preflight PASS。真实
  preflight 仍须在固定 revision 的 CUDA 机器执行。
- 固定源码依据：CUDA capability 8.0 默认选择 `triton_attn`；Kunlun shim 同样
  报告 capability 8.0，固定 plugin 没有视觉 attention override。正式启动日志仍
  要记录实际解析出的 multimodal backend。

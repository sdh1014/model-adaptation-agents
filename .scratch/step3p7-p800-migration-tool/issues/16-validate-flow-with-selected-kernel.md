# 使用扫描后选中的 kernel 验证端到端流程

Type: task
Status: ready-for-human
Blocked by: 14, 15

## What to build

人工把交接包复制到 P800 后，先验证 manifest，再用
`_swiglu_silu_clamp_mul` 的一至三个 Golden Samples 验证：

```text
恢复 Spec -> baseline replay -> 精度比较 -> 最多五轮修复 -> 最终结论
```

只有 P800 baseline 至少一个样本执行或精度失败，它才是本 Demo 的真实 Operator
Gap。源码缺少绑定或预期性能较慢都不能代替 baseline。

## Acceptance criteria

- P800 开始执行前，bundle、manifest、Spec 绑定和固定源码 revision 全部通过。
- baseline 使用样本中的 `x` 和 `gemm1_limit` 调用当前 Kunlun MoE 的
  `kunlun_ops.swiglu` 路径，不加载完整 MoE 或新增 helper。
- baseline 对全部样本执行，不计 repair attempt。
- baseline FAIL 后最多五轮修复；每轮一个假设，失败恢复基线，通过 patch 保持为
  唯一工作区修改。
- PASS 要求全部样本通过结构、dtype、有限值和固定
  `torch.testing.assert_close` 门槛。
- baseline 全部 PASS 时，如实进入 `BLOCKED`，说明首选静态候选不是实机
  correctness gap；不伪造失败、不重访 CUDA。
- 需要新增 C++/自定义 kernel/底层注册、五轮耗尽或出现未知工作区修改时，形成证据
  完整的 `BLOCKED`。
- 最终 Run、补丁、比较结果、日志和 Spec 状态可由新会话独立复核。

## Comments

本票据只关闭一个真实缺口来证明流程。剩余 gap queue 后续按一个 kernel 一张票据
扩展。

2026-07-18 已接受 P800 manifest verification：

- `runs/p800-handoff-review-r5-001` 确认 P800 verification 分支只新增一个 Run，
  13 个文件与 CUDA manifest 一致，审查未读取 Tensor；
- Working State 已进入 revision 28 `ACTIVE / P800_REPAIR`，baseline 尚未执行，
  `attempts_used` 仍为 0；
- `model-adaptation/references/p800-baseline-replay.md` 固定三个全新 Run：
  workspace check、bundle Golden kernel replay 和 baseline assessment；
- baseline 只调用现有 `kunlun_ops.swiglu`，不启动完整模型、不修改
  SGLang-Kunlun。数值 FAIL 是正常的 gap 证据，全部 PASS 则按 Contract
  `BLOCKED`，两种结果都先上传后停止。
- `runs/adapter-003` 收紧失败分类：只有现有 Kunlun 调用本身的执行失败或明确的
  输出/精度失败形成 baseline FAIL；依赖、设备搬运或比较器异常属于工具错误，
  不能写成 Operator Gap。

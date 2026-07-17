# CUDA 采集 preflight

本步骤让 Claude Code 在 N 卡机器上验证三件事：revision 4 的 target-only eager `scan-004` 与固定原始源码一致；SGLang 的通用 Hook 能直接打到 `Step3p5MLP.forward`；TP8 配置下只有 rank 0 会保存最多三种 shape 的 `x` 和 `output`。

preflight 使用小型合成 BF16 tensor 和原始 `Step3p5MLP.forward`，不启动模型、不读取 checkpoint，也不消耗唯一正式 CUDA Capture Session。它只验证 Hook 与样本格式，不声称完成 checkpoint 权重参与的模型内重放。

## 前提

- 当前目录是 `model-adaptation-agents` 仓库。
- 当前 Python 是将来启动 SGLang 的 CUDA Python，`torch.cuda.is_available()` 为真。
- 调用者已显式设置非空的 `SGLANG_WORKTREE`，且它指向固定原始 revision `49e384ce9d304648e9959666ecb8ce8cd98d0deb` 的 SGLang 0.5.14 源码；Agent 不猜测默认目录。
- `runs/cuda-preflight-001` 尚不存在；如果已经存在，换一个新的编号，不能覆盖旧 Run。

## 执行

先硬检查调用者确实提供了源码路径。失败时停止，不继续安装或创建 Run：

```bash
if [ -z "${SGLANG_WORKTREE:-}" ]; then
  echo "SGLANG_WORKTREE must be set to the fixed SGLang checkout" >&2
  exit 2
fi
```

然后用固定源码中的真实 `HookRegistry` 做一次 CPU 侧接线检查。该测试不会加载 checkpoint，也不会消耗 CUDA Session：

```bash
SGLANG_WORKTREE="$SGLANG_WORKTREE" \
  python tests/test_ticket12_real_sglang_hook_integration.py -v
```

结果必须是 `1 test ... OK`，不能是 `skipped`。然后安装本仓库提供的轻量采集插件；它没有额外依赖：

```bash
python -m pip install -e ./model-adaptation --no-deps --no-build-isolation
```

然后执行：

```bash
python model-adaptation/scripts/capture_golden.py \
  --mode preflight \
  --spec migration-spec.md \
  --run-dir runs/cuda-preflight-001 \
  --scan-result runs/scan-004/result.json \
  --operator-id Step3p5MLP.forward \
  --sglang-worktree "$SGLANG_WORKTREE"
```

命令会启动一个 preflight worker，通过 SGLang 的 `sglang.srt.plugins` 入口安装 `Step3p5ForCausalLM.forward` 上下文 Hook 和原始 `Step3p5MLP.forward` Hook。worker 使用原始 MLP 方法与小型投影替身保存 rank 0 的三种 shape，并验证重复调用和第四种 shape 不会新增样本。它不会创建或调用任何特殊 SwiGLU 函数。

## 验收

只有以下条件同时满足，preflight 才算通过：

- 真实 `HookRegistry` 接线测试在固定 SGLang revision 上得到 `1 test ... OK`。
- `runs/cuda-preflight-001/result.json` 中 `passed` 为 `true`、`capture_status` 为 `PREFLIGHT_PASSED`、`consumes_capture_session` 为 `false`。
- `capture-state.json` 中 `tp_rank=0`、`tensor_parallel_size=8`、`saved_shape_count=3`、`repeated_call_count=1`、`skipped_call_count=1`，并保留一条不含 tensor 数值的 `skipped_signatures` 记录。
- `samples/` 只有三份 `.pt`，每份只包含 `x`、CUDA `output`、`limit` 和必要元数据，不包含权重、`gate_up`、`gate` 或 `up`。
- `preflight-capture-summary.json` 记录两个已应用 Hook、实际 Torch 版本、CUDA 设备、原始模型源码路径、`tp_rank=0`、`tensor_parallel_size=8`、`checkpoint_loaded=false` 和 `loaded_model_replay_performed=false`。

失败时保留整个 Run，报告 `result.json` 和 preflight 日志，不重写同一个 Run，也不要把失败解释成 P800 算子缺口。

通过后同样停止，把以下文件交回当前设计会话确认；不要直接开始正式 CUDA Session：

```text
runs/cuda-preflight-001/result.json
runs/cuda-preflight-001/capture-config.json
runs/cuda-preflight-001/capture-state.json
runs/cuda-preflight-001/preflight-capture-summary.json
runs/cuda-preflight-001/preflight-capture.log
```

## 后续正式启动参数边界

preflight 通过后仍需单独确认，才进入正式模型采集。CUDA 与 P800 的 SGLang 启动参数保持一致：

```bash
MODEL_PATH=stepfun-ai/Step-3.7-Flash
MODEL_REVISION=5f6244077ac62e04eec3f320501ff8c2b293373a

python -m sglang.launch_server \
  --model-path "$MODEL_PATH" \
  --revision "$MODEL_REVISION" \
  --tp-size 8 \
  --dtype bfloat16 \
  --cuda-graph-backend-decode disabled \
  --cuda-graph-backend-prefill disabled
```

不要追加 `--speculative-algorithm`、量化参数、MTP 开关、`--enable-multi-layer-eagle`、`--attention-backend`、`--prefill-attention-backend`、`--decode-attention-backend` 或 MoE backend 参数。这里的 eager 表示 decode 与 prefill 都禁用 CUDA Graph，不使用仍经过 graph capture/replay 路径的 `--debug-cuda-graph`。CUDA 正式采集时只额外设置 `MODEL_ADAPTATION_CAPTURE_CONFIG` 环境变量；它不是 SGLang 启动参数。实际解析出的 attention 与 MoE backend 从启动日志记录，不回写成 Contract 参数。

`MODEL_PATH` 与 `MODEL_REVISION` 必须分别等于 Contract checkpoint 的 `@` 前后两部分。正式采集和两端重放都使用这两个值与 TP8；插件会从 SGLang 实际生效的 `model_path` 和 `revision` 再核对一次，缺失或不一致时不会执行重放。

正式采集结束后先停止带 `MODEL_ADAPTATION_CAPTURE_CONFIG` 的模型进程，再执行 `replay_compare.py --mode prepare-model-replay`。该动作会把 `capture-state.json` 的 `capture_closed` 写为 `true`，之后任何采集调用都会失败；它不会提前把 Golden Run 标成 `SEALED`。随后用生成的 `MODEL_ADAPTATION_REPLAY_CONFIG` 和上面的同一启动命令加载 TP8 模型，发送同一 Demo 输入触发原始 `Step3p5MLP.forward` Hook；确认 `replay-result.json` 已生成后先停止 replay 模型进程，再执行 `--mode finalize-model-replay`。初始化与 finalize 都会拒绝被改写的容差或缺失的 shape；finalize 若在状态或结果写入时中断，可用相同命令安全重试。只有全部 shape 通过时状态才变为 `SEALED`，否则变为 `FAILED`。

CUDA 和 P800 都只在 rank 0 的相同 `Step3p5MLP` 实例上处理 Golden Sample；其他 rank 正常执行模型但不额外落盘或比较。不能退回为独立无权重函数调用。

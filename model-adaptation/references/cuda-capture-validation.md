# CUDA 采集 preflight

本步骤让 Claude Code 在 N 卡机器上验证三件事：`scan-002` 选择的特殊 SwiGLU 边界仍与固定源码一致；SGLang 的通用 Hook 能打到真实 helper；候选样本能由另一个进程读回并用固定精度门槛重放。

preflight 使用小型合成 BF16 tensor，不启动模型、不读取 checkpoint，也不消耗唯一正式 CUDA Capture Session。

## 前提

- 当前目录是 `model-adaptation-agents` 仓库。
- 当前 Python 是将来启动 SGLang 的 CUDA Python，`torch.cuda.is_available()` 为真。
- 调用者已显式设置非空的 `SGLANG_WORKTREE`，且它指向固定 revision `6274831d9fef7bba04eb59302caac24563a974c9` 的 SGLang 0.5.14 源码；Agent 不猜测默认目录。
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
  --scan-result runs/scan-002/result.json \
  --operator-id activation.step_swiglu_with_limit \
  --sglang-worktree "$SGLANG_WORKTREE"
```

命令会依次启动两个新进程。第一个进程通过 SGLang 的 `sglang.srt.plugins` 入口安装 Hook，调用固定源码中的真实 `step_swiglu_with_limit`，保存三种调用形态并验证重复与第四种形态不会新增样本；第二个进程用 `torch.load(..., weights_only=True)` 读回样本，再用同一个 helper 和 `torch.testing.assert_close(atol=1e-2, rtol=2e-2)` 重放。

## 验收

只有以下条件同时满足，preflight 才算通过：

- 真实 `HookRegistry` 接线测试在固定 SGLang revision 上得到 `1 test ... OK`。
- `runs/cuda-preflight-001/result.json` 中 `passed` 为 `true`、`capture_status` 为 `PREFLIGHT_PASSED`、`consumes_capture_session` 为 `false`。
- `capture-state.json` 中 `saved_sample_count=3`、`repeated_call_count=1`、`skipped_call_count=1`，并保留一条不含 tensor 数值的 `skipped_signatures` 记录。
- `preflight-capture-summary.json` 记录两个已应用 Hook、实际 Torch 版本、CUDA 设备和固定 helper 路径。
- `preflight-verify-summary.json` 记录 `new_process=true`、三份样本、`atol=0.01` 和 `rtol=0.02`。

失败时保留整个 Run，报告 `result.json` 和两份 preflight 日志，不重写同一个 Run，也不要把失败解释成 P800 算子缺口。

通过后同样停止，把以下文件交回当前设计会话确认；不要直接开始正式 CUDA Session：

```text
runs/cuda-preflight-001/result.json
runs/cuda-preflight-001/capture-state.json
runs/cuda-preflight-001/preflight-capture-summary.json
runs/cuda-preflight-001/preflight-verify-summary.json
runs/cuda-preflight-001/preflight-capture.log
runs/cuda-preflight-001/preflight-verify.log
```

## 后续正式启动参数边界

preflight 通过后仍需单独确认，才进入正式模型采集。CUDA 与 P800 的 SGLang 启动参数保持一致：

```bash
python -m sglang.launch_server \
  --model-path "$MODEL_PATH" \
  --tp-size 8 \
  --dtype bfloat16 \
  --speculative-algorithm EAGLE
```

不要追加量化参数、MTP 开关、`--enable-multi-layer-eagle`、`--attention-backend`、`--prefill-attention-backend`、`--decode-attention-backend` 或 MoE backend 参数。CUDA 正式采集时只额外设置 `MODEL_ADAPTATION_CAPTURE_CONFIG` 环境变量；它不是 SGLang 启动参数。实际解析出的 attention 与 MoE backend 从启动日志记录，不回写成 Contract 参数。

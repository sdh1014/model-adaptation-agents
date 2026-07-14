# PROTOTYPE — 基于 `torch.testing` 的最小精度比较接口

> 要验证的问题：既然用户已经在 P800 实机确认 `torch.testing` 可用，最小比较器是否可以直接绑定 `torch.testing.assert_close`，同时仍严格执行 Migration Spec 中的 Precision Gate，并给 Migration Agent 足够但不过量的失败证据？
>
> 这是方案逻辑原型，不是生产实现。P800 可用性是用户提供的实机结论；当前本地工作区没有安装 Torch，因此这里只验证接口和判定顺序，不重复冒充实机验证。

## 结论

不再设计 Torch/NumPy Adapter。比较器直接使用 `torch.testing.assert_close`，并保持为一个 Deterministic Tool 内部函数：

```python
compare_sample(
    spec_path: Path,
    golden_sample_dir: Path,
    actual_output: Any,
) -> dict
```

- `spec_path`：比较器自己从 Contract 读取 `atol`、`rtol`，接口不提供覆盖参数；
- `golden_sample_dir`：读取 CUDA expected output 和已保存的输出结构；
- `actual_output`：P800 replay 刚产生的内存对象；
- 返回值：由 replay 工具写入本轮 Run 的 `compare.json`。

P800 actual output 默认不再单独保存一份 Tensor 文件。失败 Run 保存补丁、重放命令、比较结果和日志；如果需要重新观察实际输出，P800 仍在同一台机器上，可以用 Golden input 重放。这样不会为 Contract 允许的最多五轮尝试重复保存大 Tensor。

## 为什么不能只调用一次 `assert_close`

PyTorch 官方文档给出的接口是：

```python
torch.testing.assert_close(
    actual,
    expected,
    *,
    rtol=None,
    atol=None,
    equal_nan=False,
    check_device=True,
    check_dtype=True,
    check_layout=True,
    check_stride=False,
    msg=None,
)
```

但 Precision Gate 仍需在调用前做两项检查：

1. 官方语义允许 list 与 tuple 只要长度和元素匹配就通过，而 Contract 要求输出结构一致，所以必须先按 Golden Sample 的结构元数据检查容器类型、长度、dict keys 和 Tensor 叶子路径。
2. 官方语义允许对应位置相同的正负无穷通过，而 Contract 要求 CUDA Golden 与 P800 输出全部为有限值，所以必须先执行 `torch.isfinite`。

官方文档还说明，只传 `atol` 或只传 `rtol` 会报错，省略两者则使用 dtype 默认值。因此比较器必须同时、显式传入 Contract 固定的两个值，不能使用 Torch 默认容差。

## 固定判定顺序

对每个 Golden Sample：

1. 从 Contract 读取固定 `atol`、`rtol`；
2. 从 Golden Sample 读取 expected output 和结构元数据；
3. 检查实际输出的容器类型、长度、dict keys 和叶子路径完全一致；
4. 对应 Tensor 的 shape、dtype 必须一致；非数字叶子按原值相等；
5. expected 与 actual 的每个数字 Tensor 都必须全部有限；
6. 将对应 Tensor 都复制到 CPU，保留原 dtype；P800 执行设备单独留在 Run，不拿 CUDA/P800 设备名做相等判断；
7. 每个 Tensor 叶子执行：

```python
torch.testing.assert_close(
    actual_cpu,
    expected_cpu,
    atol=contract.atol,
    rtol=contract.rtol,
    equal_nan=False,
    check_device=True,   # 两边此时都已显式归一到 CPU
    check_dtype=True,
    check_layout=True,
    check_stride=False,  # Precision Gate 未要求输出 stride 一致
)
```

8. 捕获 `AssertionError` 文本作为诊断信息，但不解析错误文本来决定 PASS；是否通过只看上述检查与 `assert_close` 是否抛错；
9. 一个样本的所有叶子都通过，该样本才通过；最多三个 Golden Samples 全部通过，本轮 repair 才通过，不允许 skip。

## `compare.json` 最小结果

```json
{
  "sample_id": "sample-001",
  "passed": false,
  "backend": "torch.testing.assert_close",
  "torch_version": "<P800 runtime value>",
  "precision_gate": {
    "atol": "<from Contract>",
    "rtol": "<from Contract>",
    "check_dtype": true,
    "finite_required": true
  },
  "checks": {
    "structure_match": true,
    "dtype_match": true,
    "expected_finite": true,
    "actual_finite": true
  },
  "tensors": [
    {
      "path": "result",
      "shape": [1, 6144],
      "dtype": "torch.bfloat16",
      "passed": false,
      "max_abs_diff": 0.03125,
      "mismatch_count": 7,
      "message": "<torch.testing.assert_close AssertionError>"
    }
  ],
  "summary": "result 未通过 Contract 固定容差"
}
```

`max_abs_diff`、`mismatch_count` 和 `message` 只帮助 Agent 形成下一轮假设，不构成第二套判定标准。没有新的错误码、可忽略失败列表或 Agent 可修改的门槛。

## 逻辑草图

```python
def compare_sample(spec_path, golden_sample_dir, actual_output):
    gate = read_human_owned_precision_gate(spec_path)
    expected, expected_structure = load_expected(golden_sample_dir)

    require_exact_structure(actual_output, expected_structure)

    results = []
    for path, actual, expected in pair_output_leaves(actual_output, expected):
        require_same_shape_and_dtype(actual, expected)
        require_finite(actual)
        require_finite(expected)

        actual_cpu = actual.detach().cpu()
        expected_cpu = expected.detach().cpu()

        try:
            torch.testing.assert_close(
                actual_cpu,
                expected_cpu,
                atol=gate.atol,
                rtol=gate.rtol,
                equal_nan=False,
                check_device=True,
                check_dtype=True,
                check_layout=True,
                check_stride=False,
            )
            passed, message = True, None
        except AssertionError as error:
            passed, message = False, str(error)

        results.append(build_diagnostics(path, actual_cpu, expected_cpu, passed, message))

    return build_result(results, torch.__version__, gate)
```

## 可运行的界面走查

伴随脚本只展示四种状态下完整 request/result 形状，不实现数值计算：

```bash
python3 work/prototypes/comparator-interface.prototype.py
```

可查看：全部通过、结构不一致、非有限值、数值超差。脚本没有持久化，也不需要 Torch。

## 代码与官方依据

- 用户已在 P800 实机确认 `torch.testing` 可用；本工作区没有对应实机命令日志，因此方案将它记为用户确认的环境事实，而非本地复验结果。
- PyTorch 官方 [`torch.testing.assert_close` 文档](https://docs.pytorch.org/docs/2.12/testing.html)说明了容差公式、结构比较、有限值、dtype/layout/device/stride 参数和异常行为。
- SGLang 当前测试已经普遍使用显式 `atol/rtol` 的 `torch.testing.assert_close`；例如 `python/sglang/jit_kernel/tests/test_minimax_m3_rmsnorm.py:35-78` 覆盖三种 shape、BF16/FP16 和 strided input。
- SGLang 现有通用调试比较器记录 shape、dtype、最大绝对差和最大差位置，见 `python/sglang/srt/debug_utils/comparator/tensor_comparator/types.py:17-44` 与 `comparator.py:135-180`。本 Demo 只保留诊断所需的最小子集，不引入其 token 对齐、可视化或 skip 体系。

## Prototype verdict

用户确认该逻辑原型：P800 actual output 只在内存中比较，Run 只保存 `compare.json` 和日志，不额外保存 actual Tensor。比较器直接绑定 `torch.testing.assert_close`，不再保留多后端适配层。

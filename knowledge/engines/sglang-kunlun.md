# SGLang-Kunlun

来源：内部知识页 `sglang-kunlun 开发规范与扩展手册`，2026-06-26 蒸馏。

## 仓库形态

目标可能是完整 fork、独立平台插件或上游仓加本地 patch。先从当前 commit 判断，不预设形态。

## 评估边界

分别检查模型 registry、weight loader、model runner、Attention/KV Cache、MoE、量化、并行、平台发现和本地运行时。

上游 SGLang 支持模型不代表 SGLang-Kunlun 在 P800 上自动支持全部路径；插件仓没有模型类也不代表模型一定不支持，可能复用上游模型并只替换 backend、kernel 或 Hook。

## 实现优先级

```text
复用 SGLang 原生模型
→ 使用当前版本扩展接口
→ Kunlun backend/plugin 最小 override
→ 必要时请求修改核心
```

新增代码前先判断改动类型：这是新增平台能力、替换上游行为、补 Kunlun kernel，还是新增模型适配。不同类型放在不同目录，不要把所有逻辑都写成 Hook。

```text
需求类型                         推荐位置                                           注册方式                                  典型场景
新增 sgl_kernel API 的 Kunlun 实现 sglang_kunlun/kernels/sgl_kernel_kunlun/        bootstrap/sgl_kernel_stub.py 绑定        上游 from sgl_kernel import xxx
替换 Triton kernel / JIT helper   sglang_kunlun/kernels/kernel_ops.py 或子模块     register_triton_op / register_jit_op     上游 kernel[grid](...) 或 JIT helper
替换 Python 类方法/函数            sglang_kunlun/hooks/<对应模块>/                  plugin_hook REPLACE / AROUND             quantization、linear、speculative、distributed
新增 Attention backend 行为        hooks/layers/attention/                         register_attention_backend 或平台工厂     attention backend、metadata、prefill/decode
新增 KV Cache / Allocator 行为     hooks/mem_cache/                                KunlunSRTPlatform.get_*_cls()            MHA/MLA/NSA KV Pool、paged allocator
新增量化方法适配                  hooks/layers/quantization/                       plugin_hook 替换上游 quant method        W8A8、unquant、FP8
新增 MoE 行为                     hooks/layers/moe/                                plugin_hook 或 __init__.py 导入           fused MoE、EP MoE、token dispatcher
新增 speculative decoding 行为     hooks/speculative/ + kernels/sgl_kernel_kunlun/speculative.py  plugin_hook + stub/kernel 绑定        EAGLE、tree verify、draft/target worker
新增模型适配                      sglang_kunlun/models/                            SGLANG_EXTERNAL_MODEL_PACKAGE            模型结构、模型注册、特殊 forward
修改平台默认配置                  sglang_kunlun/platform/srt.py                    平台工厂方法 / apply_server_args_defaults page size、默认 backend、KV pool 类
启动前环境兼容                    sglang_kunlun/bootstrap/pre_shim.py              early shim                                Triton / torch.cuda 导入期兼容
```

## 开发流程

推荐流程：

```text
明确需求
→ 定位上游调用点
→ 分类改动类型
→ 选择目录
→ 最小实现
→ 注册导入
→ 写机制单测
→ 需要真实 Kunlun 验证时跑硬件 E2E
→ 提交
```

具体步骤：

1. 定位上游调用点：先读 `python/sglang/...` 中的原始实现和调用链。
2. 分类：判断是 stub、kernel_ops、plugin_hook、platform factory 还是 model。
3. 最小实现：只改当前必要路径，不做泛化重构。
4. 注册导入链：确保生产启动时能加载，不依赖测试偶然 import。
5. 普通 CI 先验证机制：import、Hook、stub、kernel replacement。
6. 真实 Kunlun 上验证数值、性能和多卡。
7. 保持 `test_cuda_only_contract.py` 通过。

## 新增算子放置规则

新增算子先判断上游入口来源：

```text
上游调用方式             放置位置 / 处理方式
sgl_kernel API           kernels/sgl_kernel_kunlun/，并在 sgl_kernel_stub.py 绑定
Triton kernel            register_triton_op，放到 kernel_ops.py 或 kernels 子模块
Python 方法或类逻辑       hooks/ 对应子目录，使用 plugin_hook
平台工厂可替换            优先在 platform/srt.py 返回子类
模型私有 forward          models/ 或 hooks/model_executor/
```

### sgl_kernel API

如果上游写法类似：

```python
from sgl_kernel import rmsnorm
from sgl_kernel.sampling import top_k_renorm_prob
```

推荐做法：

1. 在 `sglang_kunlun/kernels/sgl_kernel_kunlun/<domain>.py` 增加实现。
2. 在 `sglang_kunlun/bootstrap/sgl_kernel_stub.py` 把对应 symbol 绑定到 stub module。
3. 如果是子模块 API，确保 `sys.modules["sgl_kernel.<submodule>"]` 也有对应属性。
4. 写 import-safe 测试，验证没有真实 CUDA `sgl_kernel` 时也能导入并调用 stub。
5. 未实现 API 必须显式抛错，不静默切换到错误路径。

原文列出的参考点：

```text
sglang_kunlun/kernels/sgl_kernel_kunlun/sampling.py
sglang_kunlun/kernels/sgl_kernel_kunlun/speculative.py
```

### Triton kernel / JIT helper 替换

如果上游调用形态是：

```python
some_kernel[grid](arg1, arg2, BLOCK_SIZE=...)
```

使用 `kernel_ops.register_triton_op()`：

```python
@register_triton_op("sglang.srt.some.module", "some_kernel")
def some_kernel_kunlun(arg1, arg2, BLOCK_SIZE: int, **kwargs):
    torch.ops.xspeedgate_ops.some_kernel(arg1, arg2)
```

规则：

- replacement 函数签名尽量兼容上游参数。
- 不需要的 Triton meta 参数可以保留但忽略。
- 如果上游直接函数调用而不是 `kernel[grid]`，metadata 加 `{"call_style": "direct"}`。
- 显式处理 dtype 转换，常见做法是转为 Kunlun op 需要的 `torch.int32` 或 `torch.int64`。
- 如果上游模块已经把旧 symbol import 到别处，机制测试要覆盖已导入别名也能被替换。

原文列出的参考点：

```text
sglang_kunlun/kernels/kernel_ops.py: KernelLauncher
sglang_kunlun/kernels/kernel_ops.py: register_triton_op
test/test_kernel_ops.py: direct call style
```

### Python 行为替换

如果要改变上游 Python 方法，例如量化方法、worker 行为或初始化逻辑，放到 `hooks/`。

`REPLACE` 模板：

```python
from sglang.srt.plugins.hook_registry import HookType, plugin_hook


@plugin_hook(
    "sglang.srt.some.module.SomeClass.some_method",
    type=HookType.REPLACE,
)
def some_method_kunlun(self, arg1, arg2):
    return result
```

`AROUND` 模板：

```python
@plugin_hook(
    "sglang.srt.some.module.SomeClass.some_method",
    type=HookType.AROUND,
)
def some_method_around_kunlun(original_fn, self, *args, **kwargs):
    ret = original_fn(self, *args, **kwargs)
    return ret
```

选择规则：

```text
REPLACE  上游实现不适合 Kunlun，必须完全接管
AROUND   上游大部分逻辑可复用，只修正参数、dtype、初始化行为或条件判断
```

原文列出的参考点：

```text
hooks/layers/quantization/w8a8_int8.py        完全替换 W8A8 apply
hooks/layers/linear.py                        包裹 ColumnParallelLinear.__init__ 修 bias dtype
hooks/speculative/eagle_worker_v2.py          包裹 EAGLE verify
```

## 新增模块目录规范

`hooks/` 下目录尽量镜像上游 SGLang 路径，查到上游文件后应能直接推断 Kunlun patch 位置。

```text
上游路径                         Kunlun Hook 路径
sglang.srt.layers.*              sglang_kunlun/hooks/layers/
sglang.srt.layers.attention.*    sglang_kunlun/hooks/layers/attention/
sglang.srt.layers.quantization.* sglang_kunlun/hooks/layers/quantization/
sglang.srt.layers.moe.*          sglang_kunlun/hooks/layers/moe/
sglang.srt.mem_cache.*           sglang_kunlun/hooks/mem_cache/
sglang.srt.model_executor.*      sglang_kunlun/hooks/model_executor/
sglang.srt.speculative.*         sglang_kunlun/hooks/speculative/
sglang.srt.distributed.*         sglang_kunlun/hooks/distributed/
sglang.srt.constrained.*         sglang_kunlun/hooks/constrained/
```

`@plugin_hook` 只有在模块被 import 后才会注册。新增 Hook 文件后必须确认导入链：

1. 新增在已有 package 下，例如 `hooks/layers/quantization/new_quant.py`，在对应 `__init__.py` 中 import。
2. 新增一级模块，例如 `hooks/scheduler/`，加入 `HOOK_MODULES`。
3. 不依赖测试中的偶然 import，生产启动必须能通过 `register_all()` 导入。

原文列出的参考点：

```text
sglang_kunlun/hooks/registry.py: HOOK_MODULES
sglang_kunlun/hooks/layers/__init__.py
sglang_kunlun/hooks/speculative/__init__.py
```

如果上游已经通过 `SRTPlatform` 提供工厂方法，优先在 `KunlunSRTPlatform` 返回 Kunlun 子类，不要 Hook 上游构造逻辑。

适合平台工厂的内容：

```text
KV Pool 类
Paged allocator
Attention backend class
Graph runner
编译 backend
平台默认参数
```

原文列出的参考点：`sglang_kunlun/platform/srt.py` 中的 `KunlunSRTPlatform` 工厂方法。

## 新增模型规范

模型代码优先放到：

```text
sglang_kunlun/models/<model_name>.py
```

`sglang_kunlun/models/__init__.py` 会设置 `envs.SGLANG_EXTERNAL_MODEL_PACKAGE.set(__name__)`，因此 `sglang_kunlun.models` 是 SGLang external model package。

新增模型前先判断是否真的需要模型文件：

```text
问题                                   推荐做法
HuggingFace config 识别不到架构          新增 external model registration
模型结构与上游相同，只是算子不兼容        不新增模型，补 Hook 或 kernel
forward 中有模型私有算子                 新增模型文件或 Hook 对应上游模型方法
权重命名 / 加载逻辑不同                  优先模型适配，避免污染通用层
仅 dtype / layout 不匹配                 优先 quantization / linear / attention / KV 层适配
```

模型文件保持薄适配：

1. 优先继承上游模型类。
2. 只覆盖必要方法。
3. 复用 `sglang_kunlun.hooks.layers` 中已有 Kunlun 层能力。
4. 如果通用算子缺失，补到 `kernels/` 或 `hooks/layers/`，不要写成模型私有逻辑。

新增模型至少测试：

```text
import sglang_kunlun.models.<model_name> 不失败
加载 sglang_kunlun.models 后 SGLANG_EXTERNAL_MODEL_PACKAGE 被设置
可构造小 config 时，模型类能被 registry 找到
依赖 Kunlun runtime 时，用 mock 隔离外部依赖，避免普通 CI 失败
```

## 测试规范

新增功能至少考虑四层测试：

```text
测试层级                  放置位置                  目标                                         是否需要 Kunlun 硬件
Import / registration      sglang-kunlun/test/       模块可导入、Hook 可注册、entry 行为正确       否
Mechanism                  sglang-kunlun/test/       pre-shim、kernel_ops、stub、Hook 包裹逻辑     否，使用 mock
Operator adapter           test/ 或硬件专用目录       参数转换、dtype、shape、Kunlun op 调用       可 mock 或需要硬件
E2E inference              独立硬件测试脚本 / CI      真实模型推理正确性和性能                    是
```

当前仓库机制测试参考：

```text
test_pre_shim.py              pre-shim 机制
test_kernel_ops.py            kernel replacement 机制
test_cuda_only_contract.py    禁止 torch.xpu 契约
```

### plugin_hook 测试

新增 Hook 至少测试：

1. 目标模块 import 后 Hook 函数存在。
2. `register_all()` 或对应 `__init__.py` 能导入该模块。
3. Hook 函数本身在 mock 对象上能执行关键行为。

示例：

```python
def test_my_hook_module_imports():
    import sglang_kunlun.hooks.layers.my_module  # noqa: F401


def test_my_hook_behavior_with_mock():
    from sglang_kunlun.hooks.layers.my_module import my_hook

    class Dummy:
        pass

    obj = Dummy()
    result = my_hook(lambda self: "upstream", obj)
    assert result == "expected"
```

普通单测不要依赖真实 `torch_xmlir`、真实 XPU 或真实大模型。

### kernel_ops 替换测试

新增 `kernel_ops` 替换重点测试：

1. 替换原模块符号。
2. 替换已导入别名。
3. `kernel[grid]()` 形式可用。
4. direct call style 可用。

测试模板：

```python
def test_new_kernel_patch(monkeypatch):
    import sys
    import types
    from sglang_kunlun.kernels import kernel_ops

    def original(*args, **kwargs):
        return "original"

    source = types.ModuleType("sglang.fake_source")
    source.my_kernel = original
    user = types.ModuleType("sglang.fake_user")
    user.my_kernel = original
    sys.modules[source.__name__] = source
    sys.modules[user.__name__] = user

    def replacement(*args, **kwargs):
        return "replacement"

    spec = kernel_ops.KernelSpec(source.__name__, "my_kernel", replacement)
    monkeypatch.setattr(kernel_ops, "_TRITON_OPS", {(source.__name__, "my_kernel"): spec})
    monkeypatch.setattr(kernel_ops, "_JIT_OPS", {})

    kernel_ops.install()
    assert source.my_kernel[lambda meta: (1,)]() == "replacement"
    assert user.my_kernel[lambda meta: (1,)]() == "replacement"
```

### sgl_kernel_stub API 测试

新增 stub API 后至少验证：

1. `import sgl_kernel` 不失败。
2. `from sgl_kernel.<submodule> import <symbol>` 不失败。
3. 调用时会转到 Kunlun 实现或 mock 的 Kunlun op。
4. 未实现 API 显式报错。

### 算子数值测试

如果有 CPU/PyTorch reference，可以写小 shape 数值测试：

1. 构造小 shape。
2. 用 PyTorch reference 计算 expected。
3. 调用 Kunlun adapter。
4. 比较 shape、dtype 和数值容差。

如果 adapter 内部必须调用 `kunlun_ops`，普通 CI 中使用 `unittest.mock` 替换硬件 op，只验证参数顺序、dtype 转换和输出 tensor 分配；真实数值验证放到硬件 CI 或手动验证脚本。

### E2E 建议

真实 Kunlun 环境建议保留：

```text
场景                   命令特征                                             验证点
最小 greedy decode      --tp-size 1，不开 speculative                       服务能启动，短 prompt 输出合理
W8A8 int8               --quantization w8a8_int8                            Linear / MoE 路径可运行
EAGLE speculative       --speculative-algorithm EAGLE，SGLANG_ENABLE_SPEC_V2=1 draft/verify 路径稳定
TP 多卡                 --tp-size 2/8                                       parallel state、通信、KV layout 正常
Attention backend       --attention-backend fa3/kunlun                      prefill/decode 均正常
KV cache 压力           大 --max-total-tokens                               cache 写入、page table、offloading 稳定
```

## 代码风格与维护规则

保持 import-safe：

- 不在模块顶层强制导入只能在 Kunlun 机器存在的库，除非已有 guard。
- `kunlun_ops`、`torch_xmlir` 等依赖优先放在函数内部导入。
- 必须顶层导入时，保证非 Kunlun 环境测试不会失败。

不要使用 `torch.xpu`：

- 使用 `torch.cuda`。
- platform dispatch key 保持 `cuda`。
- 设备字符串优先沿用上游传入的 device 或 `cuda`。
- 保持 `test_cuda_only_contract.py` 通过。

Hook 目标必须写完整上游路径。开发时先确认：

```text
module path
class/function 名称
方法签名
返回值约定
上游是否已有平台扩展点
```

`REPLACE` 要复制上游语义，不只满足当前模型。通用方法替换必须保持：

```text
参数名和位置兼容
返回类型兼容
dtype / device / shape 兼容
异常明确
```

如果只适配某个模型，应放在模型适配层，不要污染通用层。

对未支持 kernel 或能力，优先显式失败，不静默走 CPU 或错误 fallback。推理系统错误结果的代价高于启动失败。

外部依赖导入原则：

```text
代码位置                         是否可顶层导入硬件依赖
bootstrap/pre_shim.py             谨慎，必须 try/except
platform/srt.py                   activate() 内可探测，其他地方避免强依赖
hooks/*                           尽量函数内导入
kernels/sgl_kernel_kunlun/*       可以导入，但测试要 mock
test/*                            默认不依赖真实硬件
```

## 常见开发场景

### 场景 A：上游报 `sgl_kernel.xxx NotImplementedError`

处理方式：

1. 找到缺失 symbol。
2. 在 `kernels/sgl_kernel_kunlun/` 新增或复用实现。
3. 在 `bootstrap/sgl_kernel_stub.py` 绑定 symbol。
4. 写 stub import 测试。
5. 如果实现调用 Kunlun op，用 mock 验证调用参数。
6. 在真实 Kunlun 上跑触发该路径的最小 E2E。

### 场景 B：某个 Triton kernel 在 Kunlun 上不能跑

处理方式：

1. 定位上游 kernel 所在 module 和 symbol。
2. 在 `kernel_ops.py` 注册 `@register_triton_op(module, symbol)`。
3. replacement 函数保持签名兼容。
4. 如果已有模块导入旧 symbol，依赖 `_replace_imported_bindings()` 修复。
5. 参考 `test_kernel_ops.py` 写机制测试。

### 场景 C：新增一种量化方法

处理方式：

1. 在上游 `sglang.srt.layers.quantization` 找到 config / method 类。
2. 在 `hooks/layers/quantization/<new_quant>.py` 写 Hook。
3. 在 `hooks/layers/quantization/__init__.py` 导入新文件。
4. 如果需要 bias、weight loader 或参数 dtype 修正，在 `hooks/layers/linear.py` 或新模块中用 AROUND Hook。
5. 写 mock layer 测试 `process_weights_after_loading` 和 `apply` 的 shape/dtype 行为。
6. 硬件上跑真实模型或单层算子一致性测试。

### 场景 D：新增模型

处理方式：

1. 判断是否真需要模型适配，而不是缺通用算子。
2. 在 `sglang_kunlun/models/<model_name>.py` 增加薄适配。
3. 复用上游模型类，避免复制大段代码。
4. 确认 `sglang_kunlun.models` 作为 external model package 生效。
5. 写 import / registration 测试。
6. 用小 config 或真实模型做硬件 E2E。

### 场景 E：新增 Attention 行为

处理方式：

1. 如果是新增 backend，在 `hooks/layers/attention/attention_registry.py` 或新 registry 模块注册。
2. 如果是已有 Kunlun backend 的新 metadata，放到 `kunlun_backend.py` 或拆分子模块。
3. 如果是 NSA/FLA 特定逻辑，放入 `attention/nsa/` 或 `attention/fla/`。
4. 如果平台需要选择该 backend，在 `platform/srt.py` 增加工厂方法或默认值。
5. 写 metadata 构造测试和最小 prefill/decode E2E。

## 运行与验证

Runbook `start.sh` 保存当前版本实际可用的完整启动命令。模型、TP 和服务参数名可能随版本变化，必须依据当前 `--help`。

## Benchmark

优先使用当前版本的 `python -m sglang.bench_serving`，输出 JSONL 到 `$RUN_DIR/benchmark/`。

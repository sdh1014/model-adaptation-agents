# vLLM-Kunlun 实施指南

本文件供 `/adapt-implement` 使用。所有目录和 API 必须以 `target_repo` 当前 commit 为准。

## 1. 先确定修改归属

vLLM-Kunlun 是 vLLM 的 Kunlun 平台扩展。一个模型缺口可能属于：

1. **上游 vLLM 模型层**：模型类、config、权重加载、模型注册；
2. **Kunlun 插件层**：平台注册、模型 override、P800 算子、Attention/KV Cache、分布式或编译路径；
3. **环境/版本层**：vLLM 与插件版本不匹配、当前 Python 未加载目标仓；
4. **模型事实层**：参考实现或 checkpoint 语义尚未确认。

只有前两类属于代码实现工作项。

## 2. 修改优先级

按以下顺序选择实现位置：

1. 复用当前 commit 已有的通用模型实现；
2. 使用插件公开注册或 override 机制；
3. 在 Kunlun 专用模型、layer 或 backend 中增加最小差异；
4. 只有存在直接证据表明插件无法表达时，才请求批准修改上游核心。

禁止为了一个小差异复制完整上游模型文件。

## 3. Capability 与候选位置

| Capability | 必查位置 | 实现重点 | 最小验收示例 |
|---|---|---|---|
| plugin/platform discovery | `pyproject.toml`、注册入口、platforms | entry point 能发现，当前 Python 加载目标仓 | import + entry point 枚举 |
| model registration | `register_model`、模型映射、`models/` | architecture 指向正确实现，无重复覆盖 | registry 查询或最小 import |
| config/model structure | Kunlun override 或上游模型目录 | forward 语义与模型分析一致 | 小 config 构造 + 单层 forward |
| weight loading | 模型 `load_weights`、loader helpers | 名称映射、packed 权重、TP/EP 分片 | synthetic names 或小 checkpoint 测试 |
| Attention / KV Cache | attention backend、`ops/`、`v1/` runner | prefill/decode、head layout、cache layout | 单层 prefill/decode 或 backend unit test |
| MoE | `ops/`、model override、distributed helpers | routing、shared expert、expert layout | 小 shape routing/fused MoE 测试 |
| quantization | `quantization/`、loader、kernel wrapper | 方法识别、scale/zero-point、fallback | 量化 config + 小权重加载 |
| distributed | `distributed/`、worker/model runner | TP/EP 分片和通信语义 | 小配置 TP/EP 测试 |
| compile/graph | `compilation/`、`v1/`、patches | graph capture 边界，不改变 eager 正确性 | eager 先通过，再单独 graph 测试 |
| service/parser | parser registration、entrypoint | 请求/输出协议与模型要求一致 | parser 或 API 单测 |

候选位置只是搜索边界，不是固定路径。

## 4. 实施规则

- 模型功能和 P800 kernel 分成独立工作项；
- 正确性路径先于 fused/graph 性能路径；
- fallback 必须显式记录，且可被 `/adapt-validate` 检测；
- 若当前工作项需要改变上游公共接口，生成 `decision_required`；
- 若当前 Python import 的 vLLM 或 vLLM-Kunlun 不来自目标代码，分类为环境 blocker；
- 插件 patch 对上游版本有依赖时，记录版本边界，不在实现阶段升级依赖。

## 5. 历史和 pitfalls

先提取当前失败签名，再读取 `pitfalls.md`。历史问题只能产生候选假设，不能直接作为根因。

## 6. 局部验收

每个工作项必须使用当前 commit 可执行的真实命令。最低要求：

- model registration：能查询到目标 architecture；
- config/model：能构造模型或执行最小 forward；
- weight loading：明确校验 missing/unexpected keys；
- Attention/MoE/算子：覆盖模型实际 dtype 和关键 shape；
- distributed：至少覆盖任务指定 TP/EP 组合的局部路径；
- service/parser：固定输入得到结构化输出。

仅 `py_compile` 不能标记工作项 passed。

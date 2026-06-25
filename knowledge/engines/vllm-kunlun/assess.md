---
scope: engine/vllm-kunlun
status: verified
last_reviewed: 2026-06-24
---

# vLLM-Kunlun 适配评估

## 1. 分层判断

vLLM-Kunlun 是 vLLM 的 Kunlun 硬件插件。评估时必须区分：

1. **上游 vLLM 模型层**：模型 registry、config、model implementation、weight loader；
2. **vLLM-Kunlun 插件层**：平台发现、worker/backend、custom ops、Attention/KV Cache、量化和 patches；
3. **本地运行时层**：PyTorch、插件、本地算子库和动态库。

“上游 vLLM 支持模型”不代表 P800 路径自动支持模型所需算子；反之，插件仓中找不到模型类也不代表模型未支持，因为模型实现可能位于上游 vLLM。

## 2. 仓库输入

推荐：

```yaml
target_repo: ../vLLM-Kunlun
upstream_repo: ../vllm
```

没有 `upstream_repo` 时，使用环境勘测记录的已安装 `vllm` 包位置，但降低静态评估置信度。

## 3. 静态检查顺序

### 3.1 版本与插件发现

检查：

- `pyproject.toml` / `setup.py` / requirements；
- Python entry points；
- `vllm` 与 `vllm-kunlun` 的版本关系；
- 平台插件是否可 import 和发现；
- 当前命令实际加载的源码路径；
- patch 是否与当前上游版本匹配。

### 3.2 上游模型支持

在上游仓或已安装包中搜索：

- `config.json` 的 `architectures` 值；
- `model_type`；
- model registry；
- 对应 model class；
- `load_weights`、`weight_loader` 和 packed module mapping；
- quantization 与 TP 支持；
- 相似模型实现和测试。

若模型可通过通用 Transformers backend 或 remote code 路径运行，必须记录为 `fallback` 或 `remote_code`，并说明限制。

### 3.3 Kunlun backend 支持

根据模型分析检查：

- platform class 和设备选择；
- Attention backend 与模型 attention 形态；
- KV Cache layout/dtype；
- RoPE、norm、sampling、MoE 等 custom ops；
- quantization backend；
- TP/EP/PP 通信；
- 图编译、custom op schema 和 fallback；
- 模型特有 patch 或分支；
- 相关 UT/集成测试。

### 3.4 运行时兼容

重点确认：

- Python 实际 import 的 `vllm` 与 `vllm_kunlun` 路径；
- 上游 vLLM 与插件版本是否匹配当前仓文档；
- 依赖的 Kunlun 算子包和动态库是否可加载；
- 同类算子包是否发生冲突；
- 平台插件是否真正被 vLLM 选中。

## 4. Baseline

优先一次性离线生成，不先启动 OpenAI 服务。最小目标：

```text
plugin import
→ platform selection
→ model architecture recognition
→ config creation
→ weight load
→ 1～4 token greedy generation
```

只有目标仓当前文档、任务 target.yaml 或已确认本地知识提供可信命令时执行。不要根据记忆拼接版本相关参数。

## 5. 常见失败阶段

| 阶段 | 先检查 |
|---|---|
| import | 包版本、动态库、patch、实际 import 路径 |
| platform selection | entry point、platform class、环境变量 |
| architecture recognition | 上游 registry、Transformers fallback |
| config | 自定义 config 和版本字段 |
| weight loading | 权重命名、packed mapping、量化格式 |
| runtime op | Kunlun custom op、Attention/KV Cache、MoE、dtype |
| distributed | TP/EP 切分、XCCL/通信和设备映射 |

## 6. 工作项归属

- 上游模型语义缺失：优先归入上游 vLLM 或明确的模型扩展层；
- Kunlun backend 缺失：归入 vLLM-Kunlun；
- 包/patch/ABI 问题：环境处置，不自动修改模型代码；
- 用户要求只修改插件仓时，必须在工作项中显式记录该边界。

## 7. 当前公开资料提示

官方仓库将 vLLM-Kunlun 定义为 Kunlun XPU 的硬件插件，并要求结合匹配的 vLLM 版本使用。支持列表会随版本变化，因此每次评估都应以目标 commit 为准，而不是把当前列表永久写入判断逻辑。

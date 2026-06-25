---
scope: engine/sglang-kunlun
status: verified
last_reviewed: 2026-06-24
---

# SGLang-Kunlun 适配评估

## 1. 分层判断

SGLang 目标可能是：

- 包含完整 SGLang 源码的 Kunlun fork；
- 基于上游 SGLang plugin system 的独立 Kunlun 扩展；
- 上游仓与本地 patch 的组合。

必须先根据当前 commit 判断实际形态，不预设仓库一定是 fork 或 plugin。

评估分为：

1. **SGLang 模型层**：模型 registry、config、model class、weight loader、输入处理；
2. **平台/后端层**：平台插件、Attention/KV Cache、custom ops、量化和并行；
3. **本地运行时层**：PyTorch、Kunlun 扩展、动态库和启动环境。

## 2. 仓库输入

完整 fork：

```yaml
target_repo: ../sglang-kunlun
upstream_repo: null
```

独立插件：

```yaml
target_repo: ../sglang-kunlun-plugin
upstream_repo: ../sglang
```

`target_repo` 必须是后续允许修改的仓库。

## 3. 静态检查顺序

### 3.1 仓库形态和版本

检查：

- package metadata 和 entry points；
- 是否包含完整 `sglang` Python 源码；
- 是否存在 platform/plugin 注册；
- fork 与 upstream 的 revision 关系；
- 当前 Python import 的 `sglang` 路径；
- Kunlun 扩展及本地算子依赖。

### 3.2 模型支持

搜索：

- `architectures` 或 model class 映射；
- model registry；
- 对应模型实现；
- config 字段兼容；
- `load_weights` 和权重映射；
- tokenizer / processor / multimodal input；
- 相似模型和测试。

### 3.3 Runtime 与后端

根据模型分析检查：

- server/model runner 的平台选择；
- Attention backend 和 Radix/KV cache 路径；
- RoPE、norm、MoE、sampling 等算子；
- 量化后端；
- TP/DP/EP/PP 支持；
- graph runner、custom op 或函数替换；
- 模型特有 server args、parser 和 feature gates。

### 3.4 插件系统

若当前 commit 支持插件系统，检查：

- 插件 entry point；
- 平台注册；
- device op / attention backend / KV cache / graph runner 的替换点；
- 插件是否在当前 Python 环境中被发现。

若当前 commit 不支持，不得根据较新上游文档假设其存在。

## 4. Baseline

优先一次性 Engine API 或最小 server launch，目标顺序：

```text
sglang import
→ Kunlun platform/backend selection
→ model architecture recognition
→ config 和 weight load
→ 1～4 token greedy generation
```

只有存在与当前 target commit 匹配的可信命令时执行。

## 5. 常见失败阶段

| 阶段 | 先检查 |
|---|---|
| import | 包路径、fork/plugin 冲突、动态库 |
| platform init | plugin entry point、platform 注册、设备接口 |
| model recognition | registry、architecture mapping |
| config/weight | 自定义 config、weight mapping、量化格式 |
| attention/KV | backend 选择、cache layout、RadixAttention 相关路径 |
| MoE/parallel | fused MoE、TP/EP/DP、通信 backend |
| service | server args、processor、chat/tool parser |

## 6. 工作项归属

- 模型层缺口：完整 fork 中修改 SGLang 模型层；插件模式下先判断扩展点是否足够；
- Kunlun 后端缺口：修改目标插件或 Kunlun backend；
- 上游版本差异：单独记录，不与模型语义修改合并；
- 环境和安装问题：不生成代码工作项。

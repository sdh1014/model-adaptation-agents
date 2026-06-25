# SGLang-Kunlun 实施指南

本文件供 `/adapt-implement` 使用。目标仓可能是 fork、in-tree backend 或插件形态，必须先以当前 commit 确认集成方式。

## 1. 先确定修改归属

缺口通常属于：

1. SGLang 模型/config/loader；
2. Kunlun 平台或 backend；
3. Attention、KV Cache、MoE、linear、norm 等 P800 layer；
4. scheduler、server 或协议层；
5. 环境和版本；
6. 模型事实。

只有前四类可形成实现工作项。

## 2. 修改优先级

1. 复用现有 SGLang 模型和 layer；
2. 使用当前 Kunlun 集成机制提供的注册、替换或 backend 扩展点；
3. 增加模型专用最小差异；
4. scheduler/cache 语义变化必须单独工作项；
5. 需要修改多个上游核心模块时生成开发者决策单。

避免在多个模块重复硬件判断。

## 3. Capability 与候选位置

| Capability | 必查位置 | 实现重点 | 最小验收示例 |
|---|---|---|---|
| model/config | `python/sglang/srt/models/`、`configs/` | architecture/config 可发现，模型类匹配 | config/model import test |
| weight loading | `model_loader/`、模型 loader | 名称、packed 权重、TP/EP 分片 | 小 checkpoint 或 synthetic mapping test |
| model execution | `model_executor/`、model runner | prefill/decode 输入输出契约 | 单步 forward |
| Attention / linear / norm | `layers/`、custom ops、backend | dtype、shape、layout、fallback | layer unit test |
| MoE | layer/custom op/distributed | routing、expert layout、shared expert | 小 shape MoE test |
| KV Cache | `mem_cache/`、attention backend | page/block/layout 和 scheduler 契约 | cache allocate/read/write test |
| TP/EP/PP | `distributed/`、model executor | 参数切分和通信 | 任务指定并行局部测试 |
| compile/graph | compilation、backend runner | eager 与 graph 语义一致 | eager 通过后 graph test |
| server/device | `server_args.py`、entrypoints、platform registry | 设备选择和启动参数 | 参数解析 + 最小启动 |
| multimodal/protocol | multimodal、processor、server | 输入结构和 processor | processor/API test |

路径必须通过当前目标仓扫描确认。

## 4. 实施规则

- 模型实现和硬件 kernel 分离；
- scheduler/cache 变化不得混入普通模型注册工作项；
- 先保证 eager/基础路径正确，再做 graph 或 fused 路径；
- 不在首次功能适配中进行性能专项改造；
- 当前 Python 未加载目标仓时分类为环境 blocker；
- 需要升级 SGLang、运行时或 Kunlun 依赖时生成 decision_required，而不是自动安装。

## 5. 历史和 pitfalls

先基于当前 stderr 提取失败签名，再读取 `pitfalls.md`。同一签名无新增证据重复 2 次时停止自动修改。

## 6. 局部验收

至少验证当前工作项直接目标：

- 注册/config：目标模型可被解析；
- loader：missing/unexpected key 符合预期；
- layer/算子：模型实际 dtype、shape 和 layout；
- cache/scheduler：状态转换和读写一致；
- distributed：任务指定并行组合；
- server/protocol：固定请求可解析并返回结构化结果。

仅 import 或语法通过不足以完成复杂工作项。

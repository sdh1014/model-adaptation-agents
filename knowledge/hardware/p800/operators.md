# P800 算子能力评估

## 核心原则

算子支持必须比较语义和约束，而不是只搜索同名函数。

## 每个算子需要记录

- capability：RMSNorm、RoPE、Attention、KV Cache、MoE、quant linear 等；
- 入口和调用路径；
- 输入/输出 shape；
- dtype；
- layout；
- alignment；
- batch、sequence、head dim、top-k 等边界；
- 是否支持动态 shape；
- 是否支持 graph/compile；
- fallback；
- 单元测试或模型覆盖证据。

## 证据优先级

1. 当前 target commit 的算子实现和测试；
2. 当前环境的最小算子测试；
3. 当前 commit 文档中的支持矩阵；
4. 知识库中版本匹配的已验证记录。

## 支持状态

- `supported`：模型语义和实际参数均在算子约束内；
- `partially_supported`：仅部分 dtype、shape、量化或运行模式支持；
- `unsupported`：明确缺少路径，且无正确 fallback；
- `unknown`：未找到足够证据。

## Fallback 判断

fallback 只有满足以下条件才能视为能力覆盖：

- 调用路径可达；
- 支持 P800 runtime；
- 语义与模型一致；
- dtype/layout 转换正确；
- 不依赖 CUDA-only 库；
- 至少存在局部运行证据或测试。

性能较慢不影响功能支持结论，但应记录为后续性能风险。

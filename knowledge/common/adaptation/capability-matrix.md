---
scope: common/adaptation
status: verified
---

# 模型适配能力矩阵

本文件定义 `adapt-assess` 的通用检查维度。实际评估只保留模型需要的能力。

## 一、模型理解层

| 能力 | 需要确认的内容 |
|---|---|
| architecture recognition | `architectures` / `model_type` 是否能映射到引擎实现 |
| config parsing | 自定义 config、嵌套 config、默认值和兼容字段 |
| model graph | block、norm、residual、LM head、特殊分支是否表达一致 |
| weight loading | 权重命名、分片、packed weight、权重转换和 tied weight |
| tokenizer/input | tokenizer、processor、chat template、multimodal 输入处理 |

## 二、运行语义层

| 能力 | 需要确认的内容 |
|---|---|
| attention | MHA/GQA/MQA/MLA、head dim、bias、mask、sliding window |
| position encoding | RoPE 类型、scaling、partial rotary、长上下文 |
| KV cache | cache layout、dtype、page/block、压缩或 latent cache |
| FFN/MoE | activation、gating、top-k、shared expert、grouped routing |
| sampling/output | logits processor、EOS、stop、特殊输出头 |

## 三、硬件后端层

| 能力 | 需要确认的内容 |
|---|---|
| platform selection | Kunlun 平台插件是否被发现并选中 |
| dtype | BF16/FP16/FP32 及模型所需低精度 |
| operators | norm、RoPE、attention、matmul、MoE、sampling 等关键算子 |
| compilation | eager、图编译、custom op、fallback 路径 |
| memory | 权重、workspace、KV cache 的分配与限制 |
| quantization | 量化格式、dequant、量化 GEMM/MoE 和权重布局 |

## 四、并行与调度层

| 能力 | 需要确认的内容 |
|---|---|
| tensor parallel | attention heads、KV heads、FFN、vocab、权重切分 |
| expert parallel | expert 分片、路由、all-to-all、shared expert |
| data parallel | 调度、DP attention、结果一致性 |
| pipeline parallel | 层切分、跨 stage 数据和模型支持 |
| scheduler | prefill/decode、chunked prefill、prefix cache、batching |

## 五、条件能力

仅在模型需要时检查：

- multimodal encoder / projector / processor；
- MTP、Eagle、speculative decoding；
- LoRA；
- embedding、reranker 或 reward head；
- tool-call parser、reasoning parser；
- PD disaggregation、分布式 KV cache；
- 自定义 remote code。

## 判定原则

- “存在同名文件”不等于能力受支持；
- “上游支持模型”不等于 Kunlun backend 支持全部算子；
- “能启动”不等于数值正确；
- “存在 fallback”必须记录限制；
- 环境失败与代码能力缺口必须分开。

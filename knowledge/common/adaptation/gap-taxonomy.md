# 适配缺口分类

本文件用于 `/adapt-assess` 统一分类，不用于直接决定实现方案。

## 分类原则

一个观察到的失败只能先归入“失败位置”，只有存在直接代码或运行证据后，才能归入“能力缺口”。

```text
环境或版本问题 ≠ 模型能力缺口
引擎未识别模型 ≠ P800 算子缺失
模型能够加载 ≠ 权重映射正确
能够返回文本 ≠ 适配正确
性能较低 ≠ 功能不支持
```

## 1. environment_or_version

范围：

- Python、PyTorch、引擎、插件、XPytorch 或运行时版本不匹配；
- 设备不可见；
- 动态库、环境变量、模型路径、存储、端口或共享内存问题；
- 当前 Python 加载的不是目标仓代码；
- dirty 工作树导致 baseline 不可归因。

处理：记录为环境 blocker，不创建目标代码工作项。

## 2. model_discovery

范围：

- `architectures`、`model_type` 或任务类型无法解析；
- 模型注册表、插件注册或映射缺失；
- 引擎选择了错误模型类；
- 通用 Transformers/fallback 路径不可达。

典型证据：注册表、entry point、模型解析日志。

## 3. configuration

范围：

- 自定义 config 类或字段未被识别；
- 默认值与模型真实行为不一致；
- 配置转换遗漏模型特有参数；
- remote code 成为不可接受的硬依赖。

## 4. model_structure

范围：

- block、norm、residual、activation 或输出头缺失；
- 模型类仅名称相似，计算语义不同；
- MTP、Eagle、reasoning head 等附加模块缺失。

## 5. weight_loading

范围：

- 权重命名映射缺失；
- packed QKV、gate/up、expert 权重处理错误；
- tied weight、共享权重、额外权重处理错误；
- TP/EP 分片或量化 scale 映射错误。

## 6. attention

范围：

- GQA/MQA/MLA、QK norm、scale、mask 或投影组织不支持；
- prefill/decode 使用的 backend 与模型语义不匹配；
- sliding、sparse、linear attention 等模式缺失。

## 7. position_encoding

范围：

- RoPE theta、scaling、partial rotary 或 position id 行为不支持；
- multimodal position encoding 或多轴 RoPE 缺失。

## 8. kv_cache

范围：

- KV layout、dtype、block/page 或压缩方式不支持；
- MLA latent cache、cross-attention cache 或特殊缓存状态缺失；
- prefix cache、chunked prefill 或 graph 模式与模型冲突。

## 9. moe

范围：

- router、top-k、grouped routing、shared expert 语义不支持；
- fused MoE 参数约束不匹配；
- expert 权重、TP/EP 切分或通信缺失。

## 10. quantization

范围：

- 方法未识别；
- 权重布局、scale、zero-point 或 group size 不匹配；
- 某个模型模块缺少量化 kernel 或正确 fallback。

## 11. multimodal

范围：

- encoder、projector、processor 或模态输入协议缺失；
- 多模态 position、cache、batch 或占位 token 行为不支持。

## 12. parallelism

范围：

- TP、PP、DP、EP 参数切分或通信不支持；
- head/expert 数量与并行度约束不满足；
- 单机/多机进程、设备映射或通信 backend 不匹配。

## 13. hardware_operator

范围：

- P800 backend 缺少必要算子；
- 存在算子但 shape、dtype、对齐或语义约束不覆盖模型；
- generic fallback 不可达或不能保持正确性。

## 14. serving_behavior

范围：

- tokenizer、chat template、EOS/stop 行为无法表达；
- reasoning parser、tool parser、embedding/reward 输出或多模态 API 缺失；
- 服务入口不能支持模型所需请求格式。

## 15. performance_only

范围：功能和正确性已满足，但专用融合算子、graph、调度或缓存策略尚未优化。

处理：不要在首次 `/adapt-assess` 中生成实现工作项；正确性通过后进入性能评估或后续优化阶段。

# 适配评估知识

## 能力维度

```text
model registration / config / model graph
weight loading / attention / position encoding / KV cache
FFN / MoE / quantization / multimodal
TP / EP / PP / platform / operator / serving behavior
```

## 缺口分类

```text
model_fact_gap          返回 model-analyze
engine_model_gap        可生成模型层工作项
hardware_backend_gap    可生成 P800 backend 工作项
packaging_version_gap   环境或版本处理
runtime_environment_gap 环境处理
configuration_gap       配置处理
validation_gap          留给 validate
unknown_gap             先补最小证据
```

只有有代码或运行证据的模型层、backend 层缺口默认生成代码工作项。

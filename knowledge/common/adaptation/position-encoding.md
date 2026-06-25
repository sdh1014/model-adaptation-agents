# 位置编码评估

## 必查模型事实

- 类型：RoPE、ALiBi、learned position、multimodal RoPE 等；
- theta/base；
- rotary dimension 或 partial rotary factor；
- scaling 类型和参数；
- original max position；
- position id 生成方式；
- 多轴或交错布局；
- 长上下文外推行为。

## 引擎检查

- config 字段是否完整传递；
- rotary cache 创建逻辑；
- prefill/decode position id 一致性；
- P800 fused rope 是否支持实际布局和 dtype；
- fallback 是否覆盖 scaling 和 partial rotary；
- graph/compile 是否固定了错误长度。

## 常见误判

- 仅支持标准 RoPE，却将自定义 scaling 标为 supported；
- 参数名相同，但执行顺序或维度布局不同；
- 短序列可运行，长上下文 position 行为错误。

# 服务行为评估

## 检查项

- tokenizer 和 processor 加载；
- chat template；
- BOS/EOS/PAD 和 stop token；
- generation config；
- reasoning parser；
- tool/function parser；
- embedding、reward、classification 输出；
- multimodal 请求格式；
- OpenAI-compatible API 能否表达模型能力。

## 边界

本阶段只确认入口和实现存在，不通过“返回了文本”判断行为正确。最终 EOS、tool call、reasoning 和多模态协议在 `/adapt-validate` 中验收。

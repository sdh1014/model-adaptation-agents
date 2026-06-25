# 多模态能力评估

## 模型事实

- 输入模态；
- encoder、projector 和语言模型组合；
- processor 和 placeholder token；
- 图像/音频 patch、position 和 mask；
- 多输入 batch 规则；
- encoder cache；
- 权重组织和独立 config。

## 引擎检查

- processor 注册；
- multimodal registry；
- encoder 模型实现；
- projector 和 embedding merge；
- scheduler 对多模态 token 的预算；
- P800 对 vision/audio 算子的支持；
- API 请求格式和数据预处理。

## 最小验证

静态评估只判断路径是否存在；最终支持需要固定多模态样例完成输出或 logits parity。

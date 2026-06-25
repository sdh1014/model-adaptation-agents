# Tokenizer 分析

检查：

- tokenizer 类型与词表文件；
- BOS、EOS、PAD、UNK；
- 多 EOS 或 stop token；
- chat template；
- added tokens；
- generation config 默认值。

后续生成一致性验证必须固定 prompt 渲染方式和特殊 token 行为。

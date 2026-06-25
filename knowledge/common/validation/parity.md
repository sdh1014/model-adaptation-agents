# Reference Parity

## 必须固定

- checkpoint 和 revision；
- tokenizer、special tokens 和 chat template；
- 原始文本及其 token ids；
- padding、truncation 和 position ids；
- temperature、top-p、top-k、seed、max tokens；
- dtype、量化和并行配置；
- EOS、stop 和 ignore-eos 行为。

## 比较优先级

1. prefill 最后位置 logits；
2. 选定层或关键模块输出；
3. 首个 decode logits/token；
4. 短 greedy token 序列；
5. 完整文本。

文本是最弱 oracle；token 或 logits 更能区分 tokenizer、sampling 与模型计算问题。

## 数值阈值

阈值应同时说明：

- absolute tolerance；
- relative tolerance；
- 比较范围，例如 top-k logits、全部 logits 或选定 token；
- 聚合指标，例如 max error、mean error、cosine similarity；
- 不同 dtype/量化的适用边界。

不得在看到结果后临时放宽阈值。

## 通用比较工具

- JSON 响应：`scripts/validation/compare_json.py`；
- logits/hidden states 数组：`scripts/validation/compare_arrays.py`，支持 JSON、NPY 和 NPZ；
- 工具只负责比较，参考数据的可信来源仍必须写入 `validation.md`。

示例：

```bash
python "$CONTROL_ROOT/scripts/validation/compare_arrays.py" \
  --reference /reference/prefill_logits.npy \
  --actual "$RUN_DIR/validation/prefill_logits.npy" \
  --abs-tol 1e-3 --rel-tol 1e-3 \
  --output "$RUN_DIR/validation/prefill-logits-comparison.json"
```

# Result Contract

`result.json` 是 attempt 的机器可读结果。

失败结果只写事实：

```json
{
  "status": "FAIL",
  "failed_check": "model_load",
  "error_signature": "unsupported-attention-backend",
  "evidence": ["logs/stderr.log:183"]
}
```

禁止在失败结果里写：

- 根因判断
- 原因分析
- 下一步建议
- 未验证结论


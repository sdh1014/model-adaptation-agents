# adapt-validate 指南

## 前置

默认要求 assessment 通过，且实现为 passed/not_required 或评估已确认支持。

## 验证计划

从模型分析和实现工作项生成 required case：加载、确定性生成、stop/EOS、batch、TP、长上下文和模型特有能力。

执行后读取：

```text
<run>/stage-result.json
<run>/runtime/logs/
<run>/runtime/validation/
```

`checks/validate.sh` 可直接粘贴多条断言命令。可选生成 `$RUN_DIR/validation/result.json` 提供逐 case 状态。

状态：passed / failed / partial / blocked。失败分类后只给一个下一步：model-analyze、adapt-assess、adapt-implement 或编辑验证脚本。

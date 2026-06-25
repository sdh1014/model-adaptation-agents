# 示例

`model.yaml` 放到：

```text
tasks/demo/model.yaml
```

目标 YAML 重命名为 `target.yaml` 后放到：

```text
tasks/demo/targets/vllm-kunlun/target.yaml
tasks/demo/targets/sglang-kunlun/target.yaml
```

随后执行：

```bash
python scripts/model_runtime.py init demo/vllm-kunlun
```

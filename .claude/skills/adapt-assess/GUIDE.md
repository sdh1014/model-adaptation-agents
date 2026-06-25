# adapt-assess 指南

## 顺序

1. 检查 `model-analysis.md` 为可用状态；
2. 创建/更新 `target.yaml`；
3. 初始化 Runbook：

```bash
python scripts/model_runtime.py init <model>/<target>
```

已有 Runbook 时只补缺失文件，不覆盖开发者命令。

4. 收集环境：

```bash
python scripts/assess.py env --engine <engine> --model-path <path> \
  --target-repo <repo> --upstream-repo <repo> --output <run>/environment.json
```

5. 静态扫描：

```bash
python scripts/assess.py repo --engine <engine> --target-repo <repo> \
  --upstream-repo <repo> --architecture <class> --output <run>/repositories.json
```

6. baseline=auto 且 Runbook 已配置时执行最小 smoke；
7. 建立能力矩阵，区分模型层、P800 backend、环境、版本和配置问题；
8. 只将确认的模型层/backend 缺口转为代码工作项。

## 结果

```text
already_supported
adaptation_required
blocked
```

模型事实不足时返回 `/model-analyze --update`。环境或版本不足时记录解除条件，不生成代码工作项。

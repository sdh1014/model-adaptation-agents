# adapt-assess 指南

1. 检查 `model-analysis.md` 为可用状态；
2. 创建/更新 `target.yaml`；
3. 初始化 Runbook：

```bash
python scripts/model_runtime.py init <model>/<target>
```

已有 Runbook 时只补缺失文件，不覆盖开发者命令。

4. 创建目标级运行目录：

```bash
RUN_DIR="$(python scripts/paths.py create-run --target <model>/<target> --stage assess)"
```

5. 收集环境：

```bash
python scripts/assess.py env --engine <engine> --model-path <path> \
  --target-repo <repo> --upstream-repo <repo> --output "$RUN_DIR/environment.json"
```

6. 静态扫描：

```bash
python scripts/assess.py repo --engine <engine> --target-repo <repo> \
  --upstream-repo <repo> --architecture <class> --output "$RUN_DIR/repositories.json"
```

7. baseline=auto 且 Runbook 已配置时执行最小 smoke；
8. 建立能力矩阵，区分模型层、P800 backend、环境、版本和配置问题；
9. 只将确认的模型层/backend 缺口转为代码工作项。

结果：`already_supported / adaptation_required / blocked`。

## 结束时必须告诉用户

Runbook 已创建后，明确列出首次运行只需编辑：

```text
runbook/env.sh
runbook/start.sh
runbook/checks/smoke.sh
```

敏感变量使用 `runbook/env.local.sh`。不要让用户自行浏览 templates 或 scripts 寻找下一步。

# model-run 指南

参数映射：

```text
--init                    model_runtime.py init
无参数                    run --check smoke
--check NAME              run --check NAME
--serve                   serve
--against-running NAME    exec --check NAME
--status                  status
--stop                    stop
```

示例：

```bash
python scripts/model_runtime.py run minimax-m3/vllm-kunlun --check smoke
```

Runbook 规则：环境放 `env.sh`，完整前台启动命令放 `start.sh`，单次探测放 `ready.sh`，测试命令放 `checks/*.sh`。禁止 `nohup` 和后台 `&`。

失败时依次查看 check stderr/stdout、server stderr/stdout、ready.log。只报告失败阶段和证据，不在本 Skill 修复代码。

## 未配置时

遇到 `MODEL_RUN_NOT_CONFIGURED` 时，不做根因分析。直接指出需要编辑的具体文件：

```text
runbook/start.sh
runbook/checks/<name>.sh
```

环境变量缺失时指出 `runbook/env.sh`；敏感值使用 `runbook/env.local.sh`。

# Model Adaptation Repository Rules

1. 本仓库管理 Skill、知识、任务和实验证据；目标框架代码位于外部仓库。
2. 当前可用 Skill 只有 `model-analyze`、`adapt-assess`、`adapt-implement`、`adapt-validate`、`adapt-benchmark` 和 `model-run`；未实现的阶段不得假装执行成功。
3. `model-analyze` 只分析模型事实，不分析具体引擎和 P800，不修改目标仓库。
4. `adapt-assess` 只生成目标引擎适配评估、缺口和实施工作项，不修改目标仓库。
5. 只有 `adapt-implement` 可以修改适配目标源码。
6. 模型级结果保存在 `tasks/<model>/`；引擎级结果保存在 `tasks/<model>/targets/<engine>/`。
7. `knowledge/` 只保存经过确认、可跨任务复用的知识；临时推断不得写入其中。
8. `runs/` 只保存执行事实和原始证据，不写未经验证的根因结论。
9. 每个 Skill 完成当前阶段后立即停止，不自动进入下一阶段。
10. 模型分析可增量修订；修订必须记录证据、变更事实和受影响能力。
11. 脚本失败时必须返回非零退出码，禁止以占位结果宣布 PASS。
12. `adapt-implement` 每次调用只处理一个工作项，并且只修改一个已声明仓库。
13. 工作项可编辑仓库必须是 `target.yaml` 中的 `target_repo` 或 `upstream_repo`。
14. 所有实现与验收命令必须通过 `scripts/run_bash.py` 或 `scripts/implementation/check_implementation.sh` 保存证据。
15. 未知 dirty 工作区、仓库漂移、环境漂移或范围扩大时停止修改。
16. 不自动安装依赖，不自动 commit、push、reset、clean、merge 或 rebase。
17. 达到停止条件后更新 `implementation.md`，并创建 `blockers/<WI-ID>.md`。
18. `model-run` 是人工运行入口，不是线性适配阶段。
19. 其他 Skill 需要运行模型时，调用 `python scripts/model_runtime.py`，不要依赖 Skill 之间互相调用。
20. `model-run` 不修改目标仓库，不判断正确性，不做性能测试。
21. 每个模型目标的唯一运行定义位于 `tasks/<model>/targets/<target>/runbook/`。
22. 环境变量写入 `env.sh`，完整服务命令写入 `start.sh`，检查命令写入 `checks/*.sh`。
23. 禁止在 Skill、`target.yaml`、阶段报告或多个脚本中复制同一套启动命令。
24. Runbook 属于开发者配置，Skill 不得在失败后自动改写它。
25. `start.sh` 必须保持前台运行，禁止 `nohup`、后台 `&` 和 daemonize。
26. 默认运行必须清理服务进程；只有显式 `--serve` 可以保留服务。
27. `adapt-validate` 和 `adapt-benchmark` 禁止修改目标仓库代码、模型权重或 Runbook。
28. 正确性命令写入 `runbook/checks/validate.sh`；性能命令写入 `runbook/checks/benchmark.sh`。
29. Smoke 或脚本退出 0 不等于正确性通过；正式通过要求结构化 required case 覆盖完整。
30. Benchmark 的 `status` 表示执行有效性，`target_met` 表示性能目标，二者不得混淆。
31. 正确性未通过时，默认禁止正式 benchmark。
32. 验证失败只分类并路由，不在验证阶段修复代码；性能未达标不在 benchmark 阶段自动调参。
33. `runs/` 保存原始事实；`validation.md` 和 `benchmark.md` 只保存最新有效结论。

# Repository Rules

1. 模型适配主流程只有六个公开 Skill：`model-analyze`、`adapt-assess`、`adapt-implement`、`model-run`、`adapt-validate`、`adapt-benchmark`。
2. `model-analyze` 只分析模型事实，不判断具体引擎或 P800 支持。
3. `adapt-assess` 只评估和拆分工作项，不修改目标代码。
4. 只有 `adapt-implement` 可以修改 `target.yaml` 明确声明的目标仓库。
5. `adapt-implement` 每次只处理一个工作项和一个可证伪假设。
6. `model-run` 只管理服务生命周期和命名检查，不判断正确性或性能目标。
7. `adapt-validate` 与 `adapt-benchmark` 不修改代码、权重或 Runbook。
8. `model-inference-release-docs` 是独立模型推理准出文档生成 Skill，不读取或修改模型适配主流程文件。
9. 每个目标的唯一运行定义位于 `tasks/<model>/targets/<target>/runbook/`。
10. 环境变量只写 `env.sh`，启动命令只写 `start.sh`，检查命令只写 `checks/*.sh`。
11. 禁止在 Skill、报告和多个脚本中复制同一套启动参数。
12. `knowledge/` 只保存经过确认、可复用的知识；临时推断留在任务报告。
13. `runs/` 只保存执行事实；模型级使用 `runs/<model>/`，目标级使用 `runs/<model>--<target>/`，阶段写入 run 目录名。
14. 模型事实缺失时返回 `/model-analyze --update`，不在下游阶段猜测。
15. 环境或目标版本失效时返回 `/adapt-assess`，不通过代码绕开。
16. 不自动安装依赖，不自动 commit、push、merge、rebase、reset、clean 或 stash。
17. 默认模型运行必须清理进程；只有显式 `--serve` 可以保留服务。
18. `start.sh` 必须保持前台运行，禁止 `nohup`、后台 `&` 和 daemonize。
19. Validation required case 未覆盖时不得标记 `passed`。
20. Benchmark 执行状态与性能目标是否达到必须分开记录。
21. 每个 Skill 完成当前阶段后立即停止，不自动进入下一阶段。

# adapt-implement 指南

## 选择工作项

优先：`in_progress` → `needs_recheck` → 首个依赖已通过的 `pending`。`--item` 可指定。

## 执行循环

1. 校验 model-analysis、assessment、目标仓和 baseline revision；
2. 汇总历史：

```bash
python scripts/implement.py history --runs-root runs/<model>/<target> \
  --item WI-001 --output <run>/history.json
```

3. 写直接观察、稳定签名、唯一假设和否证条件；
4. 创建 run 与仓库快照；
5. 只修改允许范围；
6. 执行 scope、diff、py_compile 和工作项最小验收：

```bash
python scripts/implement.py check --target-repo <repo> --base-ref <ref> \
  --run-dir <attempt>/verification --allow 'path/**' -- <command>
```

7. 记录 passed/rejected/blocked/inconclusive；
8. 更新 `implementation.md` 后停止。

## 阻塞

以下情况立即停止：模型事实缺失、assessment 失效、环境/仓库漂移、需要第二 capability、跨仓/升级依赖/新增 kernel、三个假设均被否定。

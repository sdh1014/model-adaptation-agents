# Step-3.7 P800 BUG Ledger

记录 Contract revision 10 及之后在真实 P800 运行中遇到的 BUG、最多 3 次修复尝试和
最终解决方法。一次动作的原始输出放在 Run Evidence；本文件用稳定 `bug_id` 连接完整
历史。SOURCE 静态结论和旧 replay 缺陷不算本轮自动修复证据。

当前尚未执行 revision 10 的 P800 运行：

- `auto_repair_status`: `NOT_EXERCISED`
- `active_bug`: `null`
- `active_attempt`: `0`
- `max_attempts_per_bug`: `3`

## Summary

| bug_id | status | phase | category | attempts | first_seen | resolution | closure_evidence |
|---|---|---|---|---:|---|---|---|

## Record template

每个真实 BUG 复制下面模板并追加在文件末尾。已有 attempt 不覆盖、不删除。

```markdown
## BUG-<PHASE>-<NNN>: <短标题>

- status: OPEN | RESOLVED | BLOCKED
- auto_repair_status: ACTIVE | PASS | BLOCKED
- phase:
- category:
- attempt_count: 0
- max_attempts: 3
- first_seen:
- last_updated:
- contract_revision:
- sglang_revision:
- sglang_kunlun_revision:
- checkpoint:
- first_evidence:

### Symptom and reproduction

- expected:
- actual:
- exact_command:
- return_code:
- traceback_or_mismatch:
- input_shape_dtype_layout_stride:
- environment:
- production_source_anchor:
- worktree_diff_before_repair:

### Attempts

| attempt_id | run_evidence | hypothesis | supporting_evidence | changed_files | focused_test | original_path_rerun | outcome |
|---:|---|---|---|---|---|---|---|

### Resolution or blocker

- confirmed_root_cause:
- solution:
- why_it_works:
- focused_regression:
- original_p800_path_rerun:
- fixed_requests_or_queue_cases:
- remaining_failure:
- required_next_capability:
- residual_risk:
- closure_evidence:
```

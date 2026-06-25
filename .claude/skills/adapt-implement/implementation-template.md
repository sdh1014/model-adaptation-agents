---
status: pending                 # pending | in_progress | blocked | decision_required | passed | not_required
assessment_revision: <revision>
model_analysis_revision: <revision>
base_revision: <assessment-target-revision>
current_item: null
latest_run: null
last_observed_head: null
last_observed_patch_sha256: null
updated_at: <ISO-8601>
---

# 适配实现：<model-id> / <target-id>

## 1. 当前状态

- 总体状态：
- 当前工作项：
- 最新 run：
- 当前 blocker：
- 待开发者决策：
- 下一条命令：

## 2. 工作项总览

| 顺序 | ID | Capability | 可编辑仓库 | 状态 | 依赖 | Attempts | 最新签名 | 最新证据 |
|---|---|---|---|---|---|---|---|---|

工作项状态：

- `pending`
- `in_progress`
- `blocked`
- `decision_required`
- `needs_recheck`
- `passed`
- `not_applicable`

## 3. 工作项详情

### <WI-ID>：<名称>

- Capability：
- 可编辑仓库：`target_repo` / `upstream_repo`
- 状态：
- 依赖：
- Attempts：0
- 最新 run：
- 最新失败签名：

#### 目标


#### 直接证据

- 模型要求：
- 目标仓行为：
- P800 后端行为：
- Baseline：

#### 修改范围

- 仓库：
- 允许：
- 禁止：

#### 最小验收

```bash
<command>
```

通过标准：

-

#### 当前 blocker

- 分类：
- 解除条件：
- 证据：

#### 实现历史

| 时间 | Run | 假设 | 结果 | 失败签名 | 证据 |
|---|---|---|---|---|---|

---

## 4. 全局 Blockers

| ID | 分类 | 影响工作项 | 状态 | 解除条件 | 证据 |
|---|---|---|---|---|---|

## 5. 开发者决策

| ID | 工作项 | 状态 | 推荐决定 | 文件 |
|---|---|---|---|---|

## 6. 仓库状态记录

- 当前记录对应仓库：
- Base revision：
- Last observed HEAD：
- Last observed patch SHA256：
- 已知累计修改：

## 7. 变更记录

| 时间 | Work Item | 状态变化 | Run | 说明 |
|---|---|---|---|---|

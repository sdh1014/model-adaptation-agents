# 目标仓库安全规则

## 推荐形态

每个模型/目标使用独立 clone 和独立分支。控制仓库与目标仓库保持 sibling 关系，不把目标仓嵌入控制仓 Git 工作树。

## 实现前记录

- repo 绝对路径；
- remote；
- branch；
- HEAD；
- dirty 状态；
- 累计 patch hash；
- untracked 文件。

## 禁止的自动操作

- `git reset --hard`
- `git clean -fd`
- 强制 checkout 覆盖文件
- rebase、merge、push
- 删除未知 untracked 文件
- 自动 commit

这些动作可能消除开发者或其他会话的工作，必须由开发者明确执行。

## Dirty 工作区

首次实现时未知 dirty 工作区必须停止。

后续累计修改允许存在，但当前 patch hash 应与最后一次记录一致。若不一致，先确认变化来源。

## Patch 记录

每个 run 保存相对 assessment base revision 的累计 patch。累计 patch 便于审阅整体适配；每个 attempt 另外记录本轮修改文件和假设。

## 上游边界

优先在目标插件或 fork 的既有扩展点实现。需要修改另一个仓库或上游核心时，必须产生单独工作项或开发者决定。

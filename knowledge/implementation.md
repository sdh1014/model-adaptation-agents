# 实现阶段知识

## 标准循环

```text
直接观察 → 稳定错误签名 → 单一假设 → 最小修改 → 最小判别测试 → 结论
```

每个工作项必须有单一 capability、依赖、允许范围、non-goals 和独立验收。

## 验证梯度

```text
static → unit → component → smoke
```

当前层失败即停止。完整正确性留给 validate，性能留给 benchmark。

## 停止条件

- 模型事实缺失；
- assessment 失效；
- 环境或仓库漂移；
- 必须处理第二 capability；
- 需要跨仓、升级依赖或新增 kernel；
- 三个合理假设均被否定；
- 无法设计可证伪的新测试。

达到停止条件时生成 blocker，不继续扩大修改。

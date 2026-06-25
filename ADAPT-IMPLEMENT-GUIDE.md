# adapt-implement 使用说明

## 作用

`/adapt-implement` 每次只实现 `implementation.md` 中一个工作项。

它会：

```text
读取评估与工作项
→ 复核仓库和执行环境是否漂移
→ 汇总此前命令、日志、失败签名和假设
→ 加载对应知识
→ 验证一个新假设
→ 保存 patch 与局部验收
→ 更新工作项状态
```

它不会自动处理下一个工作项，也不会自动安装依赖、提交 Git 或进入完整验证。

## 前置阶段

```text
/model-analyze minimax-m3 --model-path /models/MiniMax-M3

/adapt-assess minimax-m3/vllm-kunlun \
  --target-repo /workspace/targets/minimax-m3-vllm-kunlun/engine \
  --upstream-repo /workspace/targets/minimax-m3-vllm-kunlun/upstream
```

评估产生的每个工作项必须指定唯一可编辑仓库：`target_repo` 或 `upstream_repo`。

## 基本调用

处理第一个依赖已满足的工作项：

```text
/adapt-implement minimax-m3/vllm-kunlun
```

指定工作项：

```text
/adapt-implement minimax-m3/vllm-kunlun --item WI-003
```

默认一次调用最多验证 3 个不同假设。减少本次上限：

```text
/adapt-implement minimax-m3/vllm-kunlun \
  --item WI-003 \
  --max-attempts 1
```

## 历史证据

开始修改前，Skill 会读取当前工作项此前的：

- `attempt.md`；
- 验收命令与工作目录；
- stdout、stderr 和退出码；
- 失败签名；
- 已尝试 patch 和修改范围；
- `outcome.json` 中的 blocker 与下一动作。

没有新增证据时，不重复相同命令或改法。

## 知识读取

始终读取：

```text
knowledge/common/implementation/
knowledge/common/adaptation/work-item-rules.md
knowledge/engines/<engine>/implement.md
```

再根据工作项读取 Attention、权重加载、KV Cache、MoE、量化、并行等对应文档。

只有获得当前失败签名后才读取 `pitfalls.md`，避免由历史经验替代当前证据。

## 环境处理

`adapt-implement` 不重新做完整环境勘测，只检查：

- Python 可执行文件；
- 引擎与插件 import 来源；
- 关键包版本；
- 可编辑仓库 HEAD、branch 和 patch；
- 局部测试需要 P800 时的设备可见性。

环境相对 assessment 发生实质变化时停止修改，并返回：

```text
/adapt-assess minimax-m3/vllm-kunlun
```

## 阻塞处理

### 模型事实缺失

```text
/model-analyze minimax-m3 --update "确认 grouped routing 的归一化顺序"
```

### 工作项或引擎评估缺失

```text
/adapt-assess minimax-m3/vllm-kunlun --focus moe-routing
```

### 反复失败、范围扩大或存在取舍

生成：

```text
tasks/minimax-m3/targets/vllm-kunlun/blockers/WI-003.md
```

文件会包含：稳定失败、历史尝试、已否定假设、当前 patch、未知项、推荐决定及最多 2 个替代决定。

开发者决定后恢复：

```text
/adapt-implement minimax-m3/vllm-kunlun \
  --item WI-003 \
  --decision "允许修改 upstream_repo 中指定模型目录，不改变公共 API"
```

## 停止条件

出现任一条件即停止自动修改：

- 同一失败签名无新增证据重复 2 次；
- 已验证 3 个不同假设仍未通过；
- 无法提出可证伪的新假设；
- 需要修改第二个仓库或另一个 capability；
- 需要升级依赖、修改公共接口或接受功能限制；
- 模型事实、测试 oracle 或环境指纹不足；
- 仓库出现来源不明的 HEAD/patch 漂移。

## Run 结构

```text
runs/<model>/<target>/<timestamp>-implement-<item>/
├── metadata.json
├── work-item.md
├── execution-context.json
├── history.json
├── history-summary.md
├── repo-before.json
├── patch-before.diff
├── attempts/
│   ├── 01/
│   │   ├── attempt.md
│   │   ├── repo-before.json
│   │   ├── verification/
│   │   │   ├── scope-before.json
│   │   │   ├── command/
│   │   │   │   ├── command.json
│   │   │   │   ├── stdout.log
│   │   │   │   ├── stderr.log
│   │   │   │   └── result.json
│   │   │   ├── scope-after.json
│   │   │   └── summary.json
│   │   └── failure-signature.json
│   └── ...
├── repo-after.json
├── patch-after.diff
└── outcome.json
```

Docker 和仓库位置见 `DOCKER-WORKSPACE.md`。

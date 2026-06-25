# 适配评估要求

## 1. 阶段目标

本阶段回答以下问题：

1. 当前引擎是否能识别并执行该模型所需的计算语义；
2. P800 后端是否覆盖模型所需算子、数据类型、KV Cache、并行和量化路径；
3. 当前运行环境是否具备进行 baseline 的条件；
4. 当前组合是否已经可运行；若不可运行，最小适配缺口是什么；
5. 哪些问题属于代码适配，哪些属于环境、版本或配置问题。

本阶段不负责修改代码。

## 2. 输入有效性

### 2.1 模型分析

必须读取 `model-analysis.md` 的 frontmatter 和正文，并确认：

- `status: passed`；
- 存在 `revision`；
- 模型架构类、核心计算路径、权重组织和关键特有能力有证据；
- 不存在会阻塞当前目标判断的模型未知项。

模型事实缺失时不得自行补全，必须输出增量分析命令。

### 2.2 目标定义

`target.yaml` 至少包含：

```yaml
engine: vllm-kunlun | sglang-kunlun
hardware: p800
target_repo: <path>
upstream_repo: <path-or-null>
runtime:
  python: <python-executable>
  tensor_parallel_size: <int>
assessment:
  baseline: auto | required | skip
```

含义：

- `target_repo`：目标插件、fork 或后端仓库；
- `upstream_repo`：运行时依赖的完整上游引擎源码，可为空；
- 对插件式引擎，模型实现可能位于上游仓库。评估必须为每个工作项明确指定 `target_repo` 或 `upstream_repo` 作为唯一可编辑仓库；不得让一个工作项同时修改两个仓库。

首次执行可由命令参数创建该文件；已有文件存在时，仅更新用户明确提供的字段。

## 3. 环境勘测边界

`adapt-assess` 必须包含轻量环境勘测，因为版本错配、插件未加载或设备不可见会伪装成代码缺口。

### 3.1 本阶段必须采集

- OS、内核、CPU 架构和 Python 可执行文件；
- 模型路径、目标仓库和上游仓库是否存在、可读；
- 目标仓库 commit、分支和工作区状态；
- `torch`、引擎包、Kunlun 插件及关键运行时包的版本和安装位置；
- 引擎、插件和 `torch` 的隔离 import 结果；
- `torch.cuda` 与 `torch.xpu` 两种可能的设备接口状态；
- 可见设备数量、相关设备节点和设备工具是否存在；
- 模型所在文件系统和 `/dev/shm` 的可用空间；
- 与目标仓文档或 requirements 对比后的明显版本不一致。

### 3.2 本阶段不做

- 长时间稳定性测试；
- 时钟、功耗和温度调优；
- RDMA、跨节点带宽和完整拓扑压测；
- 正式显存峰值、吞吐或延迟测量；
- 修改系统配置或安装依赖。

这些内容属于 `model-run` 或 `adapt-benchmark`。

### 3.3 环境状态

只使用：

- `ready`：具备 baseline 所需条件；
- `degraded`：存在警告，但仍可尝试 baseline；
- `unavailable`：缺少必要设备、包、路径或 import 失败；
- `unknown`：无法可靠判断。

环境不可用不等于存在代码缺口。

## 4. 证据等级

所有结论必须标注证据，优先级如下：

1. `runtime`：本次 baseline、import 或设备检查的实际结果；
2. `code`：目标 commit 中的注册、实现、backend 或测试代码；
3. `test`：目标仓已有单元测试、集成测试或模型测试；
4. `documentation`：与目标 commit 对应的官方文档；
5. `inference`：基于相似实现的推断。

`inference` 不能单独支撑 `supported` 或已确认根因。

每项关键证据应包含：

- 文件或产物路径；
- 字段、符号、行号或日志片段位置；
- 对应 commit 或环境快照。

## 5. 能力矩阵

始终检查：

- 架构识别与模型注册；
- 配置解析；
- 模型计算图；
- checkpoint 和权重加载；
- Attention、位置编码和 mask；
- KV Cache；
- FFN 或 MoE；
- dtype；
- tensor parallel；
- 目标平台选择与插件加载；
- 模型启动和最小生成路径。

按模型特征条件检查：

- 量化；
- expert parallel / data parallel / pipeline parallel；
- multimodal encoder、projector 和输入处理；
- MTP、speculative decoding；
- LoRA、embedding、reranker；
- 自定义算子、custom op 或图编译；
- tool calling、特殊 chat template 等服务层行为。

不涉及的能力标记为 `not_applicable`，不得生成空工作项。

## 6. 能力状态与执行路径

能力状态只使用：

- `supported`
- `partially_supported`
- `unsupported`
- `unknown`
- `not_applicable`

同时记录实际执行路径：

- `native`：引擎原生模型实现；
- `upstream`：由完整上游引擎实现；
- `plugin`：由 Kunlun 插件实现或替换；
- `fallback`：通过通用或较慢 fallback；
- `remote_code`：依赖模型自定义代码；
- `none`：无可用路径；
- `unknown`：未确认。

`fallback` 必须说明正确性、性能和功能限制，不能等同于完整支持。

## 7. Baseline 规则

### 7.1 模式

- `auto`：环境为 `ready` 或 `degraded` 且已有可信命令时执行；否则记录未执行原因；
- `required`：无法执行或执行失败时，结论不能为 `already_supported`；
- `skip`：明确不执行，只做静态评估。

### 7.2 最小化要求

优先使用一次性离线命令；默认：

- 本地模型路径；
- 最小可行 TP；
- 一个短输入；
- greedy decoding；
- `max_new_tokens` 为 1～4；
- 不启用性能特性或复杂服务能力；
- 不下载网络资源；
- 不修改目标仓。

Baseline 前后必须分别记录目标仓 `git status --short`。若命令产生源码修改，标记为异常并停止使用该结果。

### 7.3 失败解释

单次失败只能确认：

- 失败阶段；
- 错误签名；
- 退出码；
- 运行环境与日志位置。

只有日志与代码路径相互印证时，才能将问题归入具体代码缺口。否则使用 `unknown`。

## 8. 缺口分类

使用 `knowledge/common/adaptation/gap-taxonomy.md` 中的分类：

- `model_fact_gap`
- `engine_model_gap`
- `hardware_backend_gap`
- `packaging_version_gap`
- `runtime_environment_gap`
- `configuration_gap`
- `validation_gap`
- `unknown_gap`

其中只有 `engine_model_gap` 和 `hardware_backend_gap` 默认转换为代码工作项。

环境、安装和配置问题应进入“阻塞项/处置建议”，不得伪装成代码修改。

## 9. 工作项要求

每个工作项必须：

- 只覆盖一个能力；
- 有唯一 ID；
- 引用评估缺口和证据；
- 指定 `editable_repository: target_repo | upstream_repo`，并列出候选修改区域；
- 列出依赖；
- 有独立验收方法；
- 不混入性能优化；
- 不把尚未确认的根因写成实现任务。

工作项排序原则：

```text
环境与版本可用
→ 架构识别和配置
→ 模型结构
→ 权重加载
→ 基础算子与 Attention/KV Cache
→ MoE/量化/并行等条件能力
→ 最小运行回归
```

## 10. 结果判定

### `already_supported`

必须同时满足：

- 模型关键能力存在已确认执行路径；
- P800 后端不存在阻塞能力；
- 环境达到 `ready` 或 `degraded`；
- 本次或与当前指纹完全匹配的 baseline 成功；
- 没有 required 能力处于 `unknown`。

### `adaptation_required`

满足任一条件：

- 存在有代码证据的 `engine_model_gap`；
- 存在有代码或测试证据的 `hardware_backend_gap`；
- baseline 与静态代码证据共同确认当前组合缺少某项能力。

环境无法运行 baseline 时，只要静态证据足够，仍可判定 `adaptation_required`，但必须降低 confidence 并明确 baseline 未执行。

### `blocked`

满足任一条件：

- 关键模型事实缺失；
- 目标仓或运行版本无法确定；
- 只有环境错误，无法判断代码能力；
- 关键能力仅有推断，没有代码或运行证据；
- `baseline=required` 但无法执行。

## 11. 重新评估

使用 `--focus <capability>` 时：

- 只重新检查指定能力及其直接依赖；
- 仍需刷新环境和 commit 指纹；
- 不删除旧结论；
- 增加 assessment revision；
- 模型分析 revision 改变时，将受影响的已完成工作项标记为 `needs_recheck`。

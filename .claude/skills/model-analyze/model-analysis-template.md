---
status: in_progress
revision: 1
latest_run: null
---

# 模型分析：<model-id>

## 1. 模型身份

| 项目 | 值 | 状态 | 证据 |
|---|---|---|---|
| 模型名称 | | | |
| 版本 / revision | | | |
| architecture | | | |
| model_type | | | |
| dtype | | | |
| 权重格式 | | | |

## 2. 主干结构

| 能力 | 事实 | 状态 | 证据 |
|---|---|---|---|
| 层数 | | | |
| hidden size | | | |
| normalization | | | |
| activation | | | |
| embedding / lm_head | | | |

## 3. Attention

| 项目 | 值 | 状态 | 证据 |
|---|---|---|---|
| attention heads | | | |
| KV heads | | | |
| head dimension | | | |
| attention 类型 | | | |
| bias | | | |
| sliding / sparse | | | |

## 4. 位置编码

| 项目 | 值 | 状态 | 证据 |
|---|---|---|---|
| 类型 | | | |
| theta | | | |
| scaling | | | |
| 最大长度 | | | |

## 5. FFN / MoE

| 项目 | 值 | 状态 | 证据 |
|---|---|---|---|
| 类型 | | | |
| intermediate size | | | |
| expert 数量 | | | |
| top-k | | | |
| shared expert | | | |
| routing 行为 | | | |

## 6. Checkpoint 与权重组织

| 项目 | 事实 | 状态 | 证据 |
|---|---|---|---|
| shard 数量 | | | |
| 索引文件 | | | |
| 参数命名 | | | |
| packed / merged 权重 | | | |
| 量化配置 | | | |

## 7. Tokenizer 与生成行为

| 项目 | 事实 | 状态 | 证据 |
|---|---|---|---|
| tokenizer 类型 | | | |
| 特殊 token | | | |
| chat template | | | |
| generation config | | | |
| stop / EOS 行为 | | | |

## 8. 模型特有能力

| 能力 | 事实 | 状态 | 证据 |
|---|---|---|---|

## 9. 后续适配必须关注的模型能力

-

## 10. 未知项

| 未知事实 | 对后续适配的影响 | 所需证据 |
|---|---|---|

## 11. 修订记录

### Revision 1

- 模式：首次分析
- 触发来源：用户显式调用
- 变更事实：建立初始模型事实基线
- 受影响能力：全部

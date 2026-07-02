# 模型推理准出文档知识

本 Skill 负责用模型能力把不完整输入整理成推理准出文档。

输入可以是：

```text
csv / tsv / xlsx
Markdown 表格
日志片段
文字说明
```

准出文档结构固定为：

```text
1. 概览
2. 精度测试
3. 性能测试
4. 功能（可选）
5. 推理指南
```

常见字段：

| 字段 | 含义 |
|---|---|
| model | 模型名 |
| framework | 框架名 |
| version | 框架版本 |
| image | 镜像版本 |
| model_url | 权重下载链接 |
| category | 所属章节 |
| hardware | 芯片或硬件 |
| scenario | 测试场景 |
| tp | 卡数 |
| concurrency | 并发 |
| output_throughput | 输出吞吐 |
| e2e_throughput | 端到端吞吐 |
| mean_ttft_ms | TTFT |
| mean_tpot_ms | TPOT |
| ratio | 置换比 |
| theoretical_ratio | 理论置换比 |
| dataset | 精度数据集 |
| score | 精度分数 |
| env | 环境变量 |
| value | 环境变量建议值 |
| description | 描述 |

生成原则：

- 已给出的测试数值必须保真；
- 未给出的测试数值写 `未提供`；
- 部署步骤、环境准备、请求格式可以按模板补全；
- 不根据缺失数据推断精度或性能是否达标；
- 输出必须是可直接发布或二次编辑的 Markdown。

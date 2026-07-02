# model-download 使用说明

这个 Skill 用于下载 Hugging Face / ModelScope 模型权重，并上传到 BOS。也支持跳过下载，直接上传已有本地模型目录。

## 文件结构

```text
.claude/skills/model-download/
├── SKILL.md
├── README.md
└── scripts/
    ├── model_download_config.sh
    ├── model_download_config.sh.example
    ├── model_transfer.py
    └── run_model_transfer.sh
```

## 第一次使用

编辑配置：

```bash
vim .claude/skills/model-download/scripts/model_download_config.sh
```

主要填写：

```bash
Source="hf"  # hf | modelscope | local
Local_model_path="/home/models/DeepSeek-V4-Pro-FP8"
HF_model_path="sgl-project/DeepSeek-V4-Pro-FP8"
MS_model_path=""
BOS_model_path="bos:/aihc-private-hcd/LLM/DeepSeek/DeepSeek-V4-Pro-FP8"
```

`model_download_config.sh` 默认已经存在，必填模型路径默认留空。`model_download_config.sh.example` 只作为字段参考。

默认代理按来源文档配置：

```bash
Proxy_url="http://192.168.48.191:18000"
```

## 执行顺序

检查依赖：

```bash
bash .claude/skills/model-download/scripts/run_model_transfer.sh --preflight
```

预览命令：

```bash
bash .claude/skills/model-download/scripts/run_model_transfer.sh --dry-run
```

正式执行：

```bash
bash .claude/skills/model-download/scripts/run_model_transfer.sh
```

## 三种场景

Hugging Face：

```bash
Source="hf"
HF_model_path="sgl-project/DeepSeek-V4-Pro-FP8"
```

ModelScope：

```bash
Source="modelscope"
MS_model_path="Qwen/Qwen2.5-32B-Instruct-AWQ"
```

已有本地目录：

```bash
Source="local"
Local_model_path="/home/models/DeepSeek-V4-Pro-FP8"
```

## 上传确认

默认上传前需要人工确认：

```bash
Skip_upload_confirmation=0
```

只有明确不需要确认时才改成：

```bash
Skip_upload_confirmation=1
```

## 进度提示

默认每 60 秒输出一次进度：

```bash
Progress_interval=60
```

进度格式：

```text
[PROGRESS] elapsed_seconds=... local_size=... files=... upload_candidates=...
```

## 输出文件

```text
outputs/download.log
outputs/watch-upload.log
BOS model directory
```

最终输出会包含：

```text
source
model_id
local_dir
bos_path
download_status
upload_status
output_dir
elapsed_seconds
```

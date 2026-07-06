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

主要填写（推荐直接贴模型网址，来源自动识别）：

```bash
Model_url="https://huggingface.co/sgl-project/DeepSeek-V4-Pro-FP8"   # 贴 HF 或 ModelScope 网址即可
Source=""                                                            # 留空由网址自动判断
Local_model_path="/home/models/DeepSeek-V4-Pro-FP8"
BOS_model_path="bos:/aihc-private-hcd/LLM/DeepSeek/DeepSeek-V4-Pro-FP8"
```

`Model_url` 会根据网址域名自动识别是 Hugging Face 还是 ModelScope，无需再手动设 `Source`：

```text
huggingface.co / hf.co       -> hf
modelscope.cn                -> modelscope
```

支持带 `/tree/main`、`/blob/...`、`/summary`、`/files` 等后缀的页面网址。若只有裸 repo id（如 `org/model`，无域名），则无法自动识别，需要显式设 `Source`，或改用完整网址。上传已有本地目录时设 `Source="local"`（不用填网址）。

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

Hugging Face（贴网址，自动识别）：

```bash
Model_url="https://huggingface.co/sgl-project/DeepSeek-V4-Pro-FP8"
Source=""
```

ModelScope（贴网址，自动识别）：

```bash
Model_url="https://www.modelscope.cn/models/Qwen/Qwen2.5-32B-Instruct-AWQ"
Source=""
```

已有本地目录（不下载，只上传）：

```bash
Source="local"
Local_model_path="/home/models/DeepSeek-V4-Pro-FP8"
```

## 打包上传（可选）

默认逐个文件上传（`bcecmd bos sync`）。如果想上传单个压缩包，加 `--pack`，格式由你指定：

```bash
# 打包为 tar（不压缩）
bash .claude/skills/model-download/scripts/run_model_transfer.sh \
  <MODEL_URL> --local <LOCAL_DIR> --bos <BOS_PATH> --pack tar

# 打包为 tar.gz（gzip 压缩）
bash .claude/skills/model-download/scripts/run_model_transfer.sh \
  <MODEL_URL> --local <LOCAL_DIR> --bos <BOS_PATH> --pack tar.gz
```

也可以在配置里设置：

```bash
Pack=""        # "" 不打包（默认）| tar | tar.gz
```

说明：

- 压缩包生成在本地目录同级，命名为 `<目录名>.tar` 或 `<目录名>.tar.gz`，包内顶层即该目录名。
- 打包在下载完成后进行，上传的是压缩包（`bcecmd bos cp` 到 `<BOS_PATH>/<压缩包名>`），此时不启用边下边传的逐文件监听。
- 缺少 `bcecmd` 或使用 `--no-upload` 时，最后打印的手动上传命令会自动改成针对压缩包的 `bcecmd bos cp`。

## 上传确认

默认上传前需要人工确认：

```bash
Skip_upload_confirmation=0
```

只有明确不需要确认时才改成：

```bash
Skip_upload_confirmation=1
```

## 没有 bcecmd 时

如果机器上没有 `bcecmd`，脚本不会直接失败：会继续完成下载、跳过上传，并打印手动上传命令，提示你自己安装 `bcecmd` 后上传：

```text
[MANUAL UPLOAD REQUIRED] bcecmd not found; download continues but upload is skipped.
Install and configure bcecmd, then upload the local directory yourself:
  bcecmd bos sync <Local_model_path> <BOS_model_path> --concurrency 64
```

此时 `upload_status` 会记为 `skipped_manual`。

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
pack
archive
bos_path
download_status
upload_status
output_dir
elapsed_seconds
```

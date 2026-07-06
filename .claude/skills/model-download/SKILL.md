---
name: model-download
description: 从 Hugging Face 或 ModelScope 下载模型权重，并通过可复用的命令流程上传到 BOS。
argument-hint: "[MODEL_URL] [--local DIR] [--bos BOS_PATH] [--pack tar|tar.gz] [--no-upload] [--dry-run]"
disable-model-invocation: true
---

# 模型下载（Model Download）

当用户需要从 Hugging Face 或 ModelScope 下载模型权重，或把已有的本地模型目录上传到 BOS 时，使用本 Skill。

## 外网代理

- 在本环境访问外部站点（GitHub、huggingface.co、modelscope.cn）前，先导出下面的代理；直连以及旧的 `192.168.48.191:18000` 代理在这里可能超时。

```bash
export HTTP_PROXY=http://agent.baidu.com:8891
export HTTPS_PROXY=http://agent.baidu.com:8891
export http_proxy=http://agent.baidu.com:8891
export https_proxy=http://agent.baidu.com:8891
```

- 下面的 agent 前置步骤（用 `WebFetch`、`curl`、`git`、`gh` 抓取仓库 README / 配置）需要用这个代理。
- 对于 `hf` / `modelscope` 下载子进程，把下载代理设成同一个地址：在配置里写 `Proxy_url="http://agent.baidu.com:8891"`，或传 `--proxy http://agent.baidu.com:8891`。只有目标机器能直连外网时才设 `Proxy_url=""`。

## 通过参数传入（来自 prompt）

- 网址、本地路径、BOS 路径都可以作为参数传入，无需改配置文件。参数始终覆盖配置文件里的值。

```bash
bash .claude/skills/model-download/scripts/run_model_transfer.sh \
  <MODEL_URL> --local <LOCAL_DIR> --bos <BOS_PATH>
```

- 支持的参数形式：
  - `<MODEL_URL>` 位置参数，或 `--url` / `--model-url` / `--model-id`
  - `--local`（别名 `--local-dir`）：权重在本机的落盘目录（始终必填）
  - `--bos`（别名 `--bos-path`）：BOS 上传目标（启用上传时必填）
  - `--pack tar|tar.gz` 可选：把下载目录打包成压缩包后再上传（见「打包上传」）
  - `--no-upload` 只下载；`--upload` 强制开启上传
  - `--source hf|modelscope|local` 仅当输入是裸 repo id、而非网址时才需要
  - 其他参数（`--dry-run`、`--preflight`）原样透传
- 光有网址不足以运行。`Local_model_path` 始终必填，启用上传时 `BOS_model_path` 必填。绝不臆测任何路径；prompt 没提供的要向用户询问，两个必填路径都确认后才继续。
- 来源（`hf` / `modelscope`）会从网址自动识别；传网址时不要设 `--source`。

## GitHub 仓库引用的模型（agent 前置步骤）

- 传输脚本**不认识** GitHub 仓库网址。不要把 `github.com/...` 网址传给 `run_model_transfer.sh`；它只接受 Hugging Face / ModelScope 网址或裸 repo id。
- 当用户给的是 GitHub 仓库（例如 `https://github.com/<org>/<repo>`），先自行检视该仓库，再对发现的每个模型逐个驱动本 Skill。不要修改脚本。
- 通过阅读仓库文档和代码来发现它依赖的模型：
  - 抓取 `README.md` / `README_*.md`，以及 `docs/`、`MODEL_ZOO*`、权重表格、`download*.sh`、配置文件等常见位置。
  - 查找 `huggingface.co` / `hf.co` / `modelscope.cn` 链接、裸 repo id，以及 `from_pretrained("...")`、`hf download ...`、`huggingface-cli download ...`、`modelscope download ...` 这类调用。
  - 用 `WebFetch`，或 `gh`/`git`/`curl`（先导出上面的代理）读取仓库。如果网络仍然不通，告诉用户并请其把相关模型链接贴过来。
- 把发现的每个 Hugging Face / ModelScope 模型都列给用户（id + 用途）。确认要下载哪些，并为每个模型取得 `Local_model_path`（如需上传还要 `BOS_model_path`）。绝不臆测路径。
- 然后对每个已确认的模型各跑一次本 Skill，各自带 `--local`（和 `--bos`）。分别汇报每个模型的结果。
- 不在范围内：仅以直链形式提供的权重（普通服务器上的 `.pth`/`.bin`、Google Drive、百度网盘等）不是 Hugging Face / ModelScope 来源。明确说明这一点；用户可手动下载该文件后用 `local` 流程上传，或在本 Skill 之外处理。

## 集中修改参数（参数方式的替代方案）

```text
.claude/skills/model-download/scripts/model_download_config.sh
```

配置文件默认已存在。必填的模型路径字段默认留空。

## 来源自动识别

- 把 Hugging Face 或 ModelScope 模型网址填进 `Model_url`，并把 `Source` 留空；来源（`hf` / `modelscope`）会从网址域名自动识别。
- 可识别的网址包括 `huggingface.co/<org>/<model>`（也支持 `hf.co`、`/tree/...`、`/blob/...` 等浏览路径）和 `modelscope.cn/models/<org>/<model>`（也支持 `/summary`、`/files`）。
- 裸 repo id（例如 `org/model`）没有域名，无法自动识别；要么用完整网址，要么显式设 `Source`。
- 若只是上传已有本地目录而不下载，设 `Source="local"`（无需网址）。

## 打包上传（可选）

- 默认行为是把下载目录里的文件逐个（`bcecmd bos sync`）上传到 BOS。加上 `--pack` 后，会先把整个下载目录打成一个压缩包，然后上传这个压缩包（`bcecmd bos cp`），BOS 上得到的是单个压缩文件而不是展开的目录。
- 格式由用户指定，只支持两种：

```text
--pack tar      打包为 <目录名>.tar（不压缩）
--pack tar.gz   打包为 <目录名>.tar.gz（gzip 压缩）
```

- 也可以在配置里设 `Pack="tar"` 或 `Pack="tar.gz"`；留空 `Pack=""` 表示不打包、按原文件上传。参数 `--pack` 覆盖配置。
- 压缩包生成在本地目录的同级，命名为 `<Local_model_path 的目录名>.tar` 或 `.tar.gz`，压缩包内顶层就是该目录名。
- 打包会在下载完成后一次性进行，因此不启用边下边传的增量监听；仅在不打包时才逐文件监听上传。
- 上传目标为 `<BOS_model_path>/<压缩包名>`。启用打包时 `BOS_model_path` 仍必填。
- 打包示例：

```bash
bash .claude/skills/model-download/scripts/run_model_transfer.sh \
  <MODEL_URL> --local <LOCAL_DIR> --bos <BOS_PATH> --pack tar.gz
```

- 缺少 `bcecmd` 或使用 `--no-upload` 时，最后打印的手动上传命令会自动变成针对压缩包的 `bcecmd bos cp <压缩包> <BOS_PATH>/<压缩包名>`。

## 主要命令

```bash
bash .claude/skills/model-download/scripts/run_model_transfer.sh
```

预览（Dry run）：

```bash
bash .claude/skills/model-download/scripts/run_model_transfer.sh --dry-run
```

预检（Preflight）：

```bash
bash .claude/skills/model-download/scripts/run_model_transfer.sh --preflight
```

## 支持的来源

```text
hf          通过 hf 或 huggingface-cli 从 Hugging Face 下载
modelscope  通过 modelscope 从 ModelScope 下载
local       跳过下载，上传已有本地目录
```

## 预期产出

```text
outputs/download.log
outputs/watch-upload.log
BOS 模型目录
```

## 规则

- 不要在本 Skill 中保存 token、密码或机器凭证。
- 把模型路径和传输设置写进 `scripts/model_download_config.sh`，不要写进 `SKILL.md`。
- `scripts/model_download_config.sh.example` 仅作为字段参考。
- 下载优先用 `Model_url` 以便自动识别来源；除非上传本地目录（`Source="local"`）或输入是裸 repo id，否则把 `Source` 留空。
- 默认代理按来源文档配置：`http://192.168.48.191:18000`；只有目标机器不需要代理时才设 `Proxy_url=""`。
- 启用上传时必须提供 `BOS_model_path`。
- 对于私有的 Hugging Face 或 ModelScope 模型，在使用本 Skill 前先运行对应的登录命令。
- 上传前需征求人工确认，除非用户的 prompt 明确表示不需要确认。
- 当明确不需要确认时，在配置里设 `Skip_upload_confirmation=1`。
- 在长时间下载或上传期间，每隔 `Progress_interval` 秒检查一次进度，汇报已用时间、本地大小、文件数量，以及最新的上传/下载状态。
- 边下载边上传会监听已稳定的文件，最后再执行一次 `bcecmd bos sync`。
- 使用 `--pack tar|tar.gz` 时改为：下载完成后打成一个压缩包，再用 `bcecmd bos cp` 上传该压缩包（不启用逐文件监听）。上传的是压缩包本身。
- 上传后，汇报本地路径、BOS 路径、下载状态、上传状态，以及未解决的失败项。

## 缺少 `bcecmd` 时

- 执行上传前，先检查 `bcecmd`（例如通过 `--preflight`）。
- 如果 `bcecmd` 缺失，不要当作失败处理。用配置里的值填好，把确切的上传命令告诉用户，让其自行执行：

```bash
bcecmd bos sync <Local_model_path> <BOS_model_path> --concurrency <Upload_concurrency>
```

- 启用 `--pack` 时，改为提示压缩包的上传命令：

```bash
bcecmd bos cp <压缩包路径> <BOS_model_path>/<压缩包名> --concurrency <Upload_concurrency>
```

- 随后让传输仅执行下载（跳过上传）。脚本会记 `upload_status=skipped_manual` 并重新打印同一条命令。把该命令转达给用户；在 `bcecmd` 安装并配置好之前，不要自己尝试上传。

## 示例

```bash
# Hugging Face
bash .claude/skills/model-download/scripts/run_model_transfer.sh

# ModelScope
bash .claude/skills/model-download/scripts/run_model_transfer.sh

# 已有本地模型
bash .claude/skills/model-download/scripts/run_model_transfer.sh
```

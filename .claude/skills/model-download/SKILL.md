---
name: model-download
description: Download Hugging Face or ModelScope model weights and upload them to BOS with a repeatable command flow.
argument-hint: "[--config PATH] [--dry-run]"
disable-model-invocation: true
---

# Model Download

Use this skill when a user needs to download model weights from Hugging Face or ModelScope, or upload an existing local model directory to BOS.

Edit parameters in one place:

```text
.claude/skills/model-download/scripts/model_download_config.sh
```

The config file exists by default. Required model path fields are intentionally empty.

Primary command:

```bash
bash .claude/skills/model-download/scripts/run_model_transfer.sh
```

Dry run:

```bash
bash .claude/skills/model-download/scripts/run_model_transfer.sh --dry-run
```

Preflight:

```bash
bash .claude/skills/model-download/scripts/run_model_transfer.sh --preflight
```

Supported sources:

```text
hf          Hugging Face download through hf or huggingface-cli
modelscope  ModelScope download through modelscope
local       skip download and upload an existing local directory
```

Expected outputs:

```text
outputs/download.log
outputs/watch-upload.log
BOS model directory
```

Rules:

- Do not store tokens, passwords, or machine credentials in this skill.
- Put model paths and transfer settings in `scripts/model_download_config.sh`, not in `SKILL.md`.
- Use `scripts/model_download_config.sh.example` only as the field reference.
- The default proxy follows the source knowledge page: `http://192.168.48.191:18000`; set `Proxy_url=""` only when the target machine does not need it.
- Require `BOS_model_path` when upload is enabled.
- For private Hugging Face or ModelScope models, run the matching login command before this skill.
- Before upload, ask for human confirmation unless the user's prompt explicitly says confirmation is not required.
- When confirmation is explicitly not required, set `Skip_upload_confirmation=1` in the config.
- During long downloads or uploads, check progress every `Progress_interval` seconds and report elapsed time, local size, file count, and latest upload/download status.
- Upload during download watches stable files, then runs one final `bcecmd bos sync`.
- After upload, report the local path, BOS path, download status, upload status, and unresolved failures.

Examples:

```bash
# Hugging Face
bash .claude/skills/model-download/scripts/run_model_transfer.sh

# ModelScope
bash .claude/skills/model-download/scripts/run_model_transfer.sh

# Existing local model
bash .claude/skills/model-download/scripts/run_model_transfer.sh
```

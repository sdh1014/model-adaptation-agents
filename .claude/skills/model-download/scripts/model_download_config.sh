#!/usr/bin/env bash

# Edit this file for each model transfer task. Required model paths are empty by default.

# Easiest path: paste a Hugging Face or ModelScope model URL here and leave Source empty.
# The source (hf / modelscope) is auto-detected from the URL.
#   e.g. Model_url="https://huggingface.co/sgl-project/DeepSeek-V4-Pro-FP8"
#   e.g. Model_url="https://www.modelscope.cn/models/Qwen/Qwen2.5-32B-Instruct-AWQ"
Model_url="${Model_url:-}"

# Only needed when NOT using Model_url. Use Source="local" to upload an existing
# local directory without downloading. Leave empty when Model_url is set.
Source="${Source:-}"  # "" | hf | modelscope | local

Local_model_path="${Local_model_path:-}"

# Legacy per-source fields. Ignored when Model_url is set. Accept id or URL.
HF_model_path="${HF_model_path:-}"
MS_model_path="${MS_model_path:-}"
BOS_model_path="${BOS_model_path:-}"
Proxy_url="${Proxy_url:-http://192.168.48.191:18000}"

Max_workers="${Max_workers:-8}"
Upload_concurrency="${Upload_concurrency:-64}"
Progress_interval="${Progress_interval:-60}"

# Optional packaging. "" = upload files as-is; "tar" or "tar.gz" packs the
# downloaded dir into one archive and uploads that archive instead.
Pack="${Pack:-}"                           # "" | tar | tar.gz

Upload="${Upload:-1}"                      # 1 uploads to BOS, 0 downloads only
Skip_upload_confirmation="${Skip_upload_confirmation:-0}"

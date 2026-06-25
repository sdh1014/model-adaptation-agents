#!/usr/bin/env bash

# 直接粘贴项目实际使用的环境变量。
export PYTHONPATH="${TARGET_REPO}:${UPSTREAM_REPO:-}:${PYTHONPATH:-}"

# 示例占位；请替换为当前 P800 环境实际变量。
# export XPU_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
# export LD_LIBRARY_PATH=/opt/kunlun/lib:${LD_LIBRARY_PATH:-}

export MODEL_HOST=127.0.0.1
export MODEL_PORT=8000
export MODEL_BASE_URL="http://${MODEL_HOST}:${MODEL_PORT}"
export MODEL_STARTUP_TIMEOUT=1200
export MODEL_CHECK_TIMEOUT=300
export MODEL_SHUTDOWN_TIMEOUT=60
export MODEL_READY_PROBE_TIMEOUT=15

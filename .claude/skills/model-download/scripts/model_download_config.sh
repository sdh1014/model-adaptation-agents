#!/usr/bin/env bash

# Edit this file for each model transfer task. Required model paths are empty by default.

Source="${Source:-hf}"  # hf | modelscope | local

Local_model_path="${Local_model_path:-}"
HF_model_path="${HF_model_path:-}"
MS_model_path="${MS_model_path:-}"
BOS_model_path="${BOS_model_path:-}"
Proxy_url="${Proxy_url:-http://192.168.48.191:18000}"

Max_workers="${Max_workers:-8}"
Upload_concurrency="${Upload_concurrency:-64}"
Progress_interval="${Progress_interval:-60}"

Upload="${Upload:-1}"                      # 1 uploads to BOS, 0 downloads only
Private_model="${Private_model:-0}"        # 1 after CLI login, 0 for public models
Skip_upload_confirmation="${Skip_upload_confirmation:-0}"

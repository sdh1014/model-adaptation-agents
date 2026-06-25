#!/usr/bin/env bash
set -euo pipefail
curl --fail --silent --show-error "${MODEL_BASE_URL}/health" >/dev/null

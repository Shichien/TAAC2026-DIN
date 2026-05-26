#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH:-}"

python3 -u "${SCRIPT_DIR}/train.py"     --ns_tokenizer_type rankmixer     --user_ns_tokens 5     --item_ns_tokens 2     --num_queries 2     --use_absolute_time_features     --split_user_dense_tokens     --use_item_context     --use_aligned_user_pair_features     --ns_groups_json ""     --emb_skip_threshold 1000000     --dense_optimizer_type muon     --num_workers 8     --num_epochs 6     --amp     --amp_dtype bfloat16     "$@"

#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH:-}"
python3 -u "${SCRIPT_DIR}/train.py" \
    --emb_skip_threshold 1000000 \
    --seq_hash_embedding seq_b:69:65536:4,seq_c:29:65536:4,seq_c:34:65536:4,seq_c:47:65536:4 \
    --hash_sparse_lr 0.01 \
    --dense_optimizer_type adamw \
    --reinit_cardinality_threshold 1 \
    --split_method row_group \
    --valid_ratio 0.05 \
    --num_workers 8 \
    --num_epochs 4 \
    --save_each_epoch_ckpt \
    --seed 42 \
    --amp \
    --amp_dtype bfloat16 \
    --compile_model \
    --compile_mode reduce-overhead \
    "$@"

#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH}"

# ---- 当前配置：RankMixer NS tokenizer（无需 ns_groups.json） ----
python3 -u "${SCRIPT_DIR}/train.py" \
    --ns_tokenizer_type rankmixer \
    --user_ns_tokens 5 \
    --item_ns_tokens 2 \
    --num_queries 2 \
    --ns_groups_json "" \
    --emb_skip_threshold 1000000 \
    --num_workers 8 \
    "$@"

# ---- 备选配置：由 ns_groups.json 驱动的 GroupNSTokenizer ----
# 使用 ns_groups.json 中的特征分组（7 个用户组 + 4 个物品组）。
# 当 d_model=64 且 num_ns=12（7 个 user_int + 1 个 user_dense + 4 个 item_int）时，
# 只有 num_queries=1 满足 d_model % T == 0（T = num_queries*4 + num_ns）。
# 若要切换，请注释上方代码块并取消注释下方代码块。
#
# python3 -u "${SCRIPT_DIR}/train.py" \
#     --ns_tokenizer_type group \
#     --ns_groups_json "${SCRIPT_DIR}/ns_groups.json" \
#     --num_queries 1 \
#     --emb_skip_threshold 100000 \
#     --num_workers 8 \
#     "$@"

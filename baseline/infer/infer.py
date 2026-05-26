"""PCVRHyFormer 推理脚本（由参赛者上传到评测容器）。

模型构造与 ``train.py`` 保持镜像一致：我们根据 ``schema.json`` + ``ns_groups.json`` + ``train_config.json`` 重建模型。所有模型超参数优先从 ckpt 目录中的 ``train_config.json`` 解析（``trainer.py`` 保存检查点时写入），然后回退到下面的 ``_FALLBACK_MODEL_CFG``（必须与 ``train.py`` 中的 CLI 默认值保持一致）。

只支持 Parquet 数据格式。

环境变量:
    MODEL_OUTPUT_PATH  检查点目录（指向包含 ``model.pt`` / ``train_config.json`` 的 ``global_step`` 子目录）。
    EVAL_DATA_PATH     测试数据目录（*.parquet + schema.json）。
    EVAL_RESULT_PATH   生成 ``predictions.json`` 的目录。
"""

import os
import json
import logging
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import FeatureSchema, PCVRParquetDataset, NUM_TIME_BUCKETS
from model import PCVRHyFormer, ModelInput


logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


# 仅当 ckpt 目录缺少 ``train_config.json`` 时，才使用这些回退值。
#
#
# 这些值必须与 ``train.py`` 中的 argparse 默认值保持一致；否则一旦
# 实际走到回退路径，构建出的模型形状会与保存的
# state_dict 不匹配。
#
# 关于 ``num_time_buckets`` 的特别说明：该值严格由
# ``dataset.BUCKET_BOUNDARIES`` 决定，不是独立超参数。
# 因此当该特性启用时，我们使用 dataset 模块暴露的常量；
# ``0`` 表示禁用。
_FALLBACK_MODEL_CFG = {
    "d_model": 64,
    "emb_dim": 64,
    "num_queries": 1,
    "num_hyformer_blocks": 2,
    "num_heads": 4,
    "seq_encoder_type": "transformer",
    "hidden_mult": 4,
    "dropout_rate": 0.01,
    "seq_top_k": 50,
    "seq_causal": False,
    "action_num": 1,
    "num_time_buckets": NUM_TIME_BUCKETS,
    "rank_mixer_mode": "full",
    "use_rope": False,
    "rope_base": 10000.0,
    "emb_skip_threshold": 0,
    "seq_id_threshold": 10000,
    "ns_tokenizer_type": "rankmixer",
    "user_ns_tokens": 0,
    "item_ns_tokens": 0,
}

_FALLBACK_SEQ_MAX_LENS = "seq_a:256,seq_b:256,seq_c:512,seq_d:512"
_FALLBACK_BATCH_SIZE = 256
_FALLBACK_NUM_WORKERS = 16


# 用于构建模型的超参数键。``train_config.json`` 中的其他内容
# 在构造 ``PCVRHyFormer`` 时会被忽略。
_MODEL_CFG_KEYS = list(_FALLBACK_MODEL_CFG.keys())


def build_feature_specs(
    schema: FeatureSchema, per_position_vocab_sizes: List[int]
) -> List[Tuple[int, int, int]]:
    """按 ``schema.entries`` 的顺序构造 ``feature_specs = [(vocab_size, offset, length), ...]``。"""
    specs: List[Tuple[int, int, int]] = []
    for fid, offset, length in schema.entries:
        vs = max(per_position_vocab_sizes[offset : offset + length])
        specs.append((vs, offset, length))
    return specs


def _parse_seq_max_lens(sml_str: str) -> Dict[str, int]:
    """将形如 ``'seq_a:256,seq_b:256,...'`` 的字符串解析为 dict。"""
    seq_max_lens: Dict[str, int] = {}
    for pair in sml_str.split(","):
        k, v = pair.split(":")
        seq_max_lens[k.strip()] = int(v.strip())
    return seq_max_lens


def load_train_config(model_dir: str) -> Dict[str, Any]:
    """从 ckpt 目录加载 ``train_config.json``。

    如果文件不存在，返回空 dict（会触发 fallback 解析）。
    """
    train_config_path = os.path.join(model_dir, "train_config.json")
    if os.path.exists(train_config_path):
        with open(train_config_path, "r") as f:
            cfg = json.load(f)
        logging.info(f"Loaded train_config from {train_config_path}")
        return cfg
    logging.warning(
        f"train_config.json not found in {model_dir}, "
        f"falling back to hardcoded defaults. "
        f"Shape mismatch may occur if training used non-default hyperparameters."
    )
    return {}


def resolve_model_cfg(train_config: Dict[str, Any]) -> Dict[str, Any]:
    """从 ``train_config`` 提取模型超参数；缺失键回退到 ``_FALLBACK_MODEL_CFG``。

    对 ``num_time_buckets`` 做特殊处理：它不是 CLI 上的独立超参数；bucket 数量由 ``dataset.BUCKET_BOUNDARIES`` 的长度唯一决定。解析顺序:

      1) ``train_config`` 直接包含 ``num_time_buckets``（旧版 ckpt）
         -> 使用该值；
      2) ``train_config`` 包含 ``use_time_buckets``（新版训练）
         -> 推导为 ``NUM_TIME_BUCKETS`` 或 ``0``；
      3) 两者都不存在 -> 回退到 ``_FALLBACK_MODEL_CFG[...]``。
    """
    cfg: Dict[str, Any] = {}
    for key in _MODEL_CFG_KEYS:
        if key == "num_time_buckets":
            if "num_time_buckets" in train_config:
                cfg[key] = train_config["num_time_buckets"]
            elif "use_time_buckets" in train_config:
                cfg[key] = (
                    NUM_TIME_BUCKETS if train_config["use_time_buckets"] else 0
                )
            else:
                cfg[key] = _FALLBACK_MODEL_CFG[key]
                logging.warning(
                    f"train_config missing both 'num_time_buckets' and 'use_time_buckets', "
                    f"using fallback = {cfg[key]}"
                )
            continue

        if key in train_config:
            cfg[key] = train_config[key]
        else:
            cfg[key] = _FALLBACK_MODEL_CFG[key]
            logging.warning(
                f"train_config missing '{key}', using fallback = {cfg[key]}"
            )
    return cfg


def build_model(
    dataset: PCVRParquetDataset,
    model_cfg: Dict[str, Any],
    ns_groups_json: Optional[str] = None,
    device: str = "cpu",
) -> PCVRHyFormer:
    """根据数据集 schema、NS-groups JSON 和解析后的 ``model_cfg`` dict 构造 ``PCVRHyFormer``。

    参数:
        dataset: 提供特征 schema 的 ``PCVRParquetDataset``。
        model_cfg: 解析后的模型超参数，通常是 ``resolve_model_cfg`` 的输出。
        ns_groups_json: NS-groups JSON 文件路径；``None`` / 空字符串表示禁用它（每个特征成为自己的单元素组）。
        device: torch 设备。
    """
    # NS 分组。JSON schema 使用 *fid*（特征 id）值；这里将其
    # 转换为 ``user_int_schema.entries`` /
    # ``item_int_schema.entries`` 中的位置索引，使 ``GroupNSTokenizer`` /
    # ``RankMixerNSTokenizer`` 能直接索引 ``feature_specs``。这与
    # ``train.py`` 加载 JSON 时执行的转换相同；
    # 在这里执行可让 infer.py 与训练保持对称。
    user_ns_groups: List[List[int]]
    item_ns_groups: List[List[int]]
    if ns_groups_json and os.path.exists(ns_groups_json):
        logging.info(f"Loading NS groups from {ns_groups_json}")
        with open(ns_groups_json, "r") as f:
            ns_groups_cfg = json.load(f)
        user_fid_to_idx = {
            fid: i
            for i, (fid, _, _) in enumerate(dataset.user_int_schema.entries)
        }
        item_fid_to_idx = {
            fid: i
            for i, (fid, _, _) in enumerate(dataset.item_int_schema.entries)
        }
        try:
            user_ns_groups = [
                [user_fid_to_idx[f] for f in fids]
                for fids in ns_groups_cfg["user_ns_groups"].values()
            ]
            item_ns_groups = [
                [item_fid_to_idx[f] for f in fids]
                for fids in ns_groups_cfg["item_ns_groups"].values()
            ]
        except KeyError as exc:
            raise KeyError(
                f"NS-groups JSON references fid {exc.args[0]} which is not "
                f"present in the checkpoint's schema.json. The ns_groups.json "
                f"and schema.json must come from the same training run."
            ) from exc
    else:
        logging.info(
            "No NS groups JSON found, using default: each feature as one group"
        )
        user_ns_groups = [
            [i] for i in range(len(dataset.user_int_schema.entries))
        ]
        item_ns_groups = [
            [i] for i in range(len(dataset.item_int_schema.entries))
        ]

    # 特征规格。
    user_int_feature_specs = build_feature_specs(
        dataset.user_int_schema, dataset.user_int_vocab_sizes
    )
    item_int_feature_specs = build_feature_specs(
        dataset.item_int_schema, dataset.item_int_vocab_sizes
    )

    logging.info(f"Building PCVRHyFormer with cfg: {model_cfg}")
    model = PCVRHyFormer(
        user_int_feature_specs=user_int_feature_specs,
        item_int_feature_specs=item_int_feature_specs,
        user_dense_dim=dataset.user_dense_schema.total_dim,
        item_dense_dim=dataset.item_dense_schema.total_dim,
        seq_vocab_sizes=dataset.seq_domain_vocab_sizes,
        user_ns_groups=user_ns_groups,
        item_ns_groups=item_ns_groups,
        **model_cfg,
    ).to(device)

    return model


def load_model_state_strict(
    model: nn.Module, ckpt_path: str, device: str
) -> None:
    """严格加载 ``state_dict``；任何缺失或意外 key 都会快速失败并输出诊断信息。"""
    state_dict = torch.load(ckpt_path, map_location=device)
    try:
        model.load_state_dict(state_dict, strict=True)
    except RuntimeError as e:
        logging.error(
            "Failed to load state_dict in strict mode. This usually means the "
            "model constructed by build_model does NOT match the checkpoint. "
            "Check that train_config.json in the ckpt dir is present and matches "
            "the training hyperparameters."
        )
        raise e


def get_ckpt_path() -> Optional[str]:
    """在 ``$MODEL_OUTPUT_PATH`` 指向的目录中查找第一个 ``*.pt`` 文件。

    如果未找到检查点，则返回 ``None``。
    """
    ckpt_path = os.environ.get("MODEL_OUTPUT_PATH")
    if not ckpt_path:
        return None
    for item in os.listdir(ckpt_path):
        if item.endswith(".pt"):
            return os.path.join(ckpt_path, item)
    return None


def _batch_to_model_input(batch: Dict[str, Any], device: str) -> ModelInput:
    """将 batch dict 转换为 ``ModelInput``，并处理动态序列域。"""
    device_batch: Dict[str, Any] = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            device_batch[k] = v.to(device, non_blocking=True)
        else:
            device_batch[k] = v

    seq_domains = device_batch["_seq_domains"]
    seq_data: Dict[str, torch.Tensor] = {}
    seq_lens: Dict[str, torch.Tensor] = {}
    seq_time_buckets: Dict[str, torch.Tensor] = {}
    for domain in seq_domains:
        seq_data[domain] = device_batch[domain]
        seq_lens[domain] = device_batch[f"{domain}_len"]
        B, _, L = device_batch[domain].shape
        seq_time_buckets[domain] = device_batch.get(
            f"{domain}_time_bucket",
            torch.zeros(B, L, dtype=torch.long, device=device),
        )

    return ModelInput(
        user_int_feats=device_batch["user_int_feats"],
        item_int_feats=device_batch["item_int_feats"],
        user_dense_feats=device_batch["user_dense_feats"],
        item_dense_feats=device_batch["item_dense_feats"],
        seq_data=seq_data,
        seq_lens=seq_lens,
        seq_time_buckets=seq_time_buckets,
    )


def main() -> None:
    # ---- 读取环境变量 ----
    model_dir = os.environ.get("MODEL_OUTPUT_PATH")
    data_dir = os.environ.get("EVAL_DATA_PATH")
    result_dir = os.environ.get("EVAL_RESULT_PATH")

    os.makedirs(result_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ---- Schema：优先使用 model_dir 中的版本（精确匹配训练）；
    #      若缺失则回退到 data_dir 中的版本。 ----
    schema_path = os.path.join(model_dir, "schema.json")
    if not os.path.exists(schema_path):
        schema_path = os.path.join(data_dir, "schema.json")
    logging.info(f"Using schema: {schema_path}")

    # ---- 加载 train_config.json（所有超参数的单一事实来源） ----
    train_config = load_train_config(model_dir)

    # ---- 解析 seq_max_lens ----
    sml_str = train_config.get("seq_max_lens", _FALLBACK_SEQ_MAX_LENS)
    seq_max_lens = _parse_seq_max_lens(sml_str)
    logging.info(f"seq_max_lens: {seq_max_lens}")

    # ---- 数据加载：复用训练配置中的 batch_size / num_workers ----
    batch_size = int(train_config.get("batch_size", _FALLBACK_BATCH_SIZE))
    num_workers = int(train_config.get("num_workers", _FALLBACK_NUM_WORKERS))

    test_dataset = PCVRParquetDataset(
        parquet_path=data_dir,
        schema_path=schema_path,
        batch_size=batch_size,
        seq_max_lens=seq_max_lens,
        shuffle=False,
        buffer_batches=0,
        is_training=False,
    )
    total_test_samples = test_dataset.num_rows
    logging.info(f"Total test samples: {total_test_samples}")

    # ---- 构建模型：所有结构性超参数都从 train_config 解析 ----
    model_cfg = resolve_model_cfg(train_config)

    # ns_groups_json 也来自训练配置（例如 run.sh 可能传入
    # 空字符串来禁用它）。当 trainer.py 已将 JSON 复制到
    # ckpt 目录时，train_config 只记录 basename，因此先尝试
    # 相对 ``model_dir`` 解析，再使用原始路径（可能是
    # 绝对路径）作为回退。
    ns_groups_json = train_config.get("ns_groups_json", None)
    if ns_groups_json:
        local_candidate = os.path.join(
            model_dir, os.path.basename(ns_groups_json)
        )
        if os.path.exists(local_candidate):
            ns_groups_json = local_candidate

    model = build_model(
        test_dataset,
        model_cfg=model_cfg,
        ns_groups_json=ns_groups_json,
        device=device,
    )

    # ---- 严格加载权重 ----
    ckpt_path = get_ckpt_path()
    if ckpt_path is None:
        raise FileNotFoundError(
            f"No *.pt file found under MODEL_OUTPUT_PATH={model_dir!r}. "
            f"The directory contains: {os.listdir(model_dir) if model_dir and os.path.isdir(model_dir) else 'N/A'}. "
            "This typically means the training job wrote only the sidecar "
            "files (schema.json / train_config.json) for this step but did "
            "not persist model.pt — a symptom of a race between "
            "_remove_old_best_dirs and EarlyStopping.save_checkpoint."
        )
    logging.info(f"Loading checkpoint from {ckpt_path}")
    load_model_state_strict(model, ckpt_path, device)
    model.eval()
    logging.info("Model loaded successfully")

    test_loader = DataLoader(
        test_dataset,
        batch_size=None,
        num_workers=num_workers,
        prefetch_factor=2,
        pin_memory=torch.cuda.is_available(),
    )

    all_probs = []
    all_user_ids = []
    logging.info("Starting inference...")

    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            model_input = _batch_to_model_input(batch, device)
            user_ids = batch.get("user_id", [])

            logits, _ = model.predict(model_input)
            logits = logits.squeeze(-1)
            probs = torch.sigmoid(logits).cpu().numpy()
            all_probs.extend(probs.tolist())
            all_user_ids.extend(user_ids)

            if (batch_idx + 1) % 100 == 0:
                logging.info(
                    f"  Processed {(batch_idx + 1) * batch_size} samples"
                )

    logging.info(f"Inference complete: {len(all_probs)} predictions")

    predictions = {"predictions": dict(zip(all_user_ids, all_probs))}

    # ---- 保存 predictions.json ----
    output_path = os.path.join(result_dir, "predictions.json")
    with open(output_path, "w") as f:
        json.dump(predictions, f)
    logging.info(f"Saved {len(all_probs)} predictions to {output_path}")


if __name__ == "__main__":
    main()

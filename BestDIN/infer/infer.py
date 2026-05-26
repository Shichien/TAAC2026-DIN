"""DIN + MLP 推理脚本（由参赛者上传到评测容器）。

模型构造与 ``train.py`` 保持镜像一致：我们根据 ``schema.json`` + ``train_config.json`` 重建模型。所有模型超参数优先从 ckpt 目录中的 ``train_config.json`` 解析（``trainer.py`` 保存检查点时写入），然后回退到下面的 ``_FALLBACK_MODEL_CFG``（必须与 ``train.py`` 中的 CLI 默认值保持一致）。

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

from dataset import FeatureSchema, PCVRParquetDataset
from model import PCVRHyFormer, ModelInput


logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


# 仅当 ckpt 目录缺少 ``train_config.json`` 时，才使用这些回退值。
# 这些值必须与 ``train.py`` 中的 argparse 默认值保持一致；否则一旦
# 实际走到回退路径，构建出的模型形状会与保存的
# state_dict 不匹配。
_FALLBACK_MODEL_CFG = {
    "d_model": 64,
    "emb_dim": 64,
    "hidden_mult": 4,
    "dropout_rate": 0.01,
    "action_num": 1,
    "emb_skip_threshold": 1000000,
}

_FALLBACK_SEQ_MAX_LENS = "seq_a:256,seq_b:256,seq_c:512,seq_d:512"
_FALLBACK_BATCH_SIZE = 256
_FALLBACK_NUM_WORKERS = 8


# 用于构建模型的超参数键。``train_config.json`` 中的其他内容
# 在构造 DIN 模型时会被忽略。
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


def build_dense_feature_specs(
    schema: FeatureSchema,
) -> List[Tuple[int, int, int]]:
    """构造形如 ``[(fid, offset, length), ...]`` 的 dense feature layout。"""
    return [(fid, offset, length) for fid, offset, length in schema.entries]


def build_int_feature_specs_with_fid(
    schema: FeatureSchema, per_position_vocab_sizes: List[int]
) -> List[Tuple[int, int, int, int]]:
    """构造形如 ``[(fid, vocab_size, offset, length), ...]`` 的 int feature layout。"""
    specs: List[Tuple[int, int, int, int]] = []
    for fid, offset, length in schema.entries:
        vs = max(per_position_vocab_sizes[offset : offset + length])
        specs.append((fid, vs, offset, length))
    return specs


def _parse_seq_max_lens(sml_str: str) -> Dict[str, int]:
    """将形如 ``'seq_a:256,seq_b:256,...'`` 的字符串解析为 dict。"""
    seq_max_lens: Dict[str, int] = {}
    for pair in sml_str.split(","):
        k, v = pair.split(":")
        seq_max_lens[k.strip()] = int(v.strip())
    return seq_max_lens


def _parse_hash_embedding_tokens(raw: str) -> List[Tuple[int, int, int]]:
    tokens: List[Tuple[int, int, int]] = []
    raw = str(raw or "").strip()
    if not raw:
        return tokens
    for item in raw.split(","):
        token = item.strip()
        if not token:
            continue
        parts = token.split(":")
        if len(parts) != 3:
            raise ValueError(
                "hash embedding spec must be fid:H:k, for example 16:65536:4"
            )
        fid_raw, H_raw, k_raw = parts
        tokens.append((int(fid_raw), int(H_raw), int(k_raw)))
    return tokens


def build_sparse_hash_config(
    tokens: List[Tuple[int, int, int]],
    schema: FeatureSchema,
) -> Dict[int, Dict[str, Any]]:
    fid_to_spec_index = {
        int(fid): spec_index
        for spec_index, (fid, _, _) in enumerate(schema.entries)
    }
    config: Dict[int, Dict[str, Any]] = {}
    for fid, H, k in tokens:
        spec_index = fid_to_spec_index.get(int(fid))
        if spec_index is None:
            continue
        config[spec_index] = {"H": int(H), "k": int(k), "fid": int(fid)}
    return config


def parse_sparse_hash_embedding(
    raw: str,
    user_schema: FeatureSchema,
    item_schema: FeatureSchema,
) -> Tuple[Dict[int, Dict[str, Any]], Dict[int, Dict[str, Any]]]:
    tokens = _parse_hash_embedding_tokens(raw)
    if not tokens:
        return {}, {}

    user_config = build_sparse_hash_config(tokens, user_schema)
    item_config = build_sparse_hash_config(tokens, item_schema)
    matched_fids = {
        int(cfg["fid"])
        for cfg in list(user_config.values()) + list(item_config.values())
    }
    requested_fids = {fid for fid, _, _ in tokens}
    missing_fids = sorted(requested_fids - matched_fids)
    if missing_fids:
        raise ValueError(f"hash fid(s) not found in user/item int schema: {missing_fids}")
    return user_config, item_config


def parse_seq_hash_embedding(
    raw: str,
    sideinfo_fids: Dict[str, List[int]],
) -> Dict[str, Dict[int, Dict[str, Any]]]:
    result: Dict[str, Dict[int, Dict[str, Any]]] = {}
    raw = str(raw or "").strip()
    if not raw:
        return result
    for item in raw.split(","):
        token = item.strip()
        if not token:
            continue
        domain, fid_raw, H_raw, k_raw = token.split(":")
        domain = domain.strip()
        fid = int(fid_raw)
        if domain not in sideinfo_fids:
            raise ValueError(f"seq hash domain not found in schema: {domain}")
        if fid not in sideinfo_fids[domain]:
            raise ValueError(f"seq hash fid {fid} not found in {domain} sideinfo")
        slot = sideinfo_fids[domain].index(fid)
        result.setdefault(domain, {})[slot] = {
            "H": int(H_raw),
            "k": int(k_raw),
            "fid": fid,
        }
    return result


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
    """从 ``train_config`` 提取模型超参数；缺失键回退到 ``_FALLBACK_MODEL_CFG``。"""
    cfg: Dict[str, Any] = {}
    for key in _MODEL_CFG_KEYS:
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
    train_config: Dict[str, Any],
    device: str = "cpu",
) -> PCVRHyFormer:
    """根据数据集 schema 和解析后的 ``model_cfg`` dict 构造 DIN 模型。

    参数:
        dataset: 提供特征 schema 的 ``PCVRParquetDataset``。
        model_cfg: 解析后的模型超参数，通常是 ``resolve_model_cfg`` 的输出。
        device: torch 设备。
    """
    # 特征规格。
    user_int_feature_specs = build_feature_specs(
        dataset.user_int_schema, dataset.user_int_vocab_sizes
    )
    item_int_feature_specs = build_feature_specs(
        dataset.item_int_schema, dataset.item_int_vocab_sizes
    )
    user_hash_config, item_hash_config = parse_sparse_hash_embedding(
        train_config.get("hash_embedding", ""),
        dataset.user_int_schema,
        dataset.item_int_schema,
    )
    seq_hash_config = parse_seq_hash_embedding(
        train_config.get("seq_hash_embedding", ""), dataset.sideinfo_fids
    )
    if user_hash_config:
        logging.info(f"User hash config: {user_hash_config}")
    if item_hash_config:
        logging.info(f"Item hash config: {item_hash_config}")
    if seq_hash_config:
        logging.info(f"Seq hash config: {seq_hash_config}")

    logging.info(f"Building DIN + MLP with cfg: {model_cfg}")
    model = PCVRHyFormer(
        user_int_feature_specs=user_int_feature_specs,
        user_dense_feature_specs=build_dense_feature_specs(
            dataset.user_dense_schema
        ),
        user_int_pair_feature_specs=build_int_feature_specs_with_fid(
            dataset.user_int_schema, dataset.user_int_vocab_sizes
        ),
        item_int_feature_specs=item_int_feature_specs,
        item_int_feature_specs_with_fid=build_int_feature_specs_with_fid(
            dataset.item_int_schema, dataset.item_int_vocab_sizes
        ),
        user_dense_dim=dataset.user_dense_schema.total_dim,
        item_dense_dim=dataset.item_dense_schema.total_dim,
        seq_vocab_sizes=dataset.seq_domain_vocab_sizes,
        user_hash_config=user_hash_config,
        item_hash_config=item_hash_config,
        seq_hash_config=seq_hash_config,
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


def _resolve_platform_self_alias(path: str) -> str:
    """平台有时会把实际 ckpt 路径写成带 ``/self/`` 的等价路径。"""
    if os.path.exists(path):
        return path
    alias = path.replace("\\", "/").replace("/self/", "/", 1)
    if alias != path and os.path.exists(alias):
        return alias
    return path


def resolve_model_dir_and_ckpt(
    model_output_path: Optional[str],
) -> Tuple[Optional[str], Optional[str]]:
    """解析 checkpoint 叶子目录和权重文件。

    只支持训练产出的叶子目录、叶子目录的 ``/self/`` 别名，或直接的
    checkpoint 文件路径；不再向父目录递归搜索其它实验。
    """
    if not model_output_path:
        return None, None

    model_path = _resolve_platform_self_alias(model_output_path)
    if os.path.isfile(model_path):
        return os.path.dirname(model_path), model_path
    if not os.path.isdir(model_path):
        return None, None

    model_pt = os.path.join(model_path, "model.pt")
    if os.path.exists(model_pt):
        return model_path, model_pt

    for item in os.listdir(model_path):
        if item.endswith(".pt"):
            return model_path, os.path.join(model_path, item)
    return None, None


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
    for domain in seq_domains:
        seq_data[domain] = device_batch[domain]
        seq_lens[domain] = device_batch[f"{domain}_len"]

    return ModelInput(
        user_int_feats=device_batch["user_int_feats"],
        item_int_feats=device_batch["item_int_feats"],
        user_dense_feats=device_batch["user_dense_feats"],
        item_dense_feats=device_batch["item_dense_feats"],
        sample_time_feats=device_batch["sample_time_feats"],
        activity_feats=device_batch["activity_feats"],
        seq_data=seq_data,
        seq_lens=seq_lens,
    )


def main() -> None:
    # ---- 读取环境变量 ----
    model_root = os.environ.get("MODEL_OUTPUT_PATH")
    data_dir = os.environ.get("EVAL_DATA_PATH")
    result_dir = os.environ.get("EVAL_RESULT_PATH")

    os.makedirs(result_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logging.info(f"MODEL_OUTPUT_PATH raw: {model_root}")
    logging.info(f"EVAL_DATA_PATH: {data_dir}")
    logging.info(f"EVAL_RESULT_PATH: {result_dir}")
    logging.info(f"Inference device: {device}")

    model_dir, ckpt_path = resolve_model_dir_and_ckpt(model_root)
    if model_dir is None or ckpt_path is None:
        raise FileNotFoundError(
            f"Could not resolve checkpoint from MODEL_OUTPUT_PATH={model_root!r}. "
            "Expected a checkpoint leaf directory containing model.pt, "
            "a platform /self/ alias of that directory, or a direct checkpoint file."
        )
    logging.info(f"Resolved checkpoint dir: {model_dir}")
    logging.info(f"Resolved checkpoint path: {ckpt_path}")

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

    model = build_model(
        test_dataset,
        model_cfg=model_cfg,
        train_config=train_config,
        device=device,
    )

    # ---- 严格加载权重 ----
    logging.info(f"Loading checkpoint from {ckpt_path}")
    load_model_state_strict(model, ckpt_path, device)
    model.eval()
    logging.info("Model loaded successfully")

    loader_kwargs: Dict[str, Any] = {
        "batch_size": None,
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
    }
    if num_workers > 0:
        loader_kwargs["prefetch_factor"] = 2
    test_loader = DataLoader(test_dataset, **loader_kwargs)

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

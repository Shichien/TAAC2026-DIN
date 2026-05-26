"""DIN + MLP 训练入口。

用法:
    python train.py [--num_epochs 6] [--batch_size 256] ...

环境变量（优先级高于 CLI 参数）:
    TRAIN_DATA_PATH  训练数据目录（*.parquet + schema.json）
    TRAIN_CKPT_PATH  检查点输出目录
    TRAIN_LOG_PATH   日志目录
"""

import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import contextlib
import io
import argparse
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch

from utils import set_seed, EarlyStopping, create_logger
from dataset import FeatureSchema, get_pcvr_data
from model import PCVRHyFormer
from trainer import PCVRHyFormerRankingTrainer

DEFAULT_HASH_EMBEDDING = ""
DEFAULT_SEQ_HASH_EMBEDDING = (
    "seq_b:69:65536:4,seq_c:29:65536:4,seq_c:34:65536:4,seq_c:47:65536:4"
)


@contextlib.contextmanager
def _suppress_stderr_during_tensorboard_init():
    """屏蔽 TensorBoard 初始化阶段写到 stderr 的第三方底层日志。"""
    try:
        stderr_fileno = sys.stderr.fileno()
    except (AttributeError, io.UnsupportedOperation):
        with (
            open(os.devnull, "w") as devnull,
            contextlib.redirect_stderr(devnull),
        ):
            yield
        return

    sys.stderr.flush()
    saved_stderr_fileno = os.dup(stderr_fileno)
    try:
        with open(os.devnull, "w") as devnull:
            os.dup2(devnull.fileno(), stderr_fileno)
            yield
    finally:
        os.dup2(saved_stderr_fileno, stderr_fileno)
        os.close(saved_stderr_fileno)


def _create_summary_writer(log_dir: str):
    with _suppress_stderr_during_tensorboard_init():
        from torch.utils.tensorboard import SummaryWriter

        return SummaryWriter(log_dir)


def build_feature_specs(
    schema: FeatureSchema, per_position_vocab_sizes: List[int]
) -> List[Tuple[int, int, int]]:
    """构造形如 ``[(vocab_size, offset, length), ...]`` 的 feature_specs，并按 ``schema.entries`` 中记录的位置排序。"""
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
    """Parse seq hash config and convert original fid to sideinfo slot."""
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DIN + MLP Training")

    # 路径（环境变量优先）。
    parser.add_argument(
        "--data_dir",
        type=str,
        default=None,
        help="Training data directory (env: TRAIN_DATA_PATH)",
    )
    parser.add_argument(
        "--schema_path",
        type=str,
        default=None,
        help="Schema JSON path (defaults to <data_dir>/schema.json)",
    )
    parser.add_argument(
        "--ckpt_dir",
        type=str,
        default=None,
        help="Checkpoint output directory (env: TRAIN_CKPT_PATH)",
    )
    parser.add_argument(
        "--log_dir",
        type=str,
        default=None,
        help="Log directory (env: TRAIN_LOG_PATH)",
    )

    # 训练超参数。
    parser.add_argument(
        "--batch_size",
        type=int,
        default=256,
        help="Batch size for both training and validation",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="Learning rate for dense parameters",
    )
    parser.add_argument(
        "--dense_optimizer_type",
        type=str,
        default="adamw",
        choices=["adamw"],
        help="Optimizer for dense non-Embedding parameters",
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=6,
        help="Maximum number of training epochs "
        "(typically terminated earlier by early stopping)",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=999,
        help="Early-stopping patience "
        "(number of validations without improvement)",
    )
    parser.add_argument("--seed", type=int, default=47, help="Random seed")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Training device, e.g. cuda or cpu",
    )

    # 数据流水线。
    parser.add_argument(
        "--num_workers",
        type=int,
        default=16,
        help="Number of DataLoader workers",
    )
    parser.add_argument(
        "--buffer_batches",
        type=int,
        default=20,
        help="Shuffle buffer size, in units of batches. "
        "Lower values reduce memory usage.",
    )
    parser.add_argument(
        "--train_ratio",
        type=float,
        default=1.0,
        help="Fraction of training Row Groups to use (takes the first N%%)",
    )
    parser.add_argument(
        "--valid_ratio",
        type=float,
        default=0.1,
        help="Fraction of all Row Groups used for validation (takes the tail)",
    )
    parser.add_argument(
        "--split_method",
        type=str,
        default="row_group",
        choices=["row_group", "timestamp"],
        help="Validation split method: row_group = physical Row Group tail; "
        "timestamp = sample-level timestamp tail",
    )
    parser.add_argument(
        "--valid_tail_minutes",
        type=float,
        default=0.0,
        help="When split_method=timestamp, use the last N minutes as validation "
        "(0 = disabled, fall back to valid_ratio-based timestamp split)",
    )
    parser.add_argument(
        "--eval_every_n_steps",
        type=int,
        default=0,
        help="Run validation every N steps (0 = only at the end of each epoch)",
    )
    parser.add_argument(
        "--enable_block_knockout",
        action="store_true",
        default=False,
        help="Run block-level feature knockout diagnostics during validation",
    )
    parser.add_argument(
        "--block_knockout_epoch",
        type=int,
        default=3,
        help="Epoch on which to run block-level knockout diagnostics",
    )
    parser.add_argument(
        "--no_eval",
        action="store_true",
        default=False,
        help="Skip validation entirely during training and save only the final checkpoint",
    )
    parser.add_argument(
        "--save_each_epoch_ckpt",
        action="store_true",
        default=False,
        help="Save a non-best checkpoint directory after every epoch",
    )
    parser.add_argument(
        "--seq_max_lens",
        type=str,
        default="seq_a:256,seq_b:256,seq_c:512,seq_d:512",
        help="Per-domain sequence truncation, format: seq_d:256,seq_c:128",
    )

    # 模型超参数。
    parser.add_argument(
        "--d_model", type=int, default=64, help="DIN representation dimension"
    )
    parser.add_argument(
        "--emb_dim",
        type=int,
        default=64,
        help="Per-Embedding-table dimension (before projection)",
    )
    parser.add_argument(
        "--hidden_mult",
        type=int,
        default=4,
        help="MLP inner-dim multiplier relative to d_model",
    )
    parser.add_argument(
        "--dropout_rate",
        type=float,
        default=0.01,
        help="Dropout rate for DIN attention and MLP",
    )
    parser.add_argument(
        "--action_num",
        type=int,
        default=1,
        help="Classifier output dimension "
        "(1 = single binary-classification logit; >1 = multi-label)",
    )

    # 损失函数。
    parser.add_argument(
        "--loss_type",
        type=str,
        default="bce",
        choices=["bce", "focal"],
        help="Loss type: bce = BCEWithLogits, focal = Focal Loss",
    )
    parser.add_argument(
        "--focal_alpha",
        type=float,
        default=0.1,
        help="Focal Loss positive-class weight alpha "
        "(effective only when --loss_type=focal)",
    )
    parser.add_argument(
        "--focal_gamma",
        type=float,
        default=2.0,
        help="Focal Loss focusing parameter gamma "
        "(effective only when --loss_type=focal)",
    )

    # 速度优化。
    parser.add_argument(
        "--amp",
        action="store_true",
        default=False,
        help="Enable CUDA automatic mixed precision during training/evaluation",
    )
    parser.add_argument(
        "--amp_dtype",
        type=str,
        default="bfloat16",
        choices=["float16", "bfloat16"],
        help="AMP dtype when --amp is enabled",
    )
    parser.add_argument(
        "--compile_model",
        action="store_true",
        default=False,
        help="Compile the training forward path with torch.compile",
    )
    parser.add_argument(
        "--compile_mode",
        type=str,
        default="default",
        choices=["default", "reduce-overhead", "max-autotune"],
        help="torch.compile mode when --compile_model is enabled",
    )

    # 稀疏优化器。
    parser.add_argument(
        "--sparse_lr",
        type=float,
        default=0.05,
        help="Learning rate for sparse parameters (Adagrad over Embeddings)",
    )
    parser.add_argument(
        "--hash_sparse_lr",
        type=float,
        default=0.01,
        help="Learning rate for hash embedding parameters.",
    )
    parser.add_argument(
        "--sparse_weight_decay",
        type=float,
        default=0.0,
        help="Weight decay for sparse parameters (Adagrad over Embeddings)",
    )
    parser.add_argument(
        "--reinit_sparse_after_epoch",
        type=int,
        default=1,
        help="Reinitialize high-cardinality sparse embeddings after this epoch "
        "(0 = disable)",
    )
    parser.add_argument(
        "--reinit_cardinality_threshold",
        type=int,
        default=1,
        help="Reinitialize embedding tables whose vocab_size is greater than "
        "this threshold",
    )

    # Embedding 构造控制。
    parser.add_argument(
        "--emb_skip_threshold",
        type=int,
        default=0,
        help="At model construction time, features whose vocab_size "
        "exceeds this value get no Embedding and are represented "
        "by a zero vector at forward time (0 = no skipping; "
        "all features get an Embedding). Useful for saving GPU "
        "memory on ultra-high-cardinality features.",
    )
    parser.add_argument(
        "--seq_hash_embedding",
        type=str,
        default=DEFAULT_SEQ_HASH_EMBEDDING,
        help=(
            "Comma-separated seq hash specs by original fid, for example "
            "seq_b:69:65536:4,seq_c:29:65536:4"
        ),
    )
    parser.add_argument(
        "--hash_embedding",
        type=str,
        default=DEFAULT_HASH_EMBEDDING,
        help="Comma-separated sparse hash specs by original fid. Empty disables ordinary sparse hash.",
    )

    args = parser.parse_args()

    # 环境变量优先。
    args.data_dir = os.environ.get("TRAIN_DATA_PATH", args.data_dir)
    args.ckpt_dir = os.environ.get("TRAIN_CKPT_PATH", args.ckpt_dir)
    args.log_dir = os.environ.get("TRAIN_LOG_PATH", args.log_dir)
    args.tf_events_dir = os.environ.get("TRAIN_TF_EVENTS_PATH")

    return args


def main() -> None:
    args = parse_args()

    # 创建输出目录。
    Path(args.ckpt_dir).mkdir(parents=True, exist_ok=True)
    Path(args.log_dir).mkdir(parents=True, exist_ok=True)
    Path(args.tf_events_dir).mkdir(parents=True, exist_ok=True)

    # 初始化 logger 和随机数生成器。
    set_seed(args.seed)
    create_logger(os.path.join(args.log_dir, "train.log"))
    logging.info(f"Args: {vars(args)}")

    writer = _create_summary_writer(args.tf_events_dir)

    # ---- 数据加载 ----
    if args.schema_path:
        schema_path = args.schema_path
    else:
        schema_path = os.path.join(args.data_dir, "schema.json")

    if not os.path.exists(schema_path):
        raise FileNotFoundError(f"schema file not found at {schema_path}")

    # 解析各 domain 的序列长度覆盖配置。
    seq_max_lens = {}
    if args.seq_max_lens:
        for pair in args.seq_max_lens.split(","):
            k, v = pair.split(":")
            seq_max_lens[k.strip()] = int(v.strip())
        logging.info(f"Seq max_lens override: {seq_max_lens}")

    logging.info("Using Parquet data format (IterableDataset)")
    train_loader, valid_loader, pcvr_dataset = get_pcvr_data(
        data_dir=args.data_dir,
        schema_path=schema_path,
        batch_size=args.batch_size,
        valid_ratio=args.valid_ratio,
        valid_tail_minutes=args.valid_tail_minutes,
        train_ratio=args.train_ratio,
        num_workers=args.num_workers,
        buffer_batches=args.buffer_batches,
        seed=args.seed,
        seq_max_lens=seq_max_lens,
        split_method=args.split_method,
    )

    # ---- 构建模型 ----
    user_int_feature_specs = build_feature_specs(
        pcvr_dataset.user_int_schema, pcvr_dataset.user_int_vocab_sizes
    )
    item_int_feature_specs = build_feature_specs(
        pcvr_dataset.item_int_schema, pcvr_dataset.item_int_vocab_sizes
    )
    user_hash_config, item_hash_config = parse_sparse_hash_embedding(
        args.hash_embedding,
        pcvr_dataset.user_int_schema,
        pcvr_dataset.item_int_schema,
    )
    if user_hash_config:
        logging.info(f"User hash config: {user_hash_config}")
    if item_hash_config:
        logging.info(f"Item hash config: {item_hash_config}")
    seq_hash_config = parse_seq_hash_embedding(
        args.seq_hash_embedding, pcvr_dataset.sideinfo_fids
    )
    if seq_hash_config:
        logging.info(f"Seq hash config: {seq_hash_config}")

    model_args = {
        "user_int_feature_specs": user_int_feature_specs,
        "user_dense_feature_specs": build_dense_feature_specs(
            pcvr_dataset.user_dense_schema
        ),
        "user_int_pair_feature_specs": build_int_feature_specs_with_fid(
            pcvr_dataset.user_int_schema, pcvr_dataset.user_int_vocab_sizes
        ),
        "item_int_feature_specs": item_int_feature_specs,
        "item_int_feature_specs_with_fid": build_int_feature_specs_with_fid(
            pcvr_dataset.item_int_schema, pcvr_dataset.item_int_vocab_sizes
        ),
        "user_dense_dim": pcvr_dataset.user_dense_schema.total_dim,
        "item_dense_dim": pcvr_dataset.item_dense_schema.total_dim,
        "seq_vocab_sizes": pcvr_dataset.seq_domain_vocab_sizes,
        "d_model": args.d_model,
        "emb_dim": args.emb_dim,
        "hidden_mult": args.hidden_mult,
        "dropout_rate": args.dropout_rate,
        "action_num": args.action_num,
        "emb_skip_threshold": args.emb_skip_threshold,
        "user_hash_config": user_hash_config,
        "item_hash_config": item_hash_config,
        "seq_hash_config": seq_hash_config,
    }

    model = PCVRHyFormer(**model_args).to(args.device)

    # 记录模型规模信息。
    logging.info(
        f"DIN + MLP model created: seq_domains={pcvr_dataset.seq_domains}, "
        f"d_model={args.d_model}, emb_dim={args.emb_dim}"
    )
    total_params = sum(p.numel() for p in model.parameters())
    logging.info(f"Total parameters: {total_params:,}")

    # ---- 训练 ----
    early_stopping = EarlyStopping(
        checkpoint_path=os.path.join(args.ckpt_dir, "placeholder", "model.pt"),
        patience=args.patience,
        label="model",
    )

    ckpt_params = {"hidden": args.d_model}

    trainer = PCVRHyFormerRankingTrainer(
        model=model,
        train_loader=train_loader,
        valid_loader=valid_loader,
        lr=args.lr,
        num_epochs=args.num_epochs,
        device=args.device,
        save_dir=args.ckpt_dir,
        early_stopping=early_stopping,
        loss_type=args.loss_type,
        focal_alpha=args.focal_alpha,
        focal_gamma=args.focal_gamma,
        sparse_lr=args.sparse_lr,
        hash_sparse_lr=args.hash_sparse_lr,
        sparse_weight_decay=args.sparse_weight_decay,
        reinit_sparse_after_epoch=args.reinit_sparse_after_epoch,
        reinit_cardinality_threshold=args.reinit_cardinality_threshold,
        ckpt_params=ckpt_params,
        writer=writer,
        schema_path=schema_path,
        eval_every_n_steps=args.eval_every_n_steps,
        no_eval=args.no_eval,
        train_config=vars(args),
        dense_optimizer_type=args.dense_optimizer_type,
        enable_block_knockout=args.enable_block_knockout,
        block_knockout_epoch=args.block_knockout_epoch,
        save_each_epoch_ckpt=args.save_each_epoch_ckpt,
        compile_model=args.compile_model,
        compile_mode=args.compile_mode,
        amp=args.amp,
        amp_dtype=args.amp_dtype,
    )

    trainer.train()
    writer.close()

    logging.info("Training complete!")


if __name__ == "__main__":
    main()

"""PCVRHyFormer pointwise 训练器（二分类，监控 AUC）。

尽管类名历史上带有 "Ranking" 后缀，训练循环实际使用 pointwise BCE / Focal loss，并评估 Binary AUC + binary logloss。
"""

import os
import glob
import shutil
import logging
import math
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import roc_auc_score

from utils import sigmoid_focal_loss, EarlyStopping
from model import ModelInput


def _orthogonalize_update(
    update: torch.Tensor, *, steps: int, eps: float = 1e-12
) -> torch.Tensor:
    original_dtype = update.dtype
    matrix = update.float().reshape(update.shape[0], -1)
    rows, cols = matrix.shape
    transposed = rows > cols
    if transposed:
        matrix = matrix.t()

    norm = matrix.norm()
    if not torch.isfinite(norm) or norm <= eps:
        return torch.zeros_like(update)

    matrix = matrix / norm.clamp_min(eps)
    for _ in range(max(1, steps)):
        gram = matrix @ matrix.t()
        matrix = 1.5 * matrix - 0.5 * gram @ matrix

    if transposed:
        matrix = matrix.t()
    return matrix.reshape_as(update).to(original_dtype)


class Muon(torch.optim.Optimizer):
    """Dense optimizer: matrix parameters use Muon, vectors use AdamW."""

    def __init__(
        self,
        params,
        *,
        lr: float,
        momentum: float = 0.95,
        nesterov: bool = True,
        ns_steps: int = 5,
        weight_decay: float = 0.0,
        adamw_betas: Tuple[float, float] = (0.9, 0.98),
        adamw_eps: float = 1e-8,
    ) -> None:
        defaults = {
            "lr": float(lr),
            "momentum": float(momentum),
            "nesterov": bool(nesterov),
            "ns_steps": int(ns_steps),
            "weight_decay": float(weight_decay),
            "adamw_betas": (float(adamw_betas[0]), float(adamw_betas[1])),
            "adamw_eps": float(adamw_eps),
        }
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):  # type: ignore[override]
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = float(group["lr"])
            momentum = float(group["momentum"])
            nesterov = bool(group["nesterov"])
            ns_steps = int(group["ns_steps"])
            weight_decay = float(group["weight_decay"])
            beta1, beta2 = group["adamw_betas"]
            adamw_eps = float(group["adamw_eps"])

            for parameter in group["params"]:
                grad = parameter.grad
                if grad is None:
                    continue
                if grad.is_sparse:
                    raise RuntimeError("Muon does not support sparse gradients")

                if parameter.ndim >= 2:
                    state = self.state[parameter]
                    momentum_buffer = state.setdefault(
                        "momentum_buffer", torch.zeros_like(parameter)
                    )
                    momentum_buffer.mul_(momentum).add_(grad)
                    update = (
                        grad.add(momentum_buffer, alpha=momentum)
                        if nesterov
                        else momentum_buffer
                    )
                    matrix = update.reshape(update.shape[0], -1)
                    update_scale = math.sqrt(
                        max(1.0, matrix.shape[0] / max(1, matrix.shape[1]))
                    )
                    orthogonal_update = _orthogonalize_update(
                        update, steps=ns_steps
                    )
                    if weight_decay != 0.0:
                        parameter.mul_(1.0 - lr * weight_decay)
                    parameter.add_(orthogonal_update, alpha=-lr * update_scale)
                    continue

                state = self.state[parameter]
                if not state:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(parameter)
                    state["exp_avg_sq"] = torch.zeros_like(parameter)

                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]
                state["step"] += 1
                step = int(state["step"])

                if weight_decay != 0.0:
                    parameter.mul_(1.0 - lr * weight_decay)

                exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)
                bias_correction1 = 1.0 - beta1**step
                bias_correction2 = 1.0 - beta2**step
                denom = (
                    exp_avg_sq.sqrt()
                    .div_(math.sqrt(bias_correction2))
                    .add_(adamw_eps)
                )
                parameter.addcdiv_(
                    exp_avg, denom, value=-(lr / bias_correction1)
                )

        return loss


def build_dense_optimizer(
    dense_params, *, dense_optimizer_type: str, lr: float
) -> torch.optim.Optimizer:
    if dense_optimizer_type == "muon":
        return Muon(
            dense_params, lr=lr, weight_decay=0.0, adamw_betas=(0.9, 0.98)
        )
    if dense_optimizer_type == "adamw":
        return torch.optim.AdamW(
            dense_params, lr=lr, betas=(0.9, 0.98), foreach=False
        )
    raise ValueError(
        f"dense_optimizer_type must be 'adamw' or 'muon', got {dense_optimizer_type!r}"
    )


def _progress_miniters(total: int, chunks: int = 5) -> int:
    """Return a fixed tqdm refresh interval so Taiji logs stay readable."""
    return max(1, (total + chunks - 1) // chunks)


class PCVRHyFormerRankingTrainer:
    """用于 pointwise 二分类的 PCVRHyFormer 训练器。

    使用 PCVR 数据布局:
    - user_int_feats, user_dense_feats
    - item_int_feats, item_dense_feats
    - seq_a, seq_b, seq_c, seq_d（每个都有 *_len 配套字段）
    - label（二分类）

    损失：BCEWithLogitsLoss 或 Focal Loss。
    指标：BinaryAUROC + binary logloss。
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        valid_loader: DataLoader,
        lr: float,
        num_epochs: int,
        device: str,
        save_dir: str,
        early_stopping: EarlyStopping,
        loss_type: str = "bce",
        focal_alpha: float = 0.1,
        focal_gamma: float = 2.0,
        sparse_lr: float = 0.05,
        sparse_weight_decay: float = 0.0,
        reinit_sparse_after_epoch: int = 1,
        reinit_cardinality_threshold: int = 0,
        ckpt_params: Optional[Dict[str, Any]] = None,
        writer: Optional[Any] = None,
        schema_path: Optional[str] = None,
        ns_groups_path: Optional[str] = None,
        eval_every_n_steps: int = 0,
        no_eval: bool = False,
        train_config: Optional[Dict[str, Any]] = None,
        dense_optimizer_type: str = "adamw",
        amp: bool = False,
        amp_dtype: str = "bfloat16",
        compile_model: bool = False,
        compile_mode: str = "default",
    ) -> None:
        self.model: nn.Module = model
        self.train_loader: DataLoader = train_loader
        self.valid_loader: DataLoader = valid_loader
        self.writer = writer
        # schema_path 会随每个检查点一起复制，使 infer.py 能
        # 重建与模型训练时完全一致的特征 schema。
        self.schema_path: Optional[str] = schema_path
        # ns_groups_path 是可选的；当提供且指向现有文件时，
        # 会复制到 schema.json 旁边。把该 JSON 保存在 ckpt 目录内，
        # 可让检查点在不单独携带 ns_groups.json 的评测环境中
        # 自包含。
        self.ns_groups_path: Optional[str] = ns_groups_path

        # 双优化器：稀疏 Embedding 使用 Adagrad，dense 参数可切换 AdamW / Muon。
        self.sparse_optimizer: Optional[torch.optim.Optimizer]
        if hasattr(model, "get_sparse_params"):
            sparse_params = model.get_sparse_params()
            dense_params = model.get_dense_params()
            sparse_param_count = sum(p.numel() for p in sparse_params)
            dense_param_count = sum(p.numel() for p in dense_params)
            logging.info(
                f"Sparse params: {len(sparse_params)} tensors, {sparse_param_count:,} parameters (Adagrad lr={sparse_lr})"
            )
            logging.info(
                f"Dense params: {len(dense_params)} tensors, {dense_param_count:,} parameters ({dense_optimizer_type} lr={lr})"
            )
            self.sparse_optimizer = torch.optim.Adagrad(
                sparse_params,
                lr=sparse_lr,
                weight_decay=sparse_weight_decay,
                foreach=False,
            )
            self.dense_optimizer: torch.optim.Optimizer = build_dense_optimizer(
                dense_params, dense_optimizer_type=dense_optimizer_type, lr=lr
            )
        else:
            self.sparse_optimizer = None
            self.dense_optimizer = build_dense_optimizer(
                list(model.parameters()),
                dense_optimizer_type=dense_optimizer_type,
                lr=lr,
            )

        self.num_epochs: int = num_epochs
        self.device: str = device
        self.save_dir: str = save_dir
        self.early_stopping: EarlyStopping = early_stopping
        self.loss_type: str = loss_type
        self.focal_alpha: float = focal_alpha
        self.focal_gamma: float = focal_gamma
        self.reinit_sparse_after_epoch: int = reinit_sparse_after_epoch
        self.reinit_cardinality_threshold: int = reinit_cardinality_threshold
        self.sparse_lr: float = sparse_lr
        self.sparse_weight_decay: float = sparse_weight_decay
        self.ckpt_params: Dict[str, Any] = ckpt_params or {}
        self.eval_every_n_steps: int = eval_every_n_steps
        self.no_eval: bool = bool(no_eval)
        self.train_config: Optional[Dict[str, Any]] = train_config

        self.amp_enabled = bool(amp and str(device).startswith("cuda"))
        if amp_dtype == "float16":
            self.amp_dtype = torch.float16
        elif amp_dtype == "bfloat16":
            self.amp_dtype = torch.bfloat16
        else:
            raise ValueError(
                f"amp_dtype must be 'float16' or 'bfloat16', got {amp_dtype!r}"
            )
        self.grad_scaler = torch.cuda.amp.GradScaler(
            enabled=self.amp_enabled and self.amp_dtype == torch.float16
        )

        self.compiled_model: Optional[nn.Module] = None
        if compile_model:
            if not hasattr(torch, "compile"):
                raise RuntimeError(
                    "--compile_model requires torch.compile, but this PyTorch build does not expose it"
                )
            self.compiled_model = torch.compile(self.model, mode=compile_mode)

        logging.info(
            f"PCVRHyFormerRankingTrainer loss_type={loss_type}, "
            f"focal_alpha={focal_alpha}, focal_gamma={focal_gamma}, "
            f"reinit_sparse_after_epoch={reinit_sparse_after_epoch}"
        )

    def _build_step_dir_name(
        self, global_step: int, is_best: bool = False
    ) -> str:
        """构造检查点子目录名，例如 ``global_step2500.layer=2.head=4.hidden=64[.best_model]``。"""
        parts = [f"global_step{global_step}"]
        for key in ("layer", "head", "hidden"):
            if key in self.ckpt_params:
                parts.append(f"{key}={self.ckpt_params[key]}")
        name = ".".join(parts)
        if is_best:
            name += ".best_model"
        return name

    def _write_sidecar_files(self, ckpt_dir: str) -> None:
        """在 ``model.pt`` 旁写入配套文件。

        目前最多持久化三个文件，每次调用都会覆盖:

        - ``schema.json``（从 ``self.schema_path`` 复制）：重建 Parquet 数据集所需的特征布局元数据。
        - ``ns_groups.json``（当设置了 ``self.ns_groups_path`` 且文件存在时复制）：用于构造 tokenizer 的 NS-token 分组。为每个 ckpt 复制一份，使评测环境无需携带原项目级 ``ns_groups.json`` 也能消费检查点。
        - ``train_config.json``（由 ``self.train_config`` 序列化）：训练时完整超参数集合。当 ``ns_groups.json`` 被复制进 ``ckpt_dir`` 时，``ns_groups_json`` 字段会重写为裸文件名，使 ``infer.py`` 相对 ``ckpt_dir`` 解析，而不是指向训练机器上的原始绝对路径。
        """
        os.makedirs(ckpt_dir, exist_ok=True)
        if self.schema_path and os.path.exists(self.schema_path):
            shutil.copy2(self.schema_path, ckpt_dir)

        ns_groups_copied = False
        if self.ns_groups_path and os.path.exists(self.ns_groups_path):
            shutil.copy2(self.ns_groups_path, ckpt_dir)
            ns_groups_copied = True

        if self.train_config:
            import json

            cfg_to_dump = self.train_config
            if ns_groups_copied:
                # 将存储路径改写为相对 ckpt_dir 的文件名；
                # 当记录路径不是绝对路径时，infer.py 已会回退到 `<ckpt_dir>/<basename>`，
                # 这样可以保持 ckpt
                # 在不同主机之间可移植。
                cfg_to_dump = dict(self.train_config)
                cfg_to_dump["ns_groups_json"] = os.path.basename(
                    self.ns_groups_path
                )
            with open(os.path.join(ckpt_dir, "train_config.json"), "w") as f:
                json.dump(cfg_to_dump, f, indent=2)

    def _save_step_checkpoint(
        self,
        global_step: int,
        is_best: bool = False,
        skip_model_file: bool = False,
    ) -> str:
        """在 ``global_step`` 子目录下保存 ``model.pt`` 和配套文件。

        参数:
            global_step: 当前 global step，用于目录命名。
            is_best: 这是否是新的最佳检查点。
            skip_model_file: 若为 True，则跳过写入 ``model.pt``（因为调用方如 EarlyStopping 已经将其持久化到相同路径）。配套文件仍会被（重新）写入。

        返回:
            检查点目录的绝对路径。
        """
        dir_name = self._build_step_dir_name(global_step, is_best=is_best)
        ckpt_dir = os.path.join(self.save_dir, dir_name)
        os.makedirs(ckpt_dir, exist_ok=True)
        if not skip_model_file:
            torch.save(
                self.model.state_dict(), os.path.join(ckpt_dir, "model.pt")
            )
        self._write_sidecar_files(ckpt_dir)
        logging.info(f"Saved checkpoint to {ckpt_dir}/model.pt")
        return ckpt_dir

    def _remove_old_best_dirs(self) -> None:
        """删除过期的 ``*.best_model`` 目录，确保磁盘上只保留最新的最佳检查点。"""
        pattern = os.path.join(self.save_dir, "global_step*.best_model")
        for old_dir in glob.glob(pattern):
            shutil.rmtree(old_dir)
            logging.info(f"Removed old best_model dir: {old_dir}")

    def _batch_to_device(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """将 ``batch`` 中所有张量移动到 ``self.device``（``non_blocking=True``，以配合 ``pin_memory``）。非张量值原样透传。"""
        device_batch: Dict[str, Any] = {}
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                device_batch[k] = v.to(self.device, non_blocking=True)
            else:
                device_batch[k] = v
        return device_batch

    def _handle_validation_result(
        self, total_step: int, val_auc: float, val_logloss: float
    ) -> None:
        """以原子化流程持久化新的最佳检查点。

        流程（有序执行，避免在磁盘上留下只有配套文件的空目录）:

        1. 使用与 ``EarlyStopping._is_not_improved`` 相同的阈值，判断 ``val_auc`` 是否“可能”超过当前最佳值，从而让预清理与 EarlyStopping 内部保存决策保持同步。
        2. 如果不可能提升，则短路：不触碰磁盘。不得修改 ``self.early_stopping.checkpoint_path`` 或调用 ``_write_sidecar_files``，因为目标目录可能还不存在（否则会创建只有配套文件、缺少 ``model.pt`` 的目录）。
        3. 如果可能提升，则将 ``EarlyStopping`` 指向规范的 ``global_stepN.best_model/model.pt`` 路径，移除过期的 ``*.best_model`` 目录，然后运行 ``EarlyStopping``（它在实际确认新最佳时写入 ``model.pt``）。
        4. 只有在 ``EarlyStopping`` 确认新最佳（``best_score != old_best``）之后，才把配套文件写入新建目录；这里会加保护，避免非常接近的分数触发 ``is_likely_new_best`` 但未通过 ``EarlyStopping`` 自身门槛时留下多余目录。
        """
        old_best = self.early_stopping.best_score
        is_likely_new_best = (
            old_best is None or val_auc > old_best + self.early_stopping.delta
        )
        if not is_likely_new_best:
            # 预计不会产生新的最佳模型：不触碰磁盘。之前的
            # best_model 目录（含 model.pt 和配套文件）仍然有效。
            self.early_stopping(
                val_auc,
                self.model,
                {"best_val_AUC": val_auc, "best_val_logloss": val_logloss},
            )
            return

        # 将 EarlyStopping 指向当前 step 的规范 best-model 位置。
        # 只在可能产生新最佳的分支执行，避免跳过保存时
        # 未使用路径泄漏到 EarlyStopping 状态中。
        best_dir = os.path.join(
            self.save_dir, self._build_step_dir_name(total_step, is_best=True)
        )
        self.early_stopping.checkpoint_path = os.path.join(best_dir, "model.pt")

        # 先移除过期 best 目录，使 EarlyStopping 的写入成为
        # 确认新最佳时唯一需要的 I/O。
        self._remove_old_best_dirs()

        self.early_stopping(
            val_auc,
            self.model,
            {"best_val_AUC": val_auc, "best_val_logloss": val_logloss},
        )

        # 仅当 EarlyStopping 实际确认新的最佳模型，
        # 且写入 model.pt 后，才写入配套文件。如果分数触发了启发式判断，
        # 但 EarlyStopping 内部拒绝保存，则跳过以避免
        # 创建空的（只有配套文件的）检查点目录。
        if self.early_stopping.best_score != old_best and os.path.exists(
            self.early_stopping.checkpoint_path
        ):
            self._save_step_checkpoint(
                total_step, is_best=True, skip_model_file=True
            )

    def train(self) -> None:
        """主训练循环：遍历 epoch，执行 step 级和 epoch 级验证，触发 EarlyStopping 以及周期性稀疏参数重新初始化策略。"""
        print("Start training (PCVRHyFormer)")
        self.model.train()
        total_step = 0

        for epoch in range(1, self.num_epochs + 1):
            train_total = len(self.train_loader)
            train_pbar = tqdm(
                enumerate(self.train_loader),
                total=train_total,
                dynamic_ncols=True,
                miniters=_progress_miniters(train_total),
                mininterval=0.0,
            )
            loss_sum = 0.0

            for step, batch in train_pbar:
                loss = self._train_step(batch)
                total_step += 1
                loss_sum += loss

                if self.writer:
                    self.writer.add_scalar("Loss/train", loss, total_step)

                train_pbar.set_postfix({"loss": f"{loss:.4f}"}, refresh=False)

                # step 级验证（仅当 eval_every_n_steps > 0 时）。
                if (
                    (not self.no_eval)
                    and self.eval_every_n_steps > 0
                    and total_step % self.eval_every_n_steps == 0
                ):
                    logging.info(f"Evaluating at step {total_step}")
                    val_auc, val_logloss = self.evaluate(epoch=epoch)
                    self.model.train()
                    torch.cuda.empty_cache()

                    logging.info(
                        f"Step {total_step} Validation | AUC: {val_auc}, LogLoss: {val_logloss}"
                    )

                    if self.writer:
                        self.writer.add_scalar("AUC/valid", val_auc, total_step)
                        self.writer.add_scalar(
                            "LogLoss/valid", val_logloss, total_step
                        )

                    self._handle_validation_result(
                        total_step, val_auc, val_logloss
                    )

                    if self.early_stopping.early_stop:
                        logging.info(f"Early stopping at step {total_step}")
                        return

            logging.info(
                f"Epoch {epoch}, Average Loss: {loss_sum / len(self.train_loader)}"
            )

            if self.no_eval:
                self._save_step_checkpoint(total_step, is_best=False)
            else:
                val_auc, val_logloss = self.evaluate(epoch=epoch)
                self.model.train()
                torch.cuda.empty_cache()

                logging.info(
                    f"Epoch {epoch} Validation | AUC: {val_auc}, LogLoss: {val_logloss}"
                )

                if self.writer:
                    self.writer.add_scalar("AUC/valid", val_auc, total_step)
                    self.writer.add_scalar(
                        "LogLoss/valid", val_logloss, total_step
                    )

                self._handle_validation_result(total_step, val_auc, val_logloss)

                if self.early_stopping.early_stop:
                    logging.info(f"Early stopping at epoch {epoch}")
                    break

            # 达到配置的 epoch 后，重新初始化高基数稀疏
            # 参数（Embedding），作为降低过拟合的冷重启形式。
            # 参考：KuaiShou Tech.，《MultiEpoch: Reusing Training Data
            # for Click-Through Rate Prediction》，
            # https://arxiv.org/pdf/2305.19531
            if (
                epoch >= self.reinit_sparse_after_epoch
                and self.sparse_optimizer is not None
            ):
                # 通过 data_ptr 按参数快照 Adagrad 状态，使低基数
                # Embedding 的状态可在重建后保留。
                old_state: Dict[int, Any] = {}
                for group in self.sparse_optimizer.param_groups:
                    for p in group["params"]:
                        if p.data_ptr() in self.sparse_optimizer.state:
                            old_state[p.data_ptr()] = (
                                self.sparse_optimizer.state[p]
                            )

                reinit_ptrs = self.model.reinit_high_cardinality_params(
                    self.reinit_cardinality_threshold
                )
                sparse_params = self.model.get_sparse_params()
                self.sparse_optimizer = torch.optim.Adagrad(
                    sparse_params,
                    lr=self.sparse_lr,
                    weight_decay=self.sparse_weight_decay,
                    foreach=False,
                )
                # 只恢复低基数 Embedding 的优化器状态。
                restored = 0
                for p in sparse_params:
                    if (
                        p.data_ptr() not in reinit_ptrs
                        and p.data_ptr() in old_state
                    ):
                        self.sparse_optimizer.state[p] = old_state[p.data_ptr()]
                        restored += 1
                logging.info(
                    f"Rebuilt Adagrad optimizer after epoch {epoch}, "
                    f"restored optimizer state for {restored} low-cardinality params"
                )

        if self.no_eval:
            logging.info(
                "No-eval training complete, saving final checkpoint to save_dir root"
            )
            torch.save(
                self.model.state_dict(), os.path.join(self.save_dir, "model.pt")
            )
            self._write_sidecar_files(self.save_dir)

    def _make_model_input(self, device_batch: Dict[str, Any]) -> ModelInput:
        """从 device_batch dict 构造 ``ModelInput`` NamedTuple。"""
        seq_domains = device_batch["_seq_domains"]
        seq_data: Dict[str, torch.Tensor] = {}
        seq_lens: Dict[str, torch.Tensor] = {}
        seq_time_buckets: Dict[str, torch.Tensor] = {}
        for domain in seq_domains:
            seq_data[domain] = device_batch[domain]
            seq_lens[domain] = device_batch[f"{domain}_len"]
            B = device_batch[domain].shape[0]
            L = device_batch[domain].shape[2]
            seq_time_buckets[domain] = device_batch.get(
                f"{domain}_time_bucket",
                torch.zeros(B, L, dtype=torch.long, device=self.device),
            )
        return ModelInput(
            user_int_feats=device_batch["user_int_feats"],
            item_int_feats=device_batch["item_int_feats"],
            user_dense_feats=device_batch["user_dense_feats"],
            item_dense_feats=device_batch["item_dense_feats"],
            seq_data=seq_data,
            seq_lens=seq_lens,
            seq_time_buckets=seq_time_buckets,
            sample_timestamps=device_batch["timestamp"],
        )

    def _forward_train(self, model_input: ModelInput) -> torch.Tensor:
        model = (
            self.compiled_model
            if self.compiled_model is not None
            else self.model
        )
        return model(model_input)

    def _autocast_context(self):
        return torch.cuda.amp.autocast(
            enabled=self.amp_enabled, dtype=self.amp_dtype
        )

    def _train_step(self, batch: Dict[str, Any]) -> float:
        """执行单个训练 step，并返回标量 loss 值。"""
        device_batch = self._batch_to_device(batch)
        label = device_batch["label"].float()

        self.dense_optimizer.zero_grad()
        if self.sparse_optimizer is not None:
            self.sparse_optimizer.zero_grad()

        model_input = self._make_model_input(device_batch)
        with self._autocast_context():
            logits = self._forward_train(model_input)  # (B, 1)
            logits = logits.squeeze(-1)  # (B,)

            if self.loss_type == "focal":
                loss = sigmoid_focal_loss(
                    logits,
                    label,
                    alpha=self.focal_alpha,
                    gamma=self.focal_gamma,
                )
            else:
                loss = F.binary_cross_entropy_with_logits(logits, label)

        if self.grad_scaler.is_enabled():
            self.grad_scaler.scale(loss).backward()
        else:
            loss.backward()
        # foreach=False：规避本项目中某些张量形状触发的
        # PyTorch _foreach_norm CUDA kernel bug。
        torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), max_norm=1.0, foreach=False
        )

        self.dense_optimizer.step()
        if self.sparse_optimizer is not None:
            self.sparse_optimizer.step()

        return loss.item()

    def evaluate(self, epoch: Optional[int] = None) -> Tuple[float, float]:
        """在 ``self.valid_loader`` 上执行验证，并返回 ``(AUC, logloss)``。

        计算两个指标前会过滤 NaN 预测值（梯度爆炸时可能出现）。
        """
        print("Start Evaluation (PCVRHyFormer) - validation")
        self.model.eval()
        if not epoch:
            epoch = -1

        valid_total = len(self.valid_loader)
        pbar = tqdm(
            enumerate(self.valid_loader),
            total=valid_total,
            miniters=_progress_miniters(valid_total),
            mininterval=0.0,
        )

        all_logits_list = []
        all_labels_list = []

        with torch.no_grad():
            for step, batch in pbar:
                logits, labels = self._evaluate_step(batch)
                all_logits_list.append(logits.detach().cpu())
                all_labels_list.append(labels.detach().cpu())

        all_logits = torch.cat(all_logits_list, dim=0).float()
        all_labels = torch.cat(all_labels_list, dim=0).long()

        # 通过 sklearn 计算二分类 AUC。
        probs = torch.sigmoid(all_logits).float().numpy()
        labels_np = all_labels.numpy()

        # 过滤 NaN 预测值（梯度爆炸时可能出现）。
        nan_mask = np.isnan(probs)
        if nan_mask.any():
            n_nan = int(nan_mask.sum())
            logging.warning(
                f"[Evaluate] {n_nan}/{len(probs)} predictions are NaN, filtering them out"
            )
            valid_mask = ~nan_mask
            probs = probs[valid_mask]
            labels_np = labels_np[valid_mask]

        if len(probs) == 0 or len(np.unique(labels_np)) < 2:
            auc = 0.0
        else:
            auc = float(roc_auc_score(labels_np, probs))

        # 二分类 logloss（使用同样的 NaN 过滤）。
        valid_logits = all_logits[~torch.isnan(all_logits)]
        valid_labels = all_labels[~torch.isnan(all_logits)]
        if len(valid_logits) > 0:
            logloss = F.binary_cross_entropy_with_logits(
                valid_logits, valid_labels.float()
            ).item()
        else:
            logloss = float("inf")

        return auc, logloss

    def _evaluate_step(
        self, batch: Dict[str, Any]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """执行单个验证 step，并返回 ``(logits, labels)``。"""
        device_batch = self._batch_to_device(batch)
        label = device_batch["label"]

        model_input = self._make_model_input(device_batch)
        with self._autocast_context():
            logits, _ = self.model.predict(model_input)  # (B, 1), (B, D)
        logits = logits.squeeze(-1)  # (B,)

        return logits, label

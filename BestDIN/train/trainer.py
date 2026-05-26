"""DIN pointwise 训练器（二分类，监控 AUC）。

尽管类名历史上带有 "Ranking" 后缀，训练循环实际使用 pointwise BCE / Focal loss，并评估 Binary AUC、binary logloss 以及概率校准辅助指标。
"""

import os
import glob
import shutil
import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score

from utils import sigmoid_focal_loss, EarlyStopping
from model import ModelInput


def build_dense_optimizer(
    dense_params,
    *,
    dense_optimizer_type: str,
    lr: float,
) -> torch.optim.Optimizer:
    dense_params = list(dense_params)
    if dense_optimizer_type == 'adamw':
        return torch.optim.AdamW(dense_params, lr=lr, betas=(0.9, 0.98), foreach=False)
    raise ValueError(f"dense_optimizer_type must be 'adamw', got {dense_optimizer_type!r}")


def _should_log_progress(current: int, total: int, chunks: int = 5) -> bool:
    if total <= 0:
        return False
    if current >= total:
        return True
    prev_bucket = (max(0, current - 1) * chunks) // total
    curr_bucket = (current * chunks) // total
    return curr_bucket > prev_bucket


class PCVRHyFormerRankingTrainer:
    """用于 pointwise 二分类的 DIN 训练器。

    使用 PCVR 数据布局:
    - user_int_feats, user_dense_feats
    - item_int_feats, item_dense_feats
    - seq_a, seq_b, seq_c, seq_d（每个都有 *_len 配套字段）
    - label（二分类）

    损失：BCEWithLogitsLoss 或 Focal Loss。
    指标：BinaryAUROC、binary logloss，以及用于观察概率校准的 PCOC / Brier Score。
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
        loss_type: str = 'bce',
        focal_alpha: float = 0.1,
        focal_gamma: float = 2.0,
        sparse_lr: float = 0.05,
        hash_sparse_lr: float = 0.01,
        sparse_weight_decay: float = 0.0,
        reinit_sparse_after_epoch: int = 1,
        reinit_cardinality_threshold: int = 1,
        ckpt_params: Optional[Dict[str, Any]] = None,
        writer: Optional[Any] = None,
        schema_path: Optional[str] = None,
        eval_every_n_steps: int = 0,
        no_eval: bool = False,
        train_config: Optional[Dict[str, Any]] = None,
        dense_optimizer_type: str = 'adamw',
        enable_block_knockout: bool = False,
        block_knockout_epoch: int = 3,
        stop_after_global_step: int = 0,
        run_name: str = '',
        persist_best_checkpoint: bool = True,
        save_each_epoch_ckpt: bool = False,

        amp: bool = False,
        amp_dtype: str = 'bfloat16',
        compile_model: bool = False,
        compile_mode: str = 'default',
    ) -> None:
        self.model: nn.Module = model
        self.train_loader: DataLoader = train_loader
        self.valid_loader: DataLoader = valid_loader
        self.writer = writer
        # schema_path 会随每个检查点一起复制，使 infer.py 能
        # 重建与模型训练时完全一致的特征 schema。
        self.schema_path: Optional[str] = schema_path

        # 双优化器：稀疏 Embedding 使用 Adagrad，dense 参数使用 AdamW。
        self.sparse_optimizer: Optional[torch.optim.Optimizer]
        self.hash_sparse_optimizer: Optional[torch.optim.Optimizer]
        if hasattr(model, 'get_sparse_params'):
            if hasattr(model, 'get_non_hash_sparse_params'):
                sparse_params = model.get_non_hash_sparse_params()
            else:
                sparse_params = model.get_sparse_params()
            if hasattr(model, 'get_hash_sparse_params'):
                hash_sparse_params = model.get_hash_sparse_params()
            else:
                hash_sparse_params = []
            dense_params = model.get_dense_params()
            sparse_param_count = sum(p.numel() for p in sparse_params)
            hash_sparse_param_count = sum(p.numel() for p in hash_sparse_params)
            dense_param_count = sum(p.numel() for p in dense_params)
            logging.info(f"Sparse params: {len(sparse_params)} tensors, {sparse_param_count:,} parameters (Adagrad lr={sparse_lr})")
            logging.info(f"Hash sparse params: {len(hash_sparse_params)} tensors, {hash_sparse_param_count:,} parameters (Adagrad lr={hash_sparse_lr})")
            logging.info(f"Dense params: {len(dense_params)} tensors, {dense_param_count:,} parameters ({dense_optimizer_type} lr={lr})")
            self.sparse_optimizer = (
                torch.optim.Adagrad(
                    sparse_params,
                    lr=sparse_lr,
                    weight_decay=sparse_weight_decay,
                    foreach=False,
                )
                if sparse_params
                else None
            )
            self.hash_sparse_optimizer = (
                torch.optim.Adagrad(
                    hash_sparse_params,
                    lr=hash_sparse_lr,
                    weight_decay=sparse_weight_decay,
                    foreach=False,
                )
                if hash_sparse_params
                else None
            )
            self.dense_optimizer: torch.optim.Optimizer = build_dense_optimizer(
                dense_params, dense_optimizer_type=dense_optimizer_type, lr=lr
            )
        else:
            self.sparse_optimizer = None
            self.hash_sparse_optimizer = None
            self.dense_optimizer = build_dense_optimizer(
                list(model.parameters()), dense_optimizer_type=dense_optimizer_type, lr=lr
            )

        self.num_epochs: int = num_epochs
        self.device: str = device
        self.save_dir: str = save_dir
        self.early_stopping: EarlyStopping = early_stopping
        self.loss_type: str = loss_type
        self.focal_alpha: float = focal_alpha
        self.focal_gamma: float = focal_gamma
        self.sparse_lr: float = sparse_lr
        self.hash_sparse_lr: float = hash_sparse_lr
        self.sparse_weight_decay: float = sparse_weight_decay
        self.reinit_sparse_after_epoch: int = int(reinit_sparse_after_epoch)
        self.reinit_cardinality_threshold: int = int(reinit_cardinality_threshold)
        self.ckpt_params: Dict[str, Any] = ckpt_params or {}
        self.eval_every_n_steps: int = eval_every_n_steps
        self.no_eval: bool = bool(no_eval)
        self.train_config: Optional[Dict[str, Any]] = train_config
        self.enable_block_knockout: bool = bool(enable_block_knockout)
        self.block_knockout_epoch: int = int(block_knockout_epoch)
        self.stop_after_global_step: int = int(stop_after_global_step)
        self.run_name: str = str(run_name)
        self.persist_best_checkpoint: bool = bool(persist_best_checkpoint)
        self.save_each_epoch_ckpt: bool = bool(save_each_epoch_ckpt)
        self.best_epoch: Optional[int] = None
        self.best_global_step: Optional[int] = None


        self.amp_enabled = bool(amp and str(device).startswith('cuda'))
        if amp_dtype == 'float16':
            self.amp_dtype = torch.float16
        elif amp_dtype == 'bfloat16':
            self.amp_dtype = torch.bfloat16
        else:
            raise ValueError(f"amp_dtype must be 'float16' or 'bfloat16', got {amp_dtype!r}")
        self.amp_device_type = 'cuda' if str(device).startswith('cuda') else 'cpu'
        self.grad_scaler = torch.amp.GradScaler(
            self.amp_device_type,
            enabled=self.amp_enabled and self.amp_dtype == torch.float16
        )

        self.compiled_model: Optional[nn.Module] = None
        if compile_model:
            if not hasattr(torch, 'compile'):
                raise RuntimeError("--compile_model requires torch.compile, but this PyTorch build does not expose it")
            self.compiled_model = torch.compile(
                self.model,
                mode=compile_mode,
                dynamic=False,
            )

        logging.info(f"DINRankingTrainer loss_type={loss_type}, "
                     f"focal_alpha={focal_alpha}, focal_gamma={focal_gamma}, "
                     f"persist_best_checkpoint={self.persist_best_checkpoint}, "
                     f"save_each_epoch_ckpt={self.save_each_epoch_ckpt}, "
                     f"enable_block_knockout={self.enable_block_knockout}, "
                     f"block_knockout_epoch={self.block_knockout_epoch}, "
                     f"reinit_sparse_after_epoch={self.reinit_sparse_after_epoch}, "
                     f"reinit_cardinality_threshold={self.reinit_cardinality_threshold}, "
                     f"hash_sparse_lr={self.hash_sparse_lr}")

    def _reinitialize_sparse_embeddings(self, epoch: int) -> None:
        if self.sparse_optimizer is None and self.hash_sparse_optimizer is None:
            return
        if not hasattr(self.model, 'reinit_high_cardinality_params'):
            raise RuntimeError(
                'sparse embedding reinit requires model.reinit_high_cardinality_params')

        old_state: Dict[int, Any] = {}
        for optimizer in (self.sparse_optimizer, self.hash_sparse_optimizer):
            if optimizer is None:
                continue
            for parameter, state in optimizer.state.items():
                old_state[parameter.data_ptr()] = state

        reinit_ptrs = self.model.reinit_high_cardinality_params(
            self.reinit_cardinality_threshold)
        if hasattr(self.model, 'get_non_hash_sparse_params'):
            sparse_params = self.model.get_non_hash_sparse_params()
        else:
            sparse_params = self.model.get_sparse_params()
        if hasattr(self.model, 'get_hash_sparse_params'):
            hash_sparse_params = self.model.get_hash_sparse_params()
        else:
            hash_sparse_params = []
        self.sparse_optimizer = torch.optim.Adagrad(
            sparse_params,
            lr=self.sparse_lr,
            weight_decay=self.sparse_weight_decay,
            foreach=False,
        ) if sparse_params else None
        self.hash_sparse_optimizer = torch.optim.Adagrad(
            hash_sparse_params,
            lr=self.hash_sparse_lr,
            weight_decay=self.sparse_weight_decay,
            foreach=False,
        ) if hash_sparse_params else None

        restored_sparse = 0
        reinitialized_sparse = 0
        for parameter in sparse_params:
            ptr = parameter.data_ptr()
            if ptr in reinit_ptrs:
                reinitialized_sparse += 1
                continue
            if ptr not in old_state:
                continue
            if self.sparse_optimizer is not None:
                self.sparse_optimizer.state[parameter] = old_state[ptr]
            restored_sparse += 1
        restored_hash_sparse = 0
        reinitialized_hash_sparse = 0
        for parameter in hash_sparse_params:
            ptr = parameter.data_ptr()
            if ptr in reinit_ptrs:
                reinitialized_hash_sparse += 1
                continue
            if ptr not in old_state:
                continue
            if self.hash_sparse_optimizer is not None:
                self.hash_sparse_optimizer.state[parameter] = old_state[ptr]
            restored_hash_sparse += 1
        logging.info(
            f"Rebuilt Adagrad optimizer after epoch {epoch}, "
            f"reinitialized_sparse={reinitialized_sparse}, "
            f"restored_sparse={restored_sparse}, "
            f"reinitialized_hash_sparse={reinitialized_hash_sparse}, "
            f"restored_hash_sparse={restored_hash_sparse}")

    def _build_step_dir_name(self, global_step: int, is_best: bool = False) -> str:
        """构造检查点子目录名，例如 ``global_step2500.hidden=64[.best_model]``。"""
        parts = [f"global_step{global_step}"]
        for key in ("hidden",):
            if key in self.ckpt_params:
                parts.append(f"{key}={self.ckpt_params[key]}")
        name = ".".join(parts)
        if is_best:
            name += ".best_model"
        return name

    def _save_epoch_checkpoint(self, epoch: int, global_step: int) -> str:
        dir_name = f"epoch{int(epoch)}.global_step{int(global_step)}"
        for key in ("hidden",):
            if key in self.ckpt_params:
                dir_name += f".{key}={self.ckpt_params[key]}"
        ckpt_dir = os.path.join(self.save_dir, dir_name)
        os.makedirs(ckpt_dir, exist_ok=True)
        torch.save(self.model.state_dict(), os.path.join(ckpt_dir, "model.pt"))
        self._write_sidecar_files(ckpt_dir)
        logging.info(f"Saved epoch checkpoint to {ckpt_dir}/model.pt")
        return ckpt_dir

    def _write_sidecar_files(self, ckpt_dir: str) -> None:
        """在 ``model.pt`` 旁写入配套文件。

        目前最多持久化两个文件，每次调用都会覆盖:

        - ``schema.json``（从 ``self.schema_path`` 复制）：重建 Parquet 数据集所需的特征布局元数据。
        - ``train_config.json``（由 ``self.train_config`` 序列化）：训练时完整超参数集合。
        """
        os.makedirs(ckpt_dir, exist_ok=True)
        if self.schema_path and os.path.exists(self.schema_path):
            shutil.copy2(self.schema_path, ckpt_dir)

        if self.train_config:
            import json
            with open(os.path.join(ckpt_dir, 'train_config.json'), 'w') as f:
                json.dump(self.train_config, f, indent=2)

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
            torch.save(self.model.state_dict(), os.path.join(ckpt_dir, "model.pt"))
        self._write_sidecar_files(ckpt_dir)
        logging.info(f"Saved checkpoint to {ckpt_dir}/model.pt")
        return ckpt_dir

    def _remove_old_best_dirs(self) -> None:
        """删除过期的 ``*.best_model`` 目录。"""
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
        self,
        epoch: int,
        total_step: int,
        val_auc: float,
        val_logloss: float,
        val_pcoc: float,
        val_brier: float,
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
            old_best is None
            or val_auc > old_best + self.early_stopping.delta
        )
        if not is_likely_new_best:
            # 预计不会产生新的最佳模型：不触碰磁盘。之前的
            # best_model 目录（含 model.pt 和配套文件）仍然有效。
            self.early_stopping(val_auc, self.model, {
                "best_val_AUC": val_auc,
                "best_val_logloss": val_logloss,
                "best_val_PCOC": val_pcoc,
                "best_val_brier": val_brier,
            })
            return

        if not self.persist_best_checkpoint:
            self.early_stopping(val_auc, self.model, {
                "best_val_AUC": val_auc,
                "best_val_logloss": val_logloss,
                "best_val_PCOC": val_pcoc,
                "best_val_brier": val_brier,
            })
            if self.early_stopping.best_score != old_best:
                self.best_epoch = epoch
                self.best_global_step = total_step
            return

        # 将 EarlyStopping 指向当前 step 的规范 best-model 位置。
        # 只在可能产生新最佳的分支执行，避免跳过保存时
        # 未使用路径泄漏到 EarlyStopping 状态中。
        best_dir = os.path.join(
            self.save_dir,
            self._build_step_dir_name(total_step, is_best=True),
        )
        self.early_stopping.checkpoint_path = os.path.join(best_dir, "model.pt")

        self._remove_old_best_dirs()

        self.early_stopping(val_auc, self.model, {
            "best_val_AUC": val_auc,
            "best_val_logloss": val_logloss,
            "best_val_PCOC": val_pcoc,
            "best_val_brier": val_brier,
        })

        # 仅当 EarlyStopping 实际确认新的最佳模型，
        # 且写入 model.pt 后，才写入配套文件。如果分数触发了启发式判断，
        # 但 EarlyStopping 内部拒绝保存，则跳过以避免
        # 创建空的（只有配套文件的）检查点目录。
        if self.early_stopping.best_score != old_best and os.path.exists(
            self.early_stopping.checkpoint_path
        ):
            self.best_epoch = epoch
            self.best_global_step = total_step
            self._save_step_checkpoint(
                total_step, is_best=True, skip_model_file=True)

    def _build_train_summary(
        self,
        total_step: int,
        epochs_completed: int,
        stop_reason: str,
    ) -> Dict[str, Any]:
        return {
            'run_name': self.run_name,
            'total_steps': int(total_step),
            'epochs_completed': int(epochs_completed),
            'stop_reason': stop_reason,
            'best_epoch': None if self.best_epoch is None else int(self.best_epoch),
            'best_global_step': None if self.best_global_step is None else int(self.best_global_step),
            'best_score': None if self.early_stopping.best_score is None else float(self.early_stopping.best_score),
            'best_extra_metrics': self.early_stopping.best_extra_metrics,
            'train_batches_per_epoch': int(len(self.train_loader)),
            'stop_after_global_step': int(self.stop_after_global_step),
            'no_eval': bool(self.no_eval),
            'save_each_epoch_ckpt': bool(self.save_each_epoch_ckpt),
        }

    def _finalize_training(
        self,
        total_step: int,
        epochs_completed: int,
        stop_reason: str,
    ) -> Dict[str, Any]:
        if self.no_eval:
            logging.info("No-eval training complete, saving final checkpoint to save_dir root")
            torch.save(self.model.state_dict(), os.path.join(self.save_dir, "model.pt"))
            self._write_sidecar_files(self.save_dir)
        summary = self._build_train_summary(total_step, epochs_completed, stop_reason)
        logging.info(f"Training summary: {summary}")
        return summary

    def train(self) -> Dict[str, Any]:
        """主训练循环：遍历 epoch，执行 step 级和 epoch 级验证，触发 EarlyStopping。"""
        print("Start training (DIN + MLP)")
        self.model.train()
        total_step = 0

        for epoch in range(1, self.num_epochs + 1):
            train_total = len(self.train_loader)
            loss_sum = 0.0

            for step, batch in enumerate(self.train_loader):
                loss = self._train_step(batch)
                total_step += 1
                loss_sum += loss

                if self.writer:
                    self.writer.add_scalar('Loss/train', loss, total_step)

                current = step + 1
                if _should_log_progress(current, train_total, chunks=5):
                    percent = 100.0 * current / max(1, train_total)
                    logging.info(
                        f"Epoch {epoch} Train progress: {current}/{train_total} "
                        f"({percent:.0f}%), loss={loss:.4f}"
                    )

                if self.stop_after_global_step > 0 and total_step >= self.stop_after_global_step:
                    logging.info(
                        f"Reached stop_after_global_step={self.stop_after_global_step} "
                        f"at epoch {epoch}, step {total_step}"
                    )
                    return self._finalize_training(
                        total_step=total_step,
                        epochs_completed=epoch,
                        stop_reason='reached_stop_after_global_step',
                    )

                # step 级验证（仅当 eval_every_n_steps > 0 时）。
                if (not self.no_eval) and self.eval_every_n_steps > 0 and total_step % self.eval_every_n_steps == 0:
                    logging.info(f"Evaluating at step {total_step}")
                    val_auc, val_logloss, val_pcoc, val_brier = self.evaluate(epoch=epoch)
                    self.model.train()
                    torch.cuda.empty_cache()

                    logging.info(
                        f"Step {total_step} Validation | "
                        f"AUC: {val_auc}, LogLoss: {val_logloss}, "
                        f"PCOC: {val_pcoc}, Brier: {val_brier}"
                    )

                    if self.writer:
                        self.writer.add_scalar('AUC/valid', val_auc, total_step)
                        self.writer.add_scalar('LogLoss/valid', val_logloss, total_step)
                        self.writer.add_scalar('PCOC/valid', val_pcoc, total_step)
                        self.writer.add_scalar('Brier/valid', val_brier, total_step)

                    self._handle_validation_result(
                        epoch, total_step, val_auc, val_logloss, val_pcoc, val_brier)

                    if self.early_stopping.early_stop:
                        logging.info(f"Early stopping at step {total_step}")
                        return self._finalize_training(
                            total_step=total_step,
                            epochs_completed=epoch,
                            stop_reason='early_stop_step',
                        )

            logging.info(f"Epoch {epoch}, Average Loss: {loss_sum / len(self.train_loader)}")

            if self.save_each_epoch_ckpt:
                self._save_epoch_checkpoint(epoch, total_step)

            if self.no_eval:
                self._save_step_checkpoint(total_step, is_best=False)
            else:
                val_auc, val_logloss, val_pcoc, val_brier = self.evaluate(epoch=epoch)
                self.model.train()
                torch.cuda.empty_cache()

                logging.info(
                    f"Epoch {epoch} Validation | "
                    f"AUC: {val_auc}, LogLoss: {val_logloss}, "
                    f"PCOC: {val_pcoc}, Brier: {val_brier}"
                )

                if self.writer:
                    self.writer.add_scalar('AUC/valid', val_auc, total_step)
                    self.writer.add_scalar('LogLoss/valid', val_logloss, total_step)
                    self.writer.add_scalar('PCOC/valid', val_pcoc, total_step)
                    self.writer.add_scalar('Brier/valid', val_brier, total_step)

                self._handle_validation_result(
                    epoch, total_step, val_auc, val_logloss, val_pcoc, val_brier)

                if self.early_stopping.early_stop:
                    logging.info(f"Early stopping at epoch {epoch}")
                    return self._finalize_training(
                        total_step=total_step,
                        epochs_completed=epoch,
                        stop_reason='early_stop_epoch',
                    )

            if self.reinit_sparse_after_epoch > 0 and epoch >= self.reinit_sparse_after_epoch:
                self._reinitialize_sparse_embeddings(epoch)

        return self._finalize_training(
            total_step=total_step,
            epochs_completed=self.num_epochs,
            stop_reason='completed',
        )

    def _make_model_input(self, device_batch: Dict[str, Any]) -> ModelInput:
        """从 device_batch dict 构造 ``ModelInput`` NamedTuple。"""
        seq_domains = device_batch['_seq_domains']
        seq_data: Dict[str, torch.Tensor] = {}
        seq_lens: Dict[str, torch.Tensor] = {}
        for domain in seq_domains:
            seq_data[domain] = device_batch[domain]
            seq_lens[domain] = device_batch[f'{domain}_len']
        return ModelInput(
            user_int_feats=device_batch['user_int_feats'],
            item_int_feats=device_batch['item_int_feats'],
            user_dense_feats=device_batch['user_dense_feats'],
            item_dense_feats=device_batch['item_dense_feats'],
            sample_time_feats=device_batch['sample_time_feats'],
            activity_feats=device_batch['activity_feats'],
            seq_data=seq_data,
            seq_lens=seq_lens,
        )


    def _forward_train(self, model_input: ModelInput) -> torch.Tensor:
        model = self.compiled_model if self.compiled_model is not None else self.model
        return model(model_input)

    def _autocast_context(self):
        return torch.amp.autocast(
            self.amp_device_type,
            enabled=self.amp_enabled,
            dtype=self.amp_dtype,
        )

    def _train_step(self, batch: Dict[str, Any]) -> float:
        """执行单个训练 step，并返回标量 loss 值。"""
        device_batch = self._batch_to_device(batch)
        label = device_batch['label'].float()
        sample_weight = device_batch.get('sample_weight')
        if sample_weight is not None:
            sample_weight = sample_weight.float()

        self.dense_optimizer.zero_grad()
        if self.sparse_optimizer is not None:
            self.sparse_optimizer.zero_grad()
        if self.hash_sparse_optimizer is not None:
            self.hash_sparse_optimizer.zero_grad()

        model_input = self._make_model_input(device_batch)
        with self._autocast_context():
            logits = self._forward_train(model_input)  # (B, 1)
            logits = logits.squeeze(-1)  # (B,)

            if self.loss_type == 'focal':
                raw_loss = sigmoid_focal_loss(
                    logits,
                    label,
                    alpha=self.focal_alpha,
                    gamma=self.focal_gamma,
                    reduction='none',
                )
            else:
                raw_loss = F.binary_cross_entropy_with_logits(
                    logits, label, reduction='none'
                )
            if sample_weight is not None:
                loss = (raw_loss * sample_weight).sum() / sample_weight.sum().clamp_min(1e-6)
            else:
                loss = raw_loss.mean()

        if self.grad_scaler.is_enabled():
            self.grad_scaler.scale(loss).backward()
        else:
            loss.backward()
        # foreach=False：规避本项目中某些张量形状触发的
        # PyTorch _foreach_norm CUDA kernel bug。
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0, foreach=False)

        self.dense_optimizer.step()
        if self.sparse_optimizer is not None:
            self.sparse_optimizer.step()
        if self.hash_sparse_optimizer is not None:
            self.hash_sparse_optimizer.step()

        return loss.item()

    def _compute_binary_metrics(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
    ) -> Tuple[float, float, float, float, int]:
        logits = logits.float()
        labels = labels.long()
        probs = torch.sigmoid(logits).float().numpy()
        labels_np = labels.numpy()

        nan_mask = np.isnan(probs)
        if nan_mask.any():
            valid_mask = ~nan_mask
            probs = probs[valid_mask]
            labels_np = labels_np[valid_mask]
            logits = logits[valid_mask]
            labels = labels[valid_mask]

        if len(probs) == 0 or len(np.unique(labels_np)) < 2:
            auc = 0.0
        else:
            auc = float(roc_auc_score(labels_np, probs))

        finite_mask = ~torch.isnan(logits)
        valid_logits = logits[finite_mask]
        valid_labels = labels[finite_mask]
        if len(valid_logits) > 0:
            logloss = F.binary_cross_entropy_with_logits(
                valid_logits, valid_labels.float()
            ).item()
        else:
            logloss = float('inf')

        if len(probs) > 0:
            label_mean = float(labels_np.mean())
            prob_mean = float(probs.mean())
            brier = float(np.mean((probs - labels_np.astype(np.float32)) ** 2))
            pcoc = prob_mean / label_mean if label_mean > 0.0 else float('nan')
        else:
            pcoc = float('nan')
            brier = float('nan')

        return auc, logloss, pcoc, brier, int(len(probs))

    def evaluate(self, epoch: Optional[int] = None) -> Tuple[float, float, float, float]:
        """在 ``self.valid_loader`` 上执行验证，并返回 ``(AUC, logloss, PCOC, brier)``。

        计算各指标前会过滤 NaN 预测值（梯度爆炸时可能出现）。
        """
        print("Start Evaluation (DIN + MLP) - validation")
        self.model.eval()
        if not epoch:
            epoch = -1

        valid_total = len(self.valid_loader)
        all_logits_list = []
        all_labels_list = []

        with torch.no_grad():
            for step, batch in enumerate(self.valid_loader):
                logits, labels = self._evaluate_step(batch)
                all_logits_list.append(logits.detach().cpu())
                all_labels_list.append(labels.detach().cpu())
                current = step + 1
                if _should_log_progress(current, valid_total, chunks=5):
                    percent = 100.0 * current / max(1, valid_total)
                    logging.info(
                        f"Epoch {epoch} Validation progress: {current}/{valid_total} "
                        f"({percent:.0f}%)"
                    )

        all_logits = torch.cat(all_logits_list, dim=0).float()
        all_labels = torch.cat(all_labels_list, dim=0).long()
        auc, logloss, pcoc, brier, _ = self._compute_binary_metrics(
            all_logits, all_labels
        )

        if (
            self.enable_block_knockout
            and int(epoch) == self.block_knockout_epoch
        ):
            self._run_block_knockout(epoch=int(epoch), name='valid')

        return auc, logloss, pcoc, brier

    def _zero_ranges(
        self,
        tensor: torch.Tensor,
        ranges: List[Tuple[int, int]],
    ) -> None:
        for offset, length in ranges:
            if length > 0:
                tensor[:, offset : offset + length] = 0

    def _clone_for_knockout(
        self,
        batch: Dict[str, Any],
        group: str,
    ) -> Dict[str, Any]:
        out = dict(batch)

        def tensor(name: str) -> torch.Tensor:
            out[name] = out[name].clone()
            return out[name]

        user_dense_encoder = self.model.user_dense_encoder
        pair_encoder = user_dense_encoder.pair_encoder

        if group == 'user_sparse_all':
            tensor('user_int_feats').zero_()
        elif group == 'user_main_ue':
            ranges = [
                (offset, length)
                for _, offset, length in user_dense_encoder.main_ue_encoder.field_specs
            ]
            self._zero_ranges(tensor('user_dense_feats'), ranges)
        elif group == 'user_pair':
            dense_ranges = [
                (dense_offset, length)
                for _, _, _, dense_offset, length in pair_encoder.pair_specs
            ]
            int_ranges = [
                (int_offset, length)
                for _, _, int_offset, _, length in pair_encoder.pair_specs
            ]
            self._zero_ranges(tensor('user_dense_feats'), dense_ranges)
            self._zero_ranges(tensor('user_int_feats'), int_ranges)
        elif group == 'user_small_ue':
            ranges = [
                (offset, length)
                for _, offset, length in user_dense_encoder.small_ue_encoder.field_specs
            ]
            self._zero_ranges(tensor('user_dense_feats'), ranges)
        elif group == 'item_sparse_all':
            tensor('item_int_feats').zero_()
        elif group == 'sample_time':
            tensor('sample_time_feats').zero_()
        elif group == 'activity':
            tensor('activity_feats').zero_()
        elif group == 'seq_all':
            for domain in self.model.seq_domains:
                tensor(domain).zero_()
                tensor(f'{domain}_len').zero_()
        elif group == 'seq_raw_all':
            for domain in self.model.seq_domains:
                seq = tensor(domain)
                raw_slot_count = max(0, len(self.model.seq_encoders[domain].vocab_sizes) - 3)
                if raw_slot_count > 0:
                    seq[:, :raw_slot_count, :] = 0
        elif group == 'seq_time_all':
            for domain in self.model.seq_domains:
                seq = tensor(domain)
                raw_slot_count = max(0, len(self.model.seq_encoders[domain].vocab_sizes) - 3)
                seq[:, raw_slot_count:, :] = 0
        elif group.startswith('seq_domain_'):
            domain = group[len('seq_domain_'):]
            if domain in self.model.seq_domains:
                tensor(domain).zero_()
                tensor(f'{domain}_len').zero_()
        else:
            raise ValueError(f'unsupported knockout group: {group}')

        return out

    def _block_knockout_groups(self) -> List[str]:
        groups = [
            'user_sparse_all',
            'user_main_ue',
            'user_pair',
            'user_small_ue',
            'item_sparse_all',
            'sample_time',
            'activity',
            'seq_all',
            'seq_raw_all',
            'seq_time_all',
        ]
        groups.extend(f'seq_domain_{domain}' for domain in self.model.seq_domains)
        return groups

    def _run_block_knockout(self, *, epoch: int, name: str) -> None:
        groups = self._block_knockout_groups()
        base_logits_list: List[torch.Tensor] = []
        labels_list: List[torch.Tensor] = []
        group_logits: Dict[str, List[torch.Tensor]] = {group: [] for group in groups}

        logging.info(
            f"Start block knockout diagnostics | epoch={epoch} | name={name} | "
            f"groups={','.join(groups)}"
        )
        self.model.eval()
        with torch.no_grad():
            for batch in self.valid_loader:
                device_batch = self._batch_to_device(batch)
                model_input = self._make_model_input(device_batch)
                with self._autocast_context():
                    base_logits, _ = self.model.predict(model_input)
                base_logits_list.append(base_logits.squeeze(-1).detach().cpu())
                labels_list.append(device_batch['label'].detach().cpu())

                for group in groups:
                    ko_batch = self._clone_for_knockout(device_batch, group)
                    ko_input = self._make_model_input(ko_batch)
                    with self._autocast_context():
                        ko_logits, _ = self.model.predict(ko_input)
                    group_logits[group].append(
                        ko_logits.squeeze(-1).detach().cpu()
                    )

        base_logits = torch.cat(base_logits_list, dim=0)
        labels = torch.cat(labels_list, dim=0)
        base_auc, base_logloss, base_pcoc, base_brier, row_count = (
            self._compute_binary_metrics(base_logits, labels)
        )

        for group in groups:
            logits = torch.cat(group_logits[group], dim=0)
            auc, logloss, pcoc, brier, _ = self._compute_binary_metrics(
                logits, labels
            )
            logging.info(
                "BLOCK_KNOCKOUT | "
                f"epoch={epoch} | name={name} | group={group} | rows={row_count} | "
                f"auc={auc:.6f} | auc_delta={auc - base_auc:.6f} | "
                f"logloss={logloss:.6f} | logloss_delta={logloss - base_logloss:.6f} | "
                f"pcoc={pcoc:.6f} | pcoc_delta={pcoc - base_pcoc:.6f} | "
                f"brier={brier:.6f} | brier_delta={brier - base_brier:.6f}"
            )

    def _evaluate_step(
        self, batch: Dict[str, Any]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """执行单个验证 step，并返回 ``(logits, labels)``。"""
        device_batch = self._batch_to_device(batch)
        label = device_batch['label']

        model_input = self._make_model_input(device_batch)
        with self._autocast_context():
            logits, _ = self.model.predict(model_input)  # (B, 1), (B, D)
        logits = logits.squeeze(-1)  # (B,)

        return logits, label

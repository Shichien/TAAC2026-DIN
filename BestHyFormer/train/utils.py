import os
import random
import copy
import logging
import time
from datetime import timedelta
from typing import Optional, Dict, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class LogFormatter:
    """自定义 ``logging.Formatter``，为每条记录加上墙钟时间戳和自该 formatter 实例构造以来经过的墙钟时间。

    前缀格式为 ``"<locale-date> <locale-time> - H:MM:SS"``，便于跟踪长时间训练任务，因为绝对时间和启动后耗时都很有用。

    多行消息会重新缩进，使续行与消息正文开头对齐（而不是与前缀对齐）。
    """

    def __init__(self) -> None:
        # 用于计算日志前缀中耗时部分的锚点。
        # 可在运行时通过 ``create_logger(...).reset_time()`` 重置。
        self.start_time: float = time.time()

    def format(self, record: logging.LogRecord) -> str:
        elapsed_seconds = round(record.created - self.start_time)

        prefix = "%s - %s" % (
            time.strftime("%x %X"),
            timedelta(seconds=elapsed_seconds),
        )
        message = record.getMessage()
        # 缩进续行，使其与消息正文对齐，
        # 而不是与时间戳前缀对齐。
        message = message.replace("\n", "\n" + " " * (len(prefix) + 3))
        return "%s - %s" % (prefix, message)


def create_logger(filepath: str) -> logging.Logger:
    """为一次训练/推理运行创建并配置 root logger。

    返回的 logger 附加两个 handler:

    * 绑定到 ``filepath`` 的 ``FileHandler``（以写模式打开，会截断旧内容），记录 ``DEBUG`` 及以上级别消息，便于事后排查。
    * 输出到 stderr 的 ``StreamHandler``，只回显 ``INFO`` 及以上级别消息，使控制台输出保持简洁。

    两个 handler 共享同一个 ``LogFormatter``，使控制台与日志文件保持一致。root logger 上已有的 handler 会被移除，避免该函数多次调用时产生重复日志。

    参数:
        filepath: 日志文件目标路径。以 ``"w"`` 模式打开，因此会覆盖旧内容。

    返回:
        root ``logging.Logger`` 实例。返回对象会被附加 ``reset_time()`` 属性，用于重置日志前缀使用的耗时计时器。当一次运行的“关键”阶段在进程启动很久之后才开始时（例如 schema 构建和数据加载之后），这很有用。
    """
    log_formatter = LogFormatter()

    file_handler = logging.FileHandler(filepath, "w")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(log_formatter)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(log_formatter)

    logger = logging.getLogger()
    logger.handlers = []
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    # 允许调用方重置日志前缀中显示的耗时计时器。
    def reset_time() -> None:
        log_formatter.start_time = time.time()

    logger.reset_time = reset_time  # type: ignore[attr-defined]

    return logger


class EarlyStopping:
    """当验证指标进入平台期时提前停止训练。

    该跟踪器假设指标越高越好（AUC 或 accuracy 常见如此）。候选 ``score`` 只有在 ``score > best_score + delta`` 时才视为提升；否则内部 ``counter`` 递增，并在 ``counter >= patience`` 时请求停止训练。

    每次提升时，当前 ``model.state_dict()`` 都会在内存中深拷贝（``self.best_model``），并持久化到磁盘 ``checkpoint_path``。最近一次真正写入的提升分数缓存到 ``self.best_saved_score``，以便调用方跳过冗余 IO。

    属性:
        checkpoint_path: 最佳 ``state_dict`` 的目标路径。
        patience: 触发 ``early_stop`` 变为 ``True`` 前允许的未提升调用次数。
        verbose: 若为 ``True``，每次写入检查点时输出一行 ``INFO`` 日志。
        counter: 当前连续未提升调用次数。
        best_score: 已观测最佳分数；首次调用前为 ``None``。
        early_stop: 当 ``counter >= patience`` 后设为 ``True``。
        delta: 重置 ``counter`` 所需的最小绝对提升。
        best_model: 内存中的最佳 ``state_dict`` 深拷贝。
        best_saved_score: 上一次实际写入磁盘的检查点对应分数。
        best_extra_metrics: 在最佳分数 step 捕获的可选辅助指标（例如 logloss、其他 AUC）。
        label: 短前缀（例如 ``"val"``），会加到日志行前，用于区分并行运行的多个跟踪器。
    """

    def __init__(
        self,
        checkpoint_path: str,
        label: str = "",
        patience: int = 5,
        verbose: bool = False,
        delta: float = 0,
    ) -> None:
        self.checkpoint_path: str = checkpoint_path
        self.patience: int = patience
        self.verbose: bool = verbose
        self.counter: int = 0
        self.best_score: Optional[float] = None
        self.early_stop: bool = False
        self.delta: float = delta
        self.best_model: Optional[Dict[str, torch.Tensor]] = None
        self.best_saved_score: float = 0.0
        self.best_extra_metrics: Optional[Dict[str, Any]] = None
        self.label: str = label
        if self.label != "":
            self.label += " "

    def _is_not_improved(self, score: float) -> bool:
        """当且仅当 ``score`` 未超过 ``best_score + delta`` 时返回 ``True``。

        作为递增 patience 计数器的门控条件。``best_score`` 必须已由之前的 ``__call__`` 初始化。
        """
        assert self.best_score is not None, (
            "call __call__ first to seed best_score"
        )
        if score > self.best_score + self.delta:
            return False
        return True

    def __call__(
        self,
        score: float,
        model: nn.Module,
        extra_metrics: Optional[Dict[str, Any]] = None,
    ) -> None:
        """向跟踪器输入新的验证分数。

        按顺序分三种分支:

        1. 首次调用（``best_score is None``）：初始化跟踪器、持久化检查点并缓存模型权重。
        2. 未提升：递增 ``counter`` 并记录进度；一旦 ``counter >= patience``，将 ``early_stop`` 置为 True。
        3. 有提升：将 ``counter`` 重置为 ``0``，更新 ``best_score`` 和 ``best_extra_metrics``，刷新内存中的 ``best_model``，并向磁盘写入新检查点。

        参数:
            score: 标量验证指标（越高越好，例如 AUC）。
            model: 提升时需要快照 ``state_dict`` 的模型。只保存参数，不保存优化器状态。
            extra_metrics: 同一 step 记录的可选辅助指标 dict，例如 ``{"best_val_AUC": ..., "best_val_logloss": ...}``。会原样存储为 ``self.best_extra_metrics``；``EarlyStopping`` 自身不解释它。
        """
        if self.best_score is None:
            self.best_score = score
            self.best_extra_metrics = extra_metrics
            self.best_saved_score = 0.0
            self.save_checkpoint(score, model)
            self.best_model = copy.deepcopy(model.state_dict())
        elif self._is_not_improved(score):
            self.counter += 1
            logging.info(
                f"{self.label}earlyStopping counter: {self.counter} / {self.patience}"
            )
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            logging.info(f"{self.label}earlyStopping counter reset!")
            self.best_score = score
            self.best_model = copy.deepcopy(model.state_dict())
            self.best_extra_metrics = extra_metrics
            self.save_checkpoint(score, model)
            self.counter = 0

    def save_checkpoint(self, score: float, model: nn.Module) -> None:
        """将 ``model.state_dict()`` 持久化到 ``self.checkpoint_path``。

        创建缺失的父目录，通过 ``torch.save`` 原子写入，并把 ``score`` 记录为 ``self.best_saved_score``，使后续调用方无需重新读取检查点文件即可判断“自上次保存以来没有新提升”。

        参数:
            score: 与正在保存的权重对应的验证分数。写入完成后通过 ``best_saved_score`` 暴露给调用方。
            model: 参数正在被快照的模型。只写入 ``state_dict()``；优化器和调度器状态明确不包含在内。
        """
        if self.verbose:
            logging.info("Validation score increased. Saving model ...")
        os.makedirs(os.path.dirname(self.checkpoint_path), exist_ok=True)
        torch.save(model.state_dict(), self.checkpoint_path)
        self.best_saved_score = score


def set_seed(seed: int) -> None:
    """为所有会影响训练可复现性的随机数生成器设定种子。

    设置 ``random``、``PYTHONHASHSEED`` 环境变量、NumPy、CPU PyTorch generator 以及所有 CUDA generator，然后强制 cuDNN 进入确定性模式。

    注意，GPU 上完全逐比特确定还需要禁用 cuDNN auto-tuning（``torch.backends.cudnn.benchmark = False``），并且可能带来不可忽略的吞吐成本；该辅助函数有意只切换 ``deterministic``，以保留常见用例下的速度。

    参数:
        seed: 上述所有 RNG 共享的非负整数种子。
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def sigmoid_focal_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = 0.1,
    gamma: float = 2.0,
    reduction: str = "mean",
) -> torch.Tensor:
    """Focal Loss：FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    参数:
        logits: (N,) 原始 logits（sigmoid 之前）。
        targets: (N,) 二分类标签 {0, 1}。
        alpha: 正类权重，位于 (0, 1)。当正样本占多数时，使用 alpha < 0.5 下调正类权重。
        gamma: 聚焦参数。gamma=0 退化为标准 BCE；gamma=2 是标准值。
        reduction: 'mean' | 'sum' | 'none'。
    """
    p = torch.sigmoid(logits)
    bce_loss = F.binary_cross_entropy_with_logits(
        logits, targets, reduction="none"
    )
    p_t = p * targets + (1 - p) * (1 - targets)
    focal_weight = (1 - p_t) ** gamma
    alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
    loss = alpha_t * focal_weight * bce_loss
    if reduction == "mean":
        return loss.mean()
    elif reduction == "sum":
        return loss.sum()
    return loss

"""PCVR Parquet 数据集模块（性能调优版）。

直接读取原始多列 Parquet，并从 ``schema.json`` 获取特征元数据。

优化点:
- 预分配 numpy buffer，消除 ``np.zeros`` + ``np.stack`` 的额外开销。
- 对序列域使用融合 padding 循环，直接写入 3D buffer。
- 预计算列索引查找，避免逐行字符串查找。
- 使用 ``file_system`` 张量共享策略，规避多 DataLoader worker 下的 ``/dev/shm`` 耗尽问题。
"""

import os
import logging
import random
import json
import gc

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
import torch.multiprocessing
from torch.utils.data import IterableDataset, DataLoader
from typing import Any, Dict, Iterator, List, Optional, Tuple

# numpy.typing 从 numpy >= 1.20 开始可用；在更旧版本中回退到
# 一个空操作 shim，使 ``npt.NDArray[np.int64]`` 这类前向引用注解
# 作为普通字符串继续工作，不会在导入时报错。
try:
    import numpy.typing as npt  # noqa: F401
except ImportError:  # pragma: no cover

    class _NptFallback:  # type: ignore[no-redef]
        NDArray = Any

    npt = _NptFallback()  # type: ignore[assignment]


# --------------------------- 特征结构 ---------------------------


class FeatureSchema:
    """为每个特征记录 ``(feature_id, offset, length)``，使下游代码能定位扁平化张量中属于某个 feature id 的片段。

    对于 int 特征:
      - int_value: length = 1
      - int_array: length = array length
      - int_array_and_float_array: int 部分长度
    对于 dense 特征:
      - float_value: length = 1
      - float_array: length = array length
      - int_array_and_float_array: float 部分长度
    """

    def __init__(self) -> None:
        # (feature_id, offset, length) 的有序列表。
        self.entries: List[Tuple[int, int, int]] = []
        self.total_dim: int = 0
        # 从 fid 快速查找其 (offset, length)。
        self._fid_to_entry: Dict[int, Tuple[int, int]] = {}

    def add(self, feature_id: int, length: int) -> None:
        """向 schema 追加一个特征。"""
        offset = self.total_dim
        self.entries.append((feature_id, offset, length))
        self._fid_to_entry[feature_id] = (offset, length)
        self.total_dim += length

    def get_offset_length(self, feature_id: int) -> Tuple[int, int]:
        """获取某个 feature_id 对应的 ``(offset, length)``。"""
        return self._fid_to_entry[feature_id]

    @property
    def feature_ids(self) -> List[int]:
        """按插入顺序返回所有 feature_id。"""
        return [fid for fid, _, _ in self.entries]

    def to_dict(self) -> Dict[str, Any]:
        """序列化为普通 dict（用于写入 JSON）。"""
        return {"entries": self.entries, "total_dim": self.total_dim}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "FeatureSchema":
        """从 dict 形式重建 :class:`FeatureSchema`。"""
        schema = cls()
        for fid, offset, length in d["entries"]:
            schema.entries.append((fid, offset, length))
            schema._fid_to_entry[fid] = (offset, length)
        schema.total_dim = d["total_dim"]
        return schema

    def __repr__(self) -> str:
        lines = [f"FeatureSchema(total_dim={self.total_dim}, features=["]
        for fid, offset, length in self.entries:
            lines.append(f"  fid={fid}: offset={offset}, length={length}")
        lines.append("])")
        return "\n".join(lines)


# 使用基于文件系统的张量共享（而不是 /dev/shm），避免
# 多个 DataLoader worker 同时运行时共享内存耗尽。
torch.multiprocessing.set_sharing_strategy("file_system")

# 时间差分桶边界（64 条边界 -> 65 个桶：0=填充，1..64）。
# fmt: off
BUCKET_BOUNDARIES = np.array([
    5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60,
    120, 180, 240, 300, 360, 420, 480, 540, 600,
    900, 1200, 1500, 1800, 2100, 2400, 2700, 3000, 3300, 3600,
    5400, 7200, 9000, 10800, 12600, 14400, 16200, 18000, 19800, 21600,
    32400, 43200, 54000, 64800, 75600, 86400,
    172800, 259200, 345600, 432000, 518400, 604800,
    1123200, 1641600, 2160000, 2592000,
    4320000, 6048000, 7776000,
    11664000, 15552000,
    31536000,
], dtype=np.int64)
# fmt: on

# 时间桶 Embedding 槽位总数（= 边界数量 + 1，且
# 包含填充桶 0）。
#
# 该常量由 BUCKET_BOUNDARIES 的长度唯一决定；在
# 模型侧，``nn.Embedding(num_embeddings=NUM_TIME_BUCKETS)`` 必须
# 与该值严格匹配，否则运行时可能抛出 IndexError。
#
# 因此 ``train.py`` / ``infer.py`` 只暴露布尔开关
# ``--use_time_buckets``，具体桶数量从这里推导。
NUM_TIME_BUCKETS = len(BUCKET_BOUNDARIES) + 1


class PCVRParquetDataset(IterableDataset):
    """直接读取原始多列 Parquet 的 PCVR 数据集。

    - int 特征：标量或列表（multi-hot）；值 <= 0 会映射为 0（padding）。
    - dense 特征：``list<float>``，变长并 padding 到 ``max_dim``。
    - sequence 特征：``list<int64>``，按 domain 分组；包含 side-info 列和可选时间戳列（用于时间分桶）。
    - label：由 ``label_type == 2`` 映射得到。
    """

    def __init__(
        self,
        parquet_path: str,
        schema_path: str,
        batch_size: int = 256,
        seq_max_lens: Optional[Dict[str, int]] = None,
        shuffle: bool = True,
        buffer_batches: int = 20,
        row_group_range: Optional[Tuple[int, int]] = None,
        clip_vocab: bool = True,
        is_training: bool = True,
    ) -> None:
        """参数:
        parquet_path: 包含 ``*.parquet`` 文件的目录，或单个 parquet 文件路径。
        schema_path: 描述特征布局的 schema JSON 路径。
        batch_size: 预分配 buffer 使用的固定 batch size。
        seq_max_lens: 可选的按 domain 覆盖序列截断长度，例如 ``{'seq_d': 256}``。未列出的 domain 回退到 schema 默认值 256。
        shuffle: 是否在 ``buffer_batches`` 大小的窗口内打乱。
        buffer_batches: 以 batch 为单位的 shuffle buffer 大小。
        row_group_range: Row Group 的 ``(start, end)`` 切片；``None`` 表示使用全部 Row Group。
        timestamp_range: 可选的样本 ``timestamp`` 列 ``(start, end)`` 过滤器。``start`` 包含，``end`` 不包含；任一侧为 ``None`` 表示无边界。
        clip_vocab: 若为 True，将越界 id 裁剪为 0；若为 False，则抛出异常。
        is_training: 若为 True，根据 ``label_type == 2`` 生成 ``label``；若为 False，则返回全 0 label 列。
        """
        super().__init__()

        # 接受目录或单个文件路径。
        if os.path.isdir(parquet_path):
            import glob

            files = sorted(glob.glob(os.path.join(parquet_path, "*.parquet")))
            if not files:
                raise FileNotFoundError(f"No .parquet files in {parquet_path}")
            self._parquet_files = files
        else:
            self._parquet_files = [parquet_path]

        self.batch_size = batch_size
        self.shuffle = shuffle
        self.buffer_batches = buffer_batches
        self.clip_vocab = clip_vocab
        self.is_training = is_training
        # 越界统计：
        #   {(group, col_idx): {'count': N, 'max': M, 'min_oob': M, 'vocab': V}}
        self._oob_stats: Dict[Tuple[str, int], Dict[str, int]] = {}

        # 构建 Row Group 列表。
        self._rg_list = []
        for f in self._parquet_files:
            pf = pq.ParquetFile(f)
            for i in range(pf.metadata.num_row_groups):
                self._rg_list.append((f, i, pf.metadata.row_group(i).num_rows))

        if row_group_range is not None:
            start, end = row_group_range
            self._rg_list = self._rg_list[start:end]

        self.num_rows = sum(r[2] for r in self._rg_list)

        # 加载 schema.json。
        self._load_schema(schema_path, seq_max_lens or {})

        # ---- 预计算列索引查找 ----
        pf = pq.ParquetFile(self._parquet_files[0])
        schema_names = pf.schema_arrow.names
        self._col_idx = {name: i for i, name in enumerate(schema_names)}

        # ---- 预分配 numpy buffer ----
        B = batch_size
        self._buf_user_int = np.zeros(
            (B, self.user_int_schema.total_dim), dtype=np.int64
        )
        self._buf_item_int = np.zeros(
            (B, self.item_int_schema.total_dim), dtype=np.int64
        )
        self._buf_user_dense = np.zeros(
            (B, self.user_dense_schema.total_dim), dtype=np.float32
        )
        self._buf_seq = {}
        self._buf_seq_tb = {}
        self._buf_seq_lens = {}
        for domain in self.seq_domains:
            max_len = self._seq_maxlen[domain]
            n_feats = len(self.sideinfo_fids[domain])
            self._buf_seq[domain] = np.zeros(
                (B, n_feats, max_len), dtype=np.int64
            )
            self._buf_seq_tb[domain] = np.zeros((B, max_len), dtype=np.int64)
            self._buf_seq_lens[domain] = np.zeros(B, dtype=np.int64)

        # ---- 为 int 列预计算 (col_idx, offset, vocab_size) 计划 ----
        self._user_int_plan = []  # [(列索引 col_idx, 维度 dim, 偏移 offset, 词表大小 vocab_size), ...]
        offset = 0
        for fid, vs, dim in self._user_int_cols:
            ci = self._col_idx.get(f"user_int_feats_{fid}")
            self._user_int_plan.append((ci, dim, offset, vs))
            offset += dim

        self._item_int_plan = []
        offset = 0
        for fid, vs, dim in self._item_int_cols:
            ci = self._col_idx.get(f"item_int_feats_{fid}")
            self._item_int_plan.append((ci, dim, offset, vs))
            offset += dim

        self._user_dense_plan = []
        offset = 0
        for fid, dim in self._user_dense_cols:
            ci = self._col_idx.get(f"user_dense_feats_{fid}")
            self._user_dense_plan.append((ci, dim, offset))
            offset += dim

        # 序列列计划：{domain: ([(col_idx, feat_slot, vocab_size), ...], ts_col_idx)}
        self._seq_plan = {}
        for domain in self.seq_domains:
            prefix = self._seq_prefix[domain]
            sideinfo_fids = self.sideinfo_fids[domain]
            ts_fid = self.ts_fids[domain]
            side_plan = []
            for slot, fid in enumerate(sideinfo_fids):
                ci = self._col_idx.get(f"{prefix}_{fid}")
                vs = self.seq_vocab_sizes[domain][fid]
                side_plan.append((ci, slot, vs))
            ts_ci = (
                self._col_idx.get(f"{prefix}_{ts_fid}")
                if ts_fid is not None
                else None
            )
            self._seq_plan[domain] = (side_plan, ts_ci)

        logging.info(
            f"PCVRParquetDataset: {self.num_rows} rows from "
            f"{len(self._parquet_files)} file(s), batch_size={batch_size}, "
            f"buffer_batches={buffer_batches}, shuffle={shuffle}"
        )

    def _load_schema(
        self, schema_path: str, seq_max_lens: Dict[str, int]
    ) -> None:
        """从 ``schema_path`` 填充各组 schema 信息。"""
        with open(schema_path, "r", encoding="utf-8") as f:
            raw = json.load(f)

        # ---- user_int: [[fid, vocab_size, dim], ...] ----
        self._user_int_cols: List[List[int]] = raw["user_int"]
        self.user_int_schema: FeatureSchema = FeatureSchema()
        self.user_int_vocab_sizes: List[int] = []
        for fid, vs, dim in self._user_int_cols:
            self.user_int_schema.add(fid, dim)
            self.user_int_vocab_sizes.extend([vs] * dim)

        # ---- item_int ----
        self._item_int_cols: List[List[int]] = raw["item_int"]
        self.item_int_schema: FeatureSchema = FeatureSchema()
        self.item_int_vocab_sizes: List[int] = []
        for fid, vs, dim in self._item_int_cols:
            self.item_int_schema.add(fid, dim)
            self.item_int_vocab_sizes.extend([vs] * dim)

        # ---- user_dense: [[fid, dim], ...] ----
        self._user_dense_cols: List[List[int]] = raw["user_dense"]
        self.user_dense_schema: FeatureSchema = FeatureSchema()
        for fid, dim in self._user_dense_cols:
            self.user_dense_schema.add(fid, dim)

        # ---- item_dense（空） ----
        self.item_dense_schema: FeatureSchema = FeatureSchema()

        # ---- 序列 domain ----
        self._seq_cfg: Dict[str, Dict[str, Any]] = raw["seq"]
        self.seq_domains: List[str] = sorted(self._seq_cfg.keys())
        self.seq_feature_ids: Dict[str, List[int]] = {}
        self.seq_vocab_sizes: Dict[str, Dict[int, int]] = {}
        self.seq_domain_vocab_sizes: Dict[str, List[int]] = {}
        self.ts_fids: Dict[str, Optional[int]] = {}
        self.sideinfo_fids: Dict[str, List[int]] = {}
        self._seq_prefix: Dict[str, str] = {}
        self._seq_maxlen: Dict[str, int] = {}

        for domain in self.seq_domains:
            cfg = self._seq_cfg[domain]
            self._seq_prefix[domain] = cfg["prefix"]
            ts_fid = cfg["ts_fid"]
            self.ts_fids[domain] = ts_fid

            all_fids = [fid for fid, vs in cfg["features"]]
            self.seq_feature_ids[domain] = all_fids
            self.seq_vocab_sizes[domain] = {
                fid: vs for fid, vs in cfg["features"]
            }

            sideinfo = [fid for fid in all_fids if fid != ts_fid]
            self.sideinfo_fids[domain] = sideinfo
            self.seq_domain_vocab_sizes[domain] = [
                self.seq_vocab_sizes[domain][fid] for fid in sideinfo
            ]

            # max_len: 来自 seq_max_lens 参数；未指定的 domain 回退到 256。
            self._seq_maxlen[domain] = seq_max_lens.get(domain, 256)

    def __len__(self) -> int:
        # 按 Row Group 向上取整；这是实际 batch 数的上界。
        return sum(
            (n + self.batch_size - 1) // self.batch_size
            for _, _, n in self._rg_list
        )

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        worker_info = torch.utils.data.get_worker_info()
        rg_list = self._rg_list
        if worker_info is not None and worker_info.num_workers > 1:
            rg_list = [
                rg
                for i, rg in enumerate(rg_list)
                if i % worker_info.num_workers == worker_info.id
            ]

        buffer: List[Dict[str, Any]] = []
        for file_path, rg_idx, _ in rg_list:
            pf = pq.ParquetFile(file_path)
            for batch in pf.iter_batches(
                batch_size=self.batch_size, row_groups=[rg_idx]
            ):
                batch_dict = self._convert_batch(batch)
                if self.shuffle and self.buffer_batches > 1:
                    buffer.append(batch_dict)
                    if len(buffer) >= self.buffer_batches:
                        yield from self._flush_buffer(buffer)
                        buffer = []
                else:
                    yield batch_dict

        if buffer:
            yield from self._flush_buffer(buffer)

        del buffer
        gc.collect()

    def _flush_buffer(
        self, buffer: List[Dict[str, Any]]
    ) -> Iterator[Dict[str, Any]]:
        """拼接缓冲区中的 batch，在行级别打乱，然后重新切片并产出 batch 大小的块。"""
        merged: Dict[str, torch.Tensor] = {}
        non_tensor_keys: Dict[str, Any] = {}
        for k in buffer[0].keys():
            if isinstance(buffer[0][k], torch.Tensor):
                merged[k] = torch.cat([b[k] for b in buffer], dim=0)
            else:
                non_tensor_keys[k] = buffer[0][k]
        total_rows = merged["label"].shape[0]
        rand_idx = (
            torch.randperm(total_rows)
            if self.shuffle
            else torch.arange(total_rows)
        )
        for i in range(0, total_rows, self.batch_size):
            end = min(i + self.batch_size, total_rows)
            batch: Dict[str, Any] = {
                k: v[rand_idx[i:end]] for k, v in merged.items()
            }
            batch.update(non_tensor_keys)
            yield batch
        del merged
        buffer.clear()

    # ---- 辅助函数 ----

    def _record_oob(
        self,
        group: str,
        col_idx: int,
        arr: "npt.NDArray[np.int64]",
        vocab_size: int,
    ) -> None:
        """记录越界索引，并可选地将其裁剪为 0，同时不打印到控制台。"""
        oob_mask = arr >= vocab_size
        if not oob_mask.any():
            return
        key = (group, col_idx)
        oob_vals = arr[oob_mask]
        n = int(oob_mask.sum())
        mx = int(oob_vals.max())
        mn = int(oob_vals.min())
        if key in self._oob_stats:
            s = self._oob_stats[key]
            s["count"] += n
            s["max"] = max(s["max"], mx)
            s["min_oob"] = min(s["min_oob"], mn)
        else:
            self._oob_stats[key] = {
                "count": n,
                "max": mx,
                "min_oob": mn,
                "vocab": vocab_size,
            }
        if self.clip_vocab:
            arr[oob_mask] = 0
        else:
            raise ValueError(
                f"{group} col_idx={col_idx}: {n} values out of range "
                f"[0, {vocab_size}), actual=[{mn}, {mx}]. "
                f"Use clip_vocab=True to clip or fix schema.json"
            )

    def dump_oob_stats(self, path: Optional[str] = None) -> None:
        """如果提供 ``path``，则将越界统计写入文件；否则写入 ``logging.info``。"""
        if not self._oob_stats:
            logging.info("No out-of-bound values detected.")
            return
        lines = ["=== Out-of-Bound Stats ==="]
        for (group, ci), s in sorted(self._oob_stats.items()):
            direction = "TOO_HIGH" if s["min_oob"] >= s["vocab"] else "TOO_LOW"
            lines.append(
                f"  {group} col_idx={ci}: vocab={s['vocab']}, "
                f"oob_count={s['count']}, range=[{s['min_oob']}, {s['max']}], "
                f"{direction}"
            )
        msg = "\n".join(lines)
        if path:
            with open(path, "w") as f:
                f.write(msg + "\n")
            logging.info(f"OOB stats written to {path}")
        else:
            logging.info(msg)

    def _pad_varlen_int_column(
        self, arrow_col: "pa.ListArray", max_len: int, B: int
    ) -> Tuple["npt.NDArray[np.int64]", "npt.NDArray[np.int64]"]:
        """将整数 Arrow ``ListArray`` padding 到形状 ``[B, max_len]``。

        值 <= 0 会映射为 0（padding）。注意：原始数据包含 -1（缺失），当前与 0（padding）按相同方式处理。

        返回:
            元组 ``(padded, lengths)``，其中 ``padded`` 形状为 ``[B, max_len]``，``lengths`` 形状为 ``[B]``。
        """
        offsets = arrow_col.offsets.to_numpy()
        values = arrow_col.values.to_numpy()

        padded = np.zeros((B, max_len), dtype=np.int64)
        lengths = np.zeros(B, dtype=np.int64)

        for i in range(B):
            start, end = int(offsets[i]), int(offsets[i + 1])
            raw_len = end - start
            if raw_len <= 0:
                continue
            use_len = min(raw_len, max_len)
            padded[i, :use_len] = values[start : start + use_len]
            lengths[i] = use_len

        padded[padded <= 0] = 0
        return padded, lengths

    # 为 bench_raw_dataset.py 及其他重命名前的外部调用方
    # 保留向后兼容别名。新代码应直接调用
    # `_pad_varlen_int_column`。
    _pad_varlen_column = _pad_varlen_int_column

    def _pad_varlen_float_column(
        self, arrow_col: "pa.ListArray", max_dim: int, B: int
    ) -> "npt.NDArray[np.float32]":
        """将 Arrow ``ListArray<float>`` padding 到形状 ``[B, max_dim]``。"""
        offsets = arrow_col.offsets.to_numpy()
        values = arrow_col.values.to_numpy()

        padded = np.zeros((B, max_dim), dtype=np.float32)

        for i in range(B):
            start, end = int(offsets[i]), int(offsets[i + 1])
            raw_len = end - start
            if raw_len <= 0:
                continue
            use_len = min(raw_len, max_dim)
            padded[i, :use_len] = values[start : start + use_len]

        return padded

    def _convert_batch(self, batch: "pa.RecordBatch") -> Dict[str, Any]:
        """将 Arrow RecordBatch 转换为可直接训练使用的张量 dict。"""
        B = batch.num_rows

        # ---- 元信息 ----
        timestamps = (
            batch.column(self._col_idx["timestamp"]).to_numpy().astype(np.int64)
        )
        if self.is_training:
            labels = (
                batch.column(self._col_idx["label_type"])
                .fill_null(0)
                .to_numpy(zero_copy_only=False)
                .astype(np.int64)
                == 2
            ).astype(np.int64)
        else:
            labels = np.zeros(B, dtype=np.int64)
        user_ids = batch.column(self._col_idx["user_id"]).to_pylist()

        # ---- user_int：写入预分配 buffer ----
        # 注意：null -> 0（通过 fill_null），-1 -> 0（通过 arr<=0）；缺失值
        # 与填充按相同方式处理。vs==0 的特征没有词表
        # 信息，会在数据集侧强制置 0，从而确保
        # 为 vs=0 创建的模型侧 1 槽 Embedding 永远不会被索引到
        # 范围之外。
        user_int = self._buf_user_int[:B]
        user_int[:] = 0
        for ci, dim, offset, vs in self._user_int_plan:
            col = batch.column(ci)
            if dim == 1:
                arr = (
                    col.fill_null(0)
                    .to_numpy(zero_copy_only=False)
                    .astype(np.int64)
                )
                arr[arr <= 0] = 0
                if vs > 0:
                    self._record_oob("user_int", ci, arr, vs)
                else:
                    arr[:] = 0
                user_int[:, offset] = arr
            else:
                padded, _ = self._pad_varlen_int_column(col, dim, B)
                if vs > 0:
                    self._record_oob("user_int", ci, padded, vs)
                else:
                    padded[:] = 0
                user_int[:, offset : offset + dim] = padded

        # ---- item_int ----
        item_int = self._buf_item_int[:B]
        item_int[:] = 0
        for ci, dim, offset, vs in self._item_int_plan:
            col = batch.column(ci)
            if dim == 1:
                arr = (
                    col.fill_null(0)
                    .to_numpy(zero_copy_only=False)
                    .astype(np.int64)
                )
                arr[arr <= 0] = 0
                if vs > 0:
                    self._record_oob("item_int", ci, arr, vs)
                else:
                    arr[:] = 0
                item_int[:, offset] = arr
            else:
                padded, _ = self._pad_varlen_int_column(col, dim, B)
                if vs > 0:
                    self._record_oob("item_int", ci, padded, vs)
                else:
                    padded[:] = 0
                item_int[:, offset : offset + dim] = padded

        # ---- user_dense ----
        user_dense = self._buf_user_dense[:B]
        user_dense[:] = 0
        for ci, dim, offset in self._user_dense_plan:
            col = batch.column(ci)
            padded = self._pad_varlen_float_column(col, dim, B)
            user_dense[:, offset : offset + dim] = padded

        result = {
            "user_int_feats": torch.from_numpy(user_int.copy()),
            "user_dense_feats": torch.from_numpy(user_dense.copy()),
            "item_int_feats": torch.from_numpy(item_int.copy()),
            "item_dense_feats": torch.zeros(B, 0, dtype=torch.float32),
            "label": torch.from_numpy(labels),
            "timestamp": torch.from_numpy(timestamps),
            "user_id": user_ids,
            "_seq_domains": self.seq_domains,
        }

        # ---- 序列特征：融合填充，直接写入 3D 缓冲区 ----
        for domain in self.seq_domains:
            max_len = self._seq_maxlen[domain]
            side_plan, ts_ci = self._seq_plan[domain]

            # 直接写入预分配的 3D buffer。
            out = self._buf_seq[domain][:B]
            out[:] = 0
            lengths = self._buf_seq_lens[domain][:B]
            lengths[:] = 0

            # 融合路径：先为每个侧信息列收集 (offsets, values, vocab_size, col_idx)，
            # 然后单次遍历填充缓冲区。
            col_data = []
            for ci, slot, vs in side_plan:
                col = batch.column(ci)
                col_data.append(
                    (col.offsets.to_numpy(), col.values.to_numpy(), vs, ci)
                )

            for c, (offs, vals, vs, ci) in enumerate(col_data):
                for i in range(B):
                    s = int(offs[i])
                    e = int(offs[i + 1])
                    rl = e - s
                    if rl <= 0:
                        continue
                    ul = min(rl, max_len)
                    out[i, c, :ul] = vals[s : s + ul]
                    if ul > lengths[i]:
                        lengths[i] = ul

            # 值 <= 0 -> 0。
            out[out <= 0] = 0

            # 按每个特征的 vocab_size 检查越界值。
            # vs==0 表示没有 vocab 信息；将整个切片强制置 0，确保
            # 模型侧 1 槽 Embedding 永远不会被索引越界。
            for c, (_, _, vs, ci) in enumerate(col_data):
                slice_c = out[:, c, :]
                if vs > 0:
                    self._record_oob(f"seq_{domain}", ci, slice_c, vs)
                else:
                    slice_c[:] = 0

            result[domain] = torch.from_numpy(out.copy())
            result[f"{domain}_len"] = torch.from_numpy(lengths.copy())

            # 时间分桶。
            time_bucket = self._buf_seq_tb[domain][:B]
            time_bucket[:] = 0
            if ts_ci is not None:
                ts_col = batch.column(ts_ci)
                ts_offs = ts_col.offsets.to_numpy()
                ts_vals = ts_col.values.to_numpy()
                # 将时间戳填充为形状 (B, max_len)。
                ts_padded = np.zeros((B, max_len), dtype=np.int64)
                for i in range(B):
                    s = int(ts_offs[i])
                    e = int(ts_offs[i + 1])
                    rl = e - s
                    if rl <= 0:
                        continue
                    ul = min(rl, max_len)
                    ts_padded[i, :ul] = ts_vals[s : s + ul]

                ts_expanded = timestamps.reshape(-1, 1)
                time_diff = np.maximum(ts_expanded - ts_padded, 0)
                # np.searchsorted 返回 [0, len(BUCKET_BOUNDARIES)] 范围内的值。
                # +1 后名义范围为 [1, len(BUCKET_BOUNDARIES)+1]；
                # 只有当 time_diff 超过最大边界（约 1 年）时，
                # 才会出现上界，并会索引超过
                # nn.Embedding(NUM_TIME_BUCKETS=len(BUCKET_BOUNDARIES)+1)。
                # 将原始结果裁剪到 [0, len(BUCKET_BOUNDARIES)-1]，使最终
                # bucket id（+1 后）保持在 [1, len(BUCKET_BOUNDARIES)] 内，
                # 并始终是有效的 Embedding 索引。超过最大边界的时间差
                # 会合并到最后一个桶。
                raw_buckets = np.clip(
                    np.searchsorted(BUCKET_BOUNDARIES, time_diff.ravel()),
                    0,
                    len(BUCKET_BOUNDARIES) - 1,
                )
                buckets = raw_buckets.reshape(B, max_len) + 1
                buckets[ts_padded == 0] = 0
                time_bucket[:] = buckets

            result[f"{domain}_time_bucket"] = torch.from_numpy(
                time_bucket.copy()
            )

        return result


def get_pcvr_data(
    data_dir: str,
    schema_path: str,
    batch_size: int = 256,
    valid_ratio: float = 0.1,
    train_ratio: float = 1.0,
    num_workers: int = 16,
    buffer_batches: int = 20,
    shuffle_train: bool = True,
    seed: int = 42,
    clip_vocab: bool = True,
    seq_max_lens: Optional[Dict[str, int]] = None,
    **kwargs: Any,
) -> Tuple[DataLoader, DataLoader, PCVRParquetDataset]:
    """从原始多列 Parquet 文件创建 train / valid DataLoader。

    默认将按 ``glob`` 返回的文件顺序，把最后 ``valid_ratio`` 比例的 Row Group 作为验证集。当 ``split_method='timestamp'`` 时，改用样本级 ``timestamp`` 分位点切分；如果 parquet 物理顺序不按时间排列，这种方式更安全。

    返回:
        元组 ``(train_loader, valid_loader, train_dataset)``。返回第三个元素是为了让调用方访问构造模型所需的特征 schema（``user_int_schema``、``item_int_schema`` 等）。
    """
    random.seed(seed)

    import glob as _glob

    pq_files = sorted(_glob.glob(os.path.join(data_dir, "*.parquet")))

    rg_info = []
    for f in pq_files:
        pf = pq.ParquetFile(f)
        for i in range(pf.metadata.num_row_groups):
            rg_info.append((f, i, pf.metadata.row_group(i).num_rows))
    total_rgs = len(rg_info)

    n_valid_rgs = max(1, int(total_rgs * valid_ratio))
    n_train_rgs = total_rgs - n_valid_rgs

    # train_ratio：仅使用训练 Row Group 的前 N%。
    if train_ratio < 1.0:
        n_train_rgs = max(1, int(n_train_rgs * train_ratio))
        logging.info(
            f"train_ratio={train_ratio}: using {n_train_rgs} train Row Groups"
        )

    train_rows = sum(r[2] for r in rg_info[:n_train_rgs])
    valid_rows = sum(r[2] for r in rg_info[n_train_rgs:])

    logging.info(
        f"Row Group split: {n_train_rgs} train ({train_rows} rows), "
        f"{n_valid_rgs} valid ({valid_rows} rows)"
    )

    train_dataset = PCVRParquetDataset(
        parquet_path=data_dir,
        schema_path=schema_path,
        batch_size=batch_size,
        seq_max_lens=seq_max_lens,
        shuffle=shuffle_train,
        buffer_batches=buffer_batches,
        row_group_range=(0, n_train_rgs),
        clip_vocab=clip_vocab,
    )

    use_cuda = torch.cuda.is_available()
    _train_kw = {}
    if num_workers > 0:
        _train_kw["persistent_workers"] = True
        _train_kw["prefetch_factor"] = 2

    train_loader = DataLoader(
        train_dataset,
        batch_size=None,
        num_workers=num_workers,
        pin_memory=use_cuda,
        **_train_kw,
    )

    valid_dataset = PCVRParquetDataset(
        parquet_path=data_dir,
        schema_path=schema_path,
        batch_size=batch_size,
        seq_max_lens=seq_max_lens,
        shuffle=False,
        buffer_batches=0,
        row_group_range=(n_train_rgs, total_rgs),
        clip_vocab=clip_vocab,
    )
    valid_loader = DataLoader(
        valid_dataset, batch_size=None, num_workers=0, pin_memory=use_cuda
    )

    logging.info(
        f"Parquet train: {train_rows} rows, valid: {valid_rows} rows, "
        f"batch_size={batch_size}, buffer_batches={buffer_batches}"
    )

    return train_loader, valid_loader, train_dataset

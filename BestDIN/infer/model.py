"""DIN + MLP model for PCVR prediction.

This branch intentionally keeps one model path only:
sequence interests are pooled with DIN attention against the candidate item
representation, then a compact MLP predicts the logit.
"""

import logging
from typing import Any, Dict, List, NamedTuple, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


USER_MAIN_UE_FIDS = {61, 87}
USER_PAIR_FIDS = {62, 63, 64, 65, 66}
USER_SMALL_UE_FIDS = {89, 90, 91}
ITEM_HIERARCHY_FIDS = (83, 84, 85)
ACTIVITY_FEATURES_PER_DOMAIN = 8
SAMPLE_TIME_EMBED_FEATURES = 2
SAMPLE_TIME_DENSE_FEATURES = 4
SEQ_TIME_FEATURE_COUNT = 6
SEQ_RECENT_WINDOW_COUNT = 11
SEQ_RECENCY_ATTENTION_BIAS = 0.12
SEQ_HORIZON_RECENCY_ATTENTION_BIAS = 0.18
REINIT_PROTECTED_VOCAB_SENTINEL = -1
HASH_MODULUS = 2_147_483_647
HASH_SLOT_MIXER = 1_000_003
HASH_MULTIPLIERS = [
    1_548_583,
    15_485_863,
    32_452_843,
    49_979_687,
    67_867_967,
    86_028_121,
    104_395_303,
    122_949_823,
]
HASH_SECOND_MULTIPLIERS = [
    13_036_657,
    29_998_001,
    47_932_009,
    65_785_237,
    83_492_513,
    101_125_021,
    118_759_333,
    136_391_201,
]
HASH_BIASES = [
    97_531,
    314_159,
    592_657,
    897_931,
    1_234_577,
    1_618_033,
    2_000_003,
    2_718_281,
]
HASH_SECOND_BIASES = [
    43_721,
    271_829,
    424_243,
    662_607,
    880_301,
    1_048_573,
    1_414_213,
    1_732_051,
]


def _stable_hash_indices(
    values: torch.Tensor,
    bucket_count: int,
    slot: int,
    fid: int,
    lane: int,
) -> torch.Tensor:
    """Independent universal hashes for multi-hash sequence embeddings."""
    mixed = torch.remainder(values.long(), HASH_MODULUS)
    feature_key = int(fid if fid > 0 else slot + 1)
    mixed = torch.remainder(
        mixed * HASH_MULTIPLIERS[lane]
        + HASH_BIASES[lane]
        + feature_key * HASH_SLOT_MIXER,
        HASH_MODULUS,
    )
    mixed = torch.remainder(
        mixed * HASH_SECOND_MULTIPLIERS[lane] + HASH_SECOND_BIASES[lane],
        HASH_MODULUS,
    )
    return torch.remainder(mixed, int(bucket_count) - 1) + 1


class ModelInput(NamedTuple):
    user_int_feats: torch.Tensor
    item_int_feats: torch.Tensor
    user_dense_feats: torch.Tensor
    item_dense_feats: torch.Tensor
    sample_time_feats: torch.Tensor
    activity_feats: torch.Tensor
    seq_data: Dict[str, torch.Tensor]
    seq_lens: Dict[str, torch.Tensor]


class SparseFeatureEncoder(nn.Module):
    """Embed flattened sparse features and project them to one dense token."""

    def __init__(
        self,
        feature_specs: List[Tuple[int, int, int]],
        emb_dim: int,
        d_model: int,
        emb_skip_threshold: int = 0,
        hash_config: Optional[Dict[int, Dict[str, Any]]] = None,
    ) -> None:
        super().__init__()
        self.feature_specs = feature_specs
        self.hash_config = hash_config or {}
        self.emb_index: List[int] = []
        self.embs = nn.ModuleList()
        self.hash_embs = nn.ModuleList()
        self.hash_index: Dict[int, Dict[str, int]] = {}

        for spec_index, (vocab_size, _, _) in enumerate(feature_specs):
            if spec_index in self.hash_config:
                cfg = self.hash_config[spec_index]
                H = int(cfg["H"])
                k = int(cfg["k"])
                fid = int(cfg.get("fid", spec_index + 1))
                if H <= 1:
                    raise ValueError(
                        f"sparse hash H must be > 1, got {H} for spec_index={spec_index}"
                    )
                if k <= 0 or emb_dim % k != 0:
                    raise ValueError(
                        f"sparse hash k must divide emb_dim, got emb_dim={emb_dim}, "
                        f"k={k}, spec_index={spec_index}"
                    )
                if k > len(HASH_MULTIPLIERS):
                    raise ValueError(
                        f"sparse hash k={k} exceeds supported lanes={len(HASH_MULTIPLIERS)}"
                    )
                start = len(self.hash_embs)
                chunk_dim = emb_dim // k
                for _ in range(k):
                    self.hash_embs.append(
                        nn.Embedding(H, chunk_dim, padding_idx=0)
                    )
                self.hash_index[spec_index] = {
                    "start": start,
                    "H": H,
                    "k": k,
                    "fid": fid,
                }
                self.emb_index.append(-1)
                continue
            if vocab_size <= 0 or (
                emb_skip_threshold > 0 and vocab_size > emb_skip_threshold
            ):
                self.emb_index.append(-1)
                continue
            self.emb_index.append(len(self.embs))
            self.embs.append(
                nn.Embedding(vocab_size + 1, emb_dim, padding_idx=0)
            )

        input_dim = max(1, len(feature_specs) * emb_dim)
        self.proj = nn.Sequential(
            nn.Linear(input_dim, d_model), nn.LayerNorm(d_model)
        )
        self.emb_dim = emb_dim

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        batch_size = feats.shape[0]
        pieces: List[torch.Tensor] = []
        for spec_index, (vocab_size, offset, length) in enumerate(
            self.feature_specs
        ):
            emb_real_idx = self.emb_index[spec_index]
            if emb_real_idx < 0:
                if spec_index in self.hash_index:
                    hash_info = self.hash_index[spec_index]
                    if length == 1:
                        values = feats[:, offset].long().clamp_min(0)
                        hash_parts = []
                        for j in range(hash_info["k"]):
                            hash_idx = _stable_hash_indices(
                                values,
                                hash_info["H"],
                                spec_index,
                                hash_info["fid"],
                                j,
                            )
                            hash_idx = torch.where(values == 0, 0, hash_idx)
                            hash_parts.append(
                                self.hash_embs[hash_info["start"] + j](hash_idx)
                            )
                        pieces.append(torch.cat(hash_parts, dim=-1))
                        continue

                    values = feats[:, offset : offset + length].long().clamp_min(0)
                    mask = (values != 0).float().unsqueeze(-1)
                    hash_parts = []
                    for j in range(hash_info["k"]):
                        hash_idx = _stable_hash_indices(
                            values,
                            hash_info["H"],
                            spec_index,
                            hash_info["fid"],
                            j,
                        )
                        hash_idx = torch.where(values == 0, 0, hash_idx)
                        hash_parts.append(
                            self.hash_embs[hash_info["start"] + j](hash_idx)
                        )
                    embedded = torch.cat(hash_parts, dim=-1)
                    denom = mask.sum(dim=1).clamp_min(1.0)
                    pieces.append((embedded * mask).sum(dim=1) / denom)
                    continue
                pieces.append(
                    feats.new_zeros(
                        batch_size, self.emb_dim, dtype=torch.float32
                    )
                )
                continue

            emb = self.embs[emb_real_idx]
            if length == 1:
                values = feats[:, offset].long().clamp(min=0, max=vocab_size)
                pieces.append(emb(values))
                continue

            values = (
                feats[:, offset : offset + length]
                .long()
                .clamp(min=0, max=vocab_size)
            )
            embedded = emb(values)
            mask = (values != 0).float().unsqueeze(-1)
            denom = mask.sum(dim=1).clamp_min(1.0)
            pieces.append((embedded * mask).sum(dim=1) / denom)

        if not pieces:
            return feats.new_zeros(
                batch_size, self.proj[0].out_features, dtype=torch.float32
            )
        return F.silu(self.proj(torch.cat(pieces, dim=-1)))

    def non_hash_sparse_parameters(self) -> List[nn.Parameter]:
        return [emb.weight for emb in self.embs]

    def hash_sparse_parameters(self) -> List[nn.Parameter]:
        return [emb.weight for emb in self.hash_embs]

    def sparse_parameters(self) -> List[nn.Parameter]:
        return self.non_hash_sparse_parameters() + self.hash_sparse_parameters()

    def reinit_specs(self) -> List[Tuple[int, nn.Embedding]]:
        specs: List[Tuple[int, nn.Embedding]] = []
        for spec_index, (vocab_size, _, _) in enumerate(self.feature_specs):
            if spec_index in self.hash_index:
                hash_info = self.hash_index[spec_index]
                for j in range(hash_info["k"]):
                    specs.append(
                        (int(vocab_size), self.hash_embs[hash_info["start"] + j])
                    )
                continue
            emb_real_idx = self.emb_index[spec_index]
            if emb_real_idx < 0:
                continue
            specs.append((int(vocab_size), self.embs[emb_real_idx]))
        return specs

    @torch.no_grad()
    def reinit_high_cardinality_params(
        self, cardinality_threshold: int
    ) -> set[int]:
        reinit_ptrs: set[int] = set()
        for vocab_size, emb in self.reinit_specs():
            if int(vocab_size) <= int(cardinality_threshold):
                continue
            nn.init.xavier_normal_(emb.weight.data)
            emb.weight.data[0, :] = 0
            reinit_ptrs.add(emb.weight.data_ptr())
        return reinit_ptrs


class DenseFeatureEncoder(nn.Module):
    """Project dense feature blocks to the model dimension."""

    def __init__(self, input_dim: int, d_model: int) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        if self.input_dim > 0:
            self.proj = nn.Sequential(
                nn.Linear(self.input_dim, d_model), nn.LayerNorm(d_model)
            )
        else:
            self.proj = None
        self.d_model = d_model

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        if self.proj is None:
            return feats.new_zeros(
                feats.shape[0], self.d_model, dtype=torch.float32
            )
        return F.silu(self.proj(feats.float()))


class DenseFieldGroupEncoder(nn.Module):
    """Project a named group of user_dense fields field-by-field."""

    def __init__(
        self, field_specs: List[Tuple[int, int, int]], d_model: int
    ) -> None:
        super().__init__()
        self.field_specs = field_specs
        self.d_model = d_model
        self.field_projs = nn.ModuleList(
            [
                nn.Sequential(nn.Linear(length, d_model), nn.LayerNorm(d_model))
                for _, _, length in field_specs
            ]
        )
        self.fuse = (
            nn.Sequential(
                nn.Linear(d_model * len(field_specs), d_model),
                nn.LayerNorm(d_model),
            )
            if field_specs
            else None
        )

    def forward(self, dense_feats: torch.Tensor) -> torch.Tensor:
        if self.fuse is None:
            return dense_feats.new_zeros(
                dense_feats.shape[0], self.d_model, dtype=torch.float32
            )

        pieces: List[torch.Tensor] = []
        for (_, offset, length), proj in zip(
            self.field_specs, self.field_projs
        ):
            field = dense_feats[:, offset : offset + length].float()
            pieces.append(F.silu(proj(field)))
        return F.silu(self.fuse(torch.cat(pieces, dim=-1)))


class AlignedUserPairEncoder(nn.Module):
    """Encode aligned user_int and user_dense fields for fids 62 to 66."""

    def __init__(
        self,
        dense_feature_specs: List[Tuple[int, int, int]],
        int_feature_specs: List[Tuple[int, int, int, int]],
        emb_dim: int,
        d_model: int,
        emb_skip_threshold: int = 0,
    ) -> None:
        super().__init__()
        dense_by_fid = {
            fid: (offset, length) for fid, offset, length in dense_feature_specs
        }
        int_by_fid = {
            fid: (vocab_size, offset, length)
            for fid, vocab_size, offset, length in int_feature_specs
        }

        self.pair_specs: List[Tuple[int, int, int, int, int]] = []
        for fid in sorted(USER_PAIR_FIDS):
            if fid not in dense_by_fid or fid not in int_by_fid:
                raise ValueError(
                    f"user fid {fid} must exist in both user_int and user_dense for pair encoding"
                )
            dense_offset, dense_length = dense_by_fid[fid]
            vocab_size, int_offset, int_length = int_by_fid[fid]
            if int_length != dense_length:
                raise ValueError(
                    f"user fid {fid} has mismatched int/dense lengths: "
                    f"int={int_length}, dense={dense_length}"
                )
            self.pair_specs.append(
                (fid, vocab_size, int_offset, dense_offset, dense_length)
            )

        self.d_model = d_model
        self.emb_dim = emb_dim
        self.emb_index: List[int] = []
        self.int_embs = nn.ModuleList()
        self.dense_projs = nn.ModuleList()
        self.pair_projs = nn.ModuleList()

        for _, vocab_size, _, _, length in self.pair_specs:
            if vocab_size <= 0 or (
                emb_skip_threshold > 0 and vocab_size > emb_skip_threshold
            ):
                self.emb_index.append(-1)
            else:
                self.emb_index.append(len(self.int_embs))
                self.int_embs.append(
                    nn.Embedding(vocab_size + 1, emb_dim, padding_idx=0)
                )
            self.dense_projs.append(
                nn.Sequential(nn.Linear(length, d_model), nn.LayerNorm(d_model))
            )
            self.pair_projs.append(
                nn.Sequential(
                    nn.Linear(d_model * 4 + 1, d_model), nn.LayerNorm(d_model)
                )
            )

        self.int_token_proj = nn.Sequential(
            nn.Linear(emb_dim, d_model), nn.LayerNorm(d_model)
        )
        self.fuse = nn.Sequential(
            nn.Linear(d_model * len(self.pair_specs), d_model),
            nn.LayerNorm(d_model),
        )

    def forward(
        self, int_feats: torch.Tensor, dense_feats: torch.Tensor
    ) -> torch.Tensor:
        field_tokens: List[torch.Tensor] = []
        for spec_index, (
            _,
            vocab_size,
            int_offset,
            dense_offset,
            length,
        ) in enumerate(self.pair_specs):
            int_values = (
                int_feats[:, int_offset : int_offset + length]
                .long()
                .clamp(min=0, max=vocab_size)
            )
            dense_values = dense_feats[
                :, dense_offset : dense_offset + length
            ].float()

            emb_real_idx = self.emb_index[spec_index]
            if emb_real_idx < 0:
                int_token = dense_feats.new_zeros(
                    dense_feats.shape[0], self.d_model, dtype=torch.float32
                )
            else:
                embedded = self.int_embs[emb_real_idx](int_values)
                mask = (int_values != 0).float().unsqueeze(-1)
                denom = mask.sum(dim=1).clamp_min(1.0)
                int_token = F.silu(
                    self.int_token_proj((embedded * mask).sum(dim=1) / denom)
                )

            dense_token = F.silu(self.dense_projs[spec_index](dense_values))
            present = (
                ((int_values != 0) | (dense_values.abs() > 0))
                .any(dim=1, keepdim=True)
                .to(dtype=dense_values.dtype)
            )
            pair_input = torch.cat(
                [
                    int_token,
                    dense_token,
                    int_token * dense_token,
                    torch.abs(int_token - dense_token),
                    present,
                ],
                dim=-1,
            )
            field_tokens.append(F.silu(self.pair_projs[spec_index](pair_input)))

        return F.silu(self.fuse(torch.cat(field_tokens, dim=-1)))

    def embedding_modules(self) -> List[nn.Embedding]:
        return list(self.int_embs)

    def sparse_parameters(self) -> List[nn.Parameter]:
        return [emb.weight for emb in self.int_embs]

    def reinit_specs(self) -> List[Tuple[int, nn.Embedding]]:
        specs: List[Tuple[int, nn.Embedding]] = []
        for spec_index, (_, vocab_size, _, _, _) in enumerate(self.pair_specs):
            emb_real_idx = self.emb_index[spec_index]
            if emb_real_idx < 0:
                continue
            specs.append((int(vocab_size), self.int_embs[emb_real_idx]))
        return specs

    @torch.no_grad()
    def reinit_high_cardinality_params(
        self, cardinality_threshold: int
    ) -> set[int]:
        reinit_ptrs: set[int] = set()
        for vocab_size, emb in self.reinit_specs():
            if int(vocab_size) <= int(cardinality_threshold):
                continue
            nn.init.xavier_normal_(emb.weight.data)
            emb.weight.data[0, :] = 0
            reinit_ptrs.add(emb.weight.data_ptr())
        return reinit_ptrs


class SeparatedUserDenseEncoder(nn.Module):
    """Separate UE vectors from aligned int/dense pair features."""

    def __init__(
        self,
        dense_feature_specs: List[Tuple[int, int, int]],
        int_pair_feature_specs: List[Tuple[int, int, int, int]],
        emb_dim: int,
        d_model: int,
        emb_skip_threshold: int = 0,
    ) -> None:
        super().__init__()
        accounted_fids = USER_MAIN_UE_FIDS | USER_PAIR_FIDS | USER_SMALL_UE_FIDS
        unknown_fids = sorted(
            fid
            for fid, _, _ in dense_feature_specs
            if fid not in accounted_fids
        )
        if unknown_fids:
            raise ValueError(
                f"unassigned user_dense fids for separated encoder: {unknown_fids}"
            )

        main_ue_specs = [
            spec for spec in dense_feature_specs if spec[0] in USER_MAIN_UE_FIDS
        ]
        small_ue_specs = [
            spec
            for spec in dense_feature_specs
            if spec[0] in USER_SMALL_UE_FIDS
        ]

        self.main_ue_encoder = DenseFieldGroupEncoder(main_ue_specs, d_model)
        self.pair_encoder = AlignedUserPairEncoder(
            dense_feature_specs=dense_feature_specs,
            int_feature_specs=int_pair_feature_specs,
            emb_dim=emb_dim,
            d_model=d_model,
            emb_skip_threshold=emb_skip_threshold,
        )
        self.small_ue_encoder = DenseFieldGroupEncoder(small_ue_specs, d_model)
        self.fuse = nn.Sequential(
            nn.Linear(d_model * 3, d_model), nn.LayerNorm(d_model)
        )

    def forward(
        self, dense_feats: torch.Tensor, int_feats: torch.Tensor
    ) -> torch.Tensor:
        return F.silu(
            self.fuse(
                torch.cat(
                    [
                        self.main_ue_encoder(dense_feats),
                        self.pair_encoder(int_feats, dense_feats),
                        self.small_ue_encoder(dense_feats),
                    ],
                    dim=-1,
                )
            )
        )

    def embedding_modules(self) -> List[nn.Embedding]:
        return self.pair_encoder.embedding_modules()

    def sparse_parameters(self) -> List[nn.Parameter]:
        return self.pair_encoder.sparse_parameters()

    @torch.no_grad()
    def reinit_high_cardinality_params(
        self, cardinality_threshold: int
    ) -> set[int]:
        return self.pair_encoder.reinit_high_cardinality_params(
            cardinality_threshold
        )


class SampleTimeEncoder(nn.Module):
    """Encode sample-level periodic and monotonic time context."""

    def __init__(self, emb_dim: int, d_model: int) -> None:
        super().__init__()
        self.hour_emb = nn.Embedding(25, emb_dim, padding_idx=0)
        self.weekday_emb = nn.Embedding(8, emb_dim, padding_idx=0)
        self.proj = nn.Sequential(
            nn.Linear(
                emb_dim * SAMPLE_TIME_EMBED_FEATURES
                + SAMPLE_TIME_DENSE_FEATURES,
                d_model,
            ),
            nn.LayerNorm(d_model),
        )

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        hour = feats[:, 0].long().clamp(min=0, max=24)
        weekday = feats[:, 1].long().clamp(min=0, max=7)
        dense_time = feats[:, 2 : 2 + SAMPLE_TIME_DENSE_FEATURES].to(
            dtype=self.hour_emb.weight.dtype
        )
        return F.silu(
            self.proj(
                torch.cat(
                    [
                        self.hour_emb(hour),
                        self.weekday_emb(weekday),
                        dense_time,
                    ],
                    dim=-1,
                )
            )
        )

    def sparse_parameters(self) -> List[nn.Parameter]:
        return [
            self.hour_emb.weight,
            self.weekday_emb.weight,
        ]

    def reinit_specs(self) -> List[Tuple[int, nn.Embedding]]:
        return [
            (REINIT_PROTECTED_VOCAB_SENTINEL, self.hour_emb),
            (REINIT_PROTECTED_VOCAB_SENTINEL, self.weekday_emb),
        ]

    @torch.no_grad()
    def reinit_high_cardinality_params(
        self, cardinality_threshold: int
    ) -> set[int]:
        reinit_ptrs: set[int] = set()
        for vocab_size, emb in self.reinit_specs():
            if int(vocab_size) <= int(cardinality_threshold):
                continue
            nn.init.xavier_normal_(emb.weight.data)
            emb.weight.data[0, :] = 0
            reinit_ptrs.add(emb.weight.data_ptr())
        return reinit_ptrs


class SequenceFeatureEncoder(nn.Module):
    """Embed per-position sequence side-info into token representations."""

    def __init__(
        self,
        vocab_sizes: List[int],
        emb_dim: int,
        d_model: int,
        emb_skip_threshold: int = 0,
        hash_config: Optional[Dict[int, Dict[str, Any]]] = None,
    ) -> None:
        super().__init__()
        self.vocab_sizes = vocab_sizes
        self.hash_config = hash_config or {}
        self.emb_index: List[int] = []
        self.embs = nn.ModuleList()
        self.hash_embs = nn.ModuleList()
        self.hash_index: Dict[int, Dict[str, int]] = {}
        for slot, vocab_size in enumerate(vocab_sizes):
            if slot in self.hash_config:
                cfg = self.hash_config[slot]
                H = int(cfg["H"])
                k = int(cfg["k"])
                fid = int(cfg.get("fid", slot + 1))
                if H <= 1:
                    raise ValueError(f"seq hash H must be > 1, got {H} for slot={slot}")
                if k <= 0 or emb_dim % k != 0:
                    raise ValueError(
                        f"seq hash k must divide emb_dim, got emb_dim={emb_dim}, k={k}, slot={slot}"
                    )
                if k > len(HASH_MULTIPLIERS):
                    raise ValueError(
                        f"seq hash k={k} exceeds supported lanes={len(HASH_MULTIPLIERS)}"
                    )
                start = len(self.hash_embs)
                chunk_dim = emb_dim // k
                for _ in range(k):
                    self.hash_embs.append(
                        nn.Embedding(H, chunk_dim, padding_idx=0)
                    )
                self.hash_index[slot] = {
                    "start": start,
                    "H": H,
                    "k": k,
                    "fid": fid,
                }
                self.emb_index.append(-1)
                continue
            if vocab_size <= 0 or (
                emb_skip_threshold > 0 and vocab_size > emb_skip_threshold
            ):
                self.emb_index.append(-1)
                continue
            self.emb_index.append(len(self.embs))
            self.embs.append(
                nn.Embedding(vocab_size + 1, emb_dim, padding_idx=0)
            )

        input_dim = max(1, len(vocab_sizes) * emb_dim)
        self.proj = nn.Sequential(
            nn.Linear(input_dim, d_model), nn.LayerNorm(d_model)
        )
        self.emb_dim = emb_dim

    def forward(self, seq: torch.Tensor) -> torch.Tensor:
        batch_size, _, seq_len = seq.shape
        pieces: List[torch.Tensor] = []
        for slot, vocab_size in enumerate(self.vocab_sizes):
            emb_real_idx = self.emb_index[slot]
            if emb_real_idx < 0:
                if slot in self.hash_index:
                    hash_info = self.hash_index[slot]
                    values = seq[:, slot, :].long().clamp_min(0)
                    hash_parts: List[torch.Tensor] = []
                    for j in range(hash_info["k"]):
                        hash_idx = _stable_hash_indices(
                            values,
                            hash_info["H"],
                            slot,
                            hash_info["fid"],
                            j,
                        )
                        hash_idx = torch.where(values == 0, 0, hash_idx)
                        hash_parts.append(
                            self.hash_embs[hash_info["start"] + j](hash_idx)
                        )
                    pieces.append(torch.cat(hash_parts, dim=-1))
                    continue
                pieces.append(
                    seq.new_zeros(
                        batch_size, seq_len, self.emb_dim, dtype=torch.float32
                    )
                )
                continue
            values = seq[:, slot, :].long().clamp(min=0, max=vocab_size)
            pieces.append(self.embs[emb_real_idx](values))

        if not pieces:
            concat = seq.new_zeros(batch_size, seq_len, 1, dtype=torch.float32)
        else:
            concat = torch.cat(pieces, dim=-1)
        return F.silu(self.proj(concat))

    def non_hash_sparse_parameters(self) -> List[nn.Parameter]:
        return [emb.weight for emb in self.embs]

    def hash_sparse_parameters(self) -> List[nn.Parameter]:
        return [emb.weight for emb in self.hash_embs]

    def sparse_parameters(self) -> List[nn.Parameter]:
        return self.non_hash_sparse_parameters() + self.hash_sparse_parameters()

    def reinit_specs(self) -> List[Tuple[int, nn.Embedding]]:
        specs: List[Tuple[int, nn.Embedding]] = []
        raw_slot_count = len(self.vocab_sizes) - SEQ_TIME_FEATURE_COUNT
        for slot, vocab_size in enumerate(self.vocab_sizes):
            effective_vocab = int(vocab_size)
            if slot >= raw_slot_count:
                effective_vocab = REINIT_PROTECTED_VOCAB_SENTINEL
            if slot in self.hash_index:
                hash_info = self.hash_index[slot]
                for j in range(hash_info["k"]):
                    specs.append(
                        (effective_vocab, self.hash_embs[hash_info["start"] + j])
                    )
                continue
            emb_real_idx = self.emb_index[slot]
            if emb_real_idx < 0:
                continue
            specs.append((effective_vocab, self.embs[emb_real_idx]))
        return specs

    @torch.no_grad()
    def reinit_high_cardinality_params(
        self, cardinality_threshold: int
    ) -> set[int]:
        reinit_ptrs: set[int] = set()
        for vocab_size, emb in self.reinit_specs():
            if int(vocab_size) <= int(cardinality_threshold):
                continue
            nn.init.xavier_normal_(emb.weight.data)
            emb.weight.data[0, :] = 0
            reinit_ptrs.add(emb.weight.data_ptr())
        return reinit_ptrs


class DINAttention(nn.Module):
    """Candidate-conditioned attention over one behavior sequence."""

    def __init__(
        self, d_model: int, hidden_mult: int = 2, dropout: float = 0.0
    ) -> None:
        super().__init__()
        hidden_dim = d_model * hidden_mult
        self.mlp = nn.Sequential(
            nn.Linear(d_model * 4, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        item_query: torch.Tensor,
        seq_tokens: torch.Tensor,
        seq_lens: torch.Tensor,
        attention_bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size, seq_len, d_model = seq_tokens.shape
        query = item_query.unsqueeze(1).expand(-1, seq_len, -1)
        attn_input = torch.cat(
            [query, seq_tokens, query - seq_tokens, query * seq_tokens], dim=-1
        )
        scores = self.mlp(attn_input).squeeze(-1)
        if attention_bias is not None:
            scores = scores + attention_bias.to(dtype=scores.dtype)
        positions = torch.arange(seq_len, device=seq_tokens.device).unsqueeze(0)
        valid = positions < seq_lens.long().unsqueeze(1)
        scores = scores.masked_fill(~valid, -1e9)
        weights = torch.softmax(scores, dim=1) * valid.float()
        weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-6)
        return torch.bmm(weights.unsqueeze(1), seq_tokens).squeeze(1)


class PCVRHyFormer(nn.Module):
    """Compatibility name for the DIN + MLP model used by train.py/infer.py."""

    def __init__(
        self,
        user_int_feature_specs: List[Tuple[int, int, int]],
        user_dense_feature_specs: List[Tuple[int, int, int]],
        user_int_pair_feature_specs: List[Tuple[int, int, int, int]],
        item_int_feature_specs: List[Tuple[int, int, int]],
        item_int_feature_specs_with_fid: List[Tuple[int, int, int, int]],
        user_dense_dim: int,
        item_dense_dim: int,
        seq_vocab_sizes: Dict[str, List[int]],
        d_model: int = 64,
        emb_dim: int = 64,
        hidden_mult: int = 4,
        dropout_rate: float = 0.01,
        action_num: int = 1,
        emb_skip_threshold: int = 0,
        user_hash_config: Optional[Dict[int, Dict[str, Any]]] = None,
        item_hash_config: Optional[Dict[int, Dict[str, Any]]] = None,
        seq_hash_config: Optional[Dict[str, Dict[int, Dict[str, Any]]]] = None,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.action_num = action_num
        self.seq_domains = sorted(seq_vocab_sizes.keys())
        self.user_hash_config = user_hash_config or {}
        self.item_hash_config = item_hash_config or {}
        self.seq_hash_config = seq_hash_config or {}

        self.user_sparse_encoder = SparseFeatureEncoder(
            user_int_feature_specs,
            emb_dim,
            d_model,
            emb_skip_threshold,
            hash_config=self.user_hash_config,
        )
        self.item_sparse_encoder = SparseFeatureEncoder(
            item_int_feature_specs,
            emb_dim,
            d_model,
            emb_skip_threshold,
            hash_config=self.item_hash_config,
        )
        self.user_dense_encoder = SeparatedUserDenseEncoder(
            user_dense_feature_specs,
            user_int_pair_feature_specs,
            emb_dim,
            d_model,
            emb_skip_threshold,
        )
        self.item_dense_encoder = DenseFeatureEncoder(item_dense_dim, d_model)
        self.sample_time_encoder = SampleTimeEncoder(emb_dim, d_model)
        self.activity_encoder = DenseFeatureEncoder(
            len(self.seq_domains) * ACTIVITY_FEATURES_PER_DOMAIN, d_model
        )
        self.seq_encoders = nn.ModuleDict(
            {
                domain: SequenceFeatureEncoder(
                    vocab_sizes,
                    emb_dim,
                    d_model,
                    emb_skip_threshold,
                    hash_config=self.seq_hash_config.get(domain, {}),
                )
                for domain, vocab_sizes in seq_vocab_sizes.items()
            }
        )
        self.seq_attentions = nn.ModuleDict(
            {
                domain: DINAttention(
                    d_model, hidden_mult=2, dropout=dropout_rate
                )
                for domain in self.seq_domains
            }
        )

        readout_dim = d_model * (len(self.seq_domains) + 6)
        hidden_dim = d_model * hidden_mult
        self.readout = nn.Sequential(
            nn.Linear(readout_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, d_model),
            nn.LayerNorm(d_model),
            nn.SiLU(),
            nn.Dropout(dropout_rate),
        )
        self.classifier = nn.Linear(d_model, action_num)
        self._init_params()

    def _init_params(self) -> None:
        for emb in self._embedding_modules():
            nn.init.xavier_normal_(emb.weight.data)
            emb.weight.data[0, :] = 0

    def _embedding_modules(self) -> List[nn.Embedding]:
        modules: List[nn.Embedding] = []
        modules.extend(self.user_sparse_encoder.embs)
        modules.extend(self.user_sparse_encoder.hash_embs)
        modules.extend(self.user_dense_encoder.embedding_modules())
        modules.extend(self.item_sparse_encoder.embs)
        modules.extend(self.item_sparse_encoder.hash_embs)
        modules.extend(
            [
                self.sample_time_encoder.hour_emb,
                self.sample_time_encoder.weekday_emb,
            ]
        )
        for encoder in self.seq_encoders.values():
            modules.extend(encoder.embs)
            modules.extend(encoder.hash_embs)
        return modules

    def get_sparse_params(self) -> List[nn.Parameter]:
        return self.get_non_hash_sparse_params() + self.get_hash_sparse_params()

    def get_non_hash_sparse_params(self) -> List[nn.Parameter]:
        params: List[nn.Parameter] = []
        params.extend(self.user_sparse_encoder.non_hash_sparse_parameters())
        params.extend(self.user_dense_encoder.sparse_parameters())
        params.extend(self.item_sparse_encoder.non_hash_sparse_parameters())
        params.extend(self.sample_time_encoder.sparse_parameters())
        for encoder in self.seq_encoders.values():
            params.extend(encoder.non_hash_sparse_parameters())
        return params

    def get_hash_sparse_params(self) -> List[nn.Parameter]:
        params: List[nn.Parameter] = []
        params.extend(self.user_sparse_encoder.hash_sparse_parameters())
        params.extend(self.item_sparse_encoder.hash_sparse_parameters())
        for encoder in self.seq_encoders.values():
            params.extend(encoder.hash_sparse_parameters())
        return params

    def get_dense_params(self) -> List[nn.Parameter]:
        sparse_ids = {id(p) for p in self.get_sparse_params()}
        return [
            p
            for p in self.parameters()
            if p.requires_grad and id(p) not in sparse_ids
        ]

    def reinit_high_cardinality_params(
        self, cardinality_threshold: int = 1
    ) -> set[int]:
        reinit_ptrs: set[int] = set()
        reinit_count = 0
        kept_count = 0

        def _apply(specs: List[Tuple[int, nn.Embedding]]) -> None:
            nonlocal reinit_count, kept_count
            for vocab_size, emb in specs:
                if int(vocab_size) > int(cardinality_threshold):
                    nn.init.xavier_normal_(emb.weight.data)
                    if emb.padding_idx is not None:
                        emb.weight.data[int(emb.padding_idx), :] = 0
                    reinit_ptrs.add(emb.weight.data_ptr())
                    reinit_count += 1
                else:
                    kept_count += 1

        _apply(self.user_sparse_encoder.reinit_specs())
        _apply(self.user_dense_encoder.pair_encoder.reinit_specs())
        _apply(self.item_sparse_encoder.reinit_specs())
        _apply(self.sample_time_encoder.reinit_specs())
        for encoder in self.seq_encoders.values():
            _apply(encoder.reinit_specs())
        logging.info(
            f"Re-initialized {reinit_count} high-cardinality Embeddings "
            f"(vocab>{int(cardinality_threshold)}), kept {kept_count}"
        )
        return reinit_ptrs

    def _encode_base(
        self, inputs: ModelInput
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        user_repr = self.user_sparse_encoder(
            inputs.user_int_feats
        ) + self.user_dense_encoder(
            inputs.user_dense_feats, inputs.user_int_feats
        )
        item_repr = self.item_sparse_encoder(
            inputs.item_int_feats
        ) + self.item_dense_encoder(inputs.item_dense_feats)
        return user_repr, item_repr

    def _build_recency_attention_bias(
        self, domain: str, seq: torch.Tensor
    ) -> Optional[torch.Tensor]:
        encoder = self.seq_encoders[domain]
        horizon_slot = len(encoder.vocab_sizes) - 1
        recent_slot = horizon_slot - 2
        if recent_slot < 0 or horizon_slot < 0:
            return None
        recent_window = seq[:, recent_slot, :].long()
        horizon_window = seq[:, horizon_slot, :].long()
        valid_recent = recent_window > 0
        valid_horizon = horizon_window > 0
        recency_score = (
            (SEQ_RECENT_WINDOW_COUNT + 1 - recent_window)
            .clamp(min=0, max=SEQ_RECENT_WINDOW_COUNT)
            .to(dtype=torch.float32)
            / float(SEQ_RECENT_WINDOW_COUNT)
        )
        horizon_score = (
            (SEQ_RECENT_WINDOW_COUNT + 1 - horizon_window)
            .clamp(min=0, max=SEQ_RECENT_WINDOW_COUNT)
            .to(dtype=torch.float32)
            / float(SEQ_RECENT_WINDOW_COUNT)
        )
        bias = recency_score.new_zeros(recency_score.shape)
        bias = bias + torch.where(
            valid_recent,
            recency_score * float(SEQ_RECENCY_ATTENTION_BIAS),
            bias,
        )
        bias = bias + torch.where(
            valid_horizon,
            horizon_score * float(SEQ_HORIZON_RECENCY_ATTENTION_BIAS),
            bias.new_zeros(bias.shape),
        )
        return bias

    def _encode_interests(
        self, inputs: ModelInput, item_repr: torch.Tensor
    ) -> List[torch.Tensor]:
        interests: List[torch.Tensor] = []
        for domain in self.seq_domains:
            seq = inputs.seq_data[domain]
            seq_tokens = self.seq_encoders[domain](seq)
            interests.append(
                self.seq_attentions[domain](
                    item_repr,
                    seq_tokens,
                    inputs.seq_lens[domain],
                    attention_bias=self._build_recency_attention_bias(
                        domain, seq
                    ),
                )
            )
        return interests

    def forward(self, inputs: ModelInput) -> torch.Tensor:
        user_repr, item_repr = self._encode_base(inputs)
        interests = self._encode_interests(inputs, item_repr)
        if interests:
            interest_stack = torch.stack(interests, dim=1)
            global_interest = interest_stack.mean(dim=1)
            interest_parts = interests
        else:
            global_interest = torch.zeros_like(item_repr)
            interest_parts = []

        features = torch.cat(
            [
                user_repr,
                item_repr,
                self.sample_time_encoder(inputs.sample_time_feats),
                self.activity_encoder(inputs.activity_feats),
                *interest_parts,
                global_interest * item_repr,
                torch.abs(global_interest - item_repr),
            ],
            dim=-1,
        )
        hidden = self.readout(features)
        return self.classifier(hidden)

    def predict(self, inputs: ModelInput) -> Tuple[torch.Tensor, torch.Tensor]:
        logits = self.forward(inputs)
        return logits, torch.sigmoid(logits)

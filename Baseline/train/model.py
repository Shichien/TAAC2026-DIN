"""PCVRHyFormer：用于点击后转化率预测的混合 Transformer 模型。"""

import logging
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, NamedTuple, Tuple, Optional, Union


class ModelInput(NamedTuple):
    user_int_feats: torch.Tensor
    item_int_feats: torch.Tensor
    user_dense_feats: torch.Tensor
    item_dense_feats: torch.Tensor
    seq_data: dict  # {domain: 张量 [B, S, L]}
    seq_lens: dict  # {domain: 张量 [B]}
    seq_time_buckets: dict  # {domain: 张量 [B, L]}


# ═══════════════════════════════════════════════════════════════════════════════
# 旋转位置编码 (RoPE)
# ═══════════════════════════════════════════════════════════════════════════════


class RotaryEmbedding(nn.Module):
    """预计算并缓存 RoPE 的 cos/sin 值。

    属性:
        dim: 旋转位置编码维度。
        max_seq_len: 缓存支持的最大序列长度。
        base: 旋转编码的基础频率。
    """

    def __init__(
        self, dim: int, max_seq_len: int = 2048, base: float = 10000.0
    ) -> None:
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base

        # 预计算 inv_freq: (dim // 2,)
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # 预计算缓存
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len: int) -> None:
        t = torch.arange(
            seq_len, dtype=self.inv_freq.dtype, device=self.inv_freq.device
        )
        freqs = torch.outer(t, self.inv_freq)  # (seq_len, dim // 2)
        emb = torch.cat([freqs, freqs], dim=-1)  # (seq_len, dim)
        self.register_buffer(
            "cos_cached", emb.cos().unsqueeze(0), persistent=False
        )  # (1, seq_len, dim)
        self.register_buffer(
            "sin_cached", emb.sin().unsqueeze(0), persistent=False
        )  # (1, seq_len, dim)

    def forward(
        self, seq_len: int, device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """计算给定序列长度对应的 cos/sin 值。

        从缓存中返回预计算切片。缓存会在 __init__ 中按 max_seq_len 构建一次；运行时不扩展缓存，从而保持 forward 过程兼容 torch.compile()。
        """
        cos = self.cos_cached[:, :seq_len, :].to(device)
        sin = self.sin_cached[:, :seq_len, :].to(device)
        return cos, sin


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """交换最后一维的前后两半，并对原后半部分取负。"""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat([-x2, x1], dim=-1)


def apply_rope_to_tensor(
    x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> torch.Tensor:
    """对单个张量应用旋转位置编码。

    参数:
        x: (B, num_heads, L, head_dim)。
        cos: (1, L_max, head_dim)，或用于批内不同位置的 (B, L, head_dim)。
        sin: 与 cos 形状相同。

    返回:
        形状为 (B, num_heads, L, head_dim) 的旋转后张量。
    """
    L = x.shape[2]
    cos_ = cos[:, :L, :].unsqueeze(1)  # (*, 1, L, head_dim)
    sin_ = sin[:, :L, :].unsqueeze(1)
    return x * cos_ + rotate_half(x) * sin_


# ═══════════════════════════════════════════════════════════════════════════════
# HyFormer 基础组件
# ═══════════════════════════════════════════════════════════════════════════════


class SwiGLU(nn.Module):
    """SwiGLU 激活：x1 * SiLU(x2)。"""

    def __init__(self, d_model: int, hidden_mult: int = 4) -> None:
        super().__init__()
        hidden_dim = d_model * hidden_mult
        self.fc = nn.Linear(d_model, 2 * hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc(x)
        x1, x2 = x.chunk(2, dim=-1)
        x = x1 * F.silu(x2)
        x = self.fc_out(x)
        return x


class RoPEMultiheadAttention(nn.Module):
    """支持旋转位置编码的多头注意力。

    手动投影 Q/K/V 并重排为多头形状，然后在投影之后、点积之前注入 RoPE。使用 F.scaled_dot_product_attention 做高效计算。
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dropout: float = 0.0,
        rope_on_q: bool = True,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.rope_on_q = rope_on_q
        self.dropout = dropout

        assert d_model % num_heads == 0, (
            "d_model must be divisible by num_heads"
        )

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        self.W_g = nn.Linear(d_model, d_model)

        nn.init.zeros_(self.W_g.weight)
        nn.init.constant_(self.W_g.bias, 1.0)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
        rope_cos: Optional[torch.Tensor] = None,
        rope_sin: Optional[torch.Tensor] = None,
        q_rope_cos: Optional[torch.Tensor] = None,
        q_rope_sin: Optional[torch.Tensor] = None,
        need_weights: bool = False,
    ) -> tuple:
        """计算可选 RoPE 的多头注意力。

        参数:
            query: (B, Lq, D)。
            key: (B, Lk, D)。
            value: (B, Lk, D)。
            key_padding_mask: (B, Lk)，True 表示 padding 位置。
            attn_mask: (Lq, Lk) 或 (B*num_heads, Lq, Lk)，加性 mask。
            rope_cos: (1, L, head_dim)，KV 侧的 RoPE，也会用于 Q，除非提供 q_rope_*。
            rope_sin: 与 rope_cos 形状相同。
            q_rope_cos: (B, Lq, head_dim) 或 (1, Lq, head_dim)，用于带聚集位置的交叉注意力的 Q 侧 RoPE。
            q_rope_sin: 与 q_rope_cos 形状相同。
            need_weights: 兼容性参数，未使用。

        返回:
            (output, None) 元组。
        """
        B, Lq, _ = query.shape
        Lk = key.shape[1]

        # 1. 线性投影
        Q = self.W_q(query)  # (B, Lq, D)
        K = self.W_k(key)  # (B, Lk, D)
        V = self.W_v(value)  # (B, Lk, D)

        # 2. 重排为 (B, num_heads, L, head_dim)
        Q = Q.view(B, Lq, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(B, Lk, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(B, Lk, self.num_heads, self.head_dim).transpose(1, 2)

        # 3. 分别对 Q 和 K 应用 RoPE
        if rope_cos is not None and rope_sin is not None:
            # K 始终使用 rope_cos/rope_sin（KV 侧位置编码）
            K = apply_rope_to_tensor(K, rope_cos, rope_sin)

            if self.rope_on_q:
                # Q 侧：优先使用专用 q_rope_cos/sin（LongerEncoder 交叉注意力中的 top_k 位置）
                q_cos = q_rope_cos if q_rope_cos is not None else rope_cos
                q_sin = q_rope_sin if q_rope_sin is not None else rope_sin
                Q = apply_rope_to_tensor(Q, q_cos, q_sin)

        # 4. 将 key_padding_mask 转换为 SDPA 格式
        sdpa_attn_mask = None
        if key_padding_mask is not None:
            # key_padding_mask: (B, Lk)，True 表示填充
            # SDPA 期望 (B, 1, 1, Lk) bool mask，True 表示可 attend
            sdpa_attn_mask = ~key_padding_mask.unsqueeze(1).unsqueeze(
                2
            )  # (B, 1, 1, Lk)
            sdpa_attn_mask = sdpa_attn_mask.expand(B, self.num_heads, Lq, Lk)

        if attn_mask is not None:
            # attn_mask: 加性 float mask (Lq, Lk)，-inf 表示不可 attend
            # 转为 bool：非 -inf 的位置为 True
            bool_attn = attn_mask == 0  # (Lq, Lk)
            bool_attn = (
                bool_attn.unsqueeze(0)
                .unsqueeze(0)
                .expand(B, self.num_heads, Lq, Lk)
            )
            if sdpa_attn_mask is not None:
                sdpa_attn_mask = sdpa_attn_mask & bool_attn
            else:
                sdpa_attn_mask = bool_attn

        # 5. 缩放点积注意力
        dropout_p = self.dropout if self.training else 0.0
        out = F.scaled_dot_product_attention(
            Q, K, V, attn_mask=sdpa_attn_mask, dropout_p=dropout_p
        )  # (B, num_heads, Lq, head_dim)

        # 将全填充 softmax 产生的 NaN 替换为 0（零向量会通过残差保留原输入）
        out = torch.nan_to_num(out, nan=0.0)

        # 6. 重排回原形状并做输出投影
        out = out.transpose(1, 2).contiguous().view(B, Lq, self.d_model)
        G = self.W_g(query)
        out = out * torch.sigmoid(G)
        out = self.W_o(out)

        return out, None


class CrossAttention(nn.Module):
    """交叉注意力模块。

    Query 来自全局 token（Q token），Key/Value 来自序列 token。只在 KV 侧应用 RoPE（rope_on_q=False）。
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dropout: float = 0.0,
        ln_mode: str = "pre",
    ) -> None:
        super().__init__()
        self.ln_mode = ln_mode

        self.attn = RoPEMultiheadAttention(
            d_model=d_model,
            num_heads=num_heads,
            dropout=dropout,
            rope_on_q=False,
        )

        if ln_mode in ["pre", "post"]:
            self.norm_q = nn.LayerNorm(d_model)
            self.norm_kv = nn.LayerNorm(d_model)

    def forward(
        self,
        query: torch.Tensor,
        key_value: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        rope_cos: Optional[torch.Tensor] = None,
        rope_sin: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """计算 query token 与序列 token 之间的交叉注意力。

        参数:
            query: (B, Nq, D)，query token。
            key_value: (B, L, D)，序列 token。
            key_padding_mask: (B, L)，True 表示 padding 位置。
            rope_cos: (1, L, head_dim)，KV 侧 RoPE 的余弦值。
            rope_sin: (1, L, head_dim)，KV 侧 RoPE 的正弦值。

        返回:
            形状为 (B, Nq, D) 的输出张量。
        """
        residual = query

        if self.ln_mode == "pre":
            query = self.norm_q(query)
            key_value = self.norm_kv(key_value)

        out, _ = self.attn(
            query=query,
            key=key_value,
            value=key_value,
            key_padding_mask=key_padding_mask,
            rope_cos=rope_cos,
            rope_sin=rope_sin,
        )

        out = residual + out

        if self.ln_mode == "post":
            out = self.norm_q(out)

        return out


class RankMixerBlock(nn.Module):
    """HyFormer 查询增强块。

    执行三步:
    1. Token Mixing：无参数张量重排。
    2. 逐 token FFN：共享参数的前馈网络。
    3. 残差连接：Q_boost = Q + Q_e。

    约束：在 'full' 模式下，d_model 必须能被 n_total 整除。
    """

    def __init__(
        self,
        d_model: int,
        n_total: int,  # T = Nq + Nns
        hidden_mult: int = 4,
        dropout: float = 0.0,
        mode: str = "full",  # 'full' | 'ffn_only' | 'none'
    ) -> None:
        super().__init__()
        self.T = n_total
        self.D = d_model
        self.mode = mode

        if mode == "none":
            # 纯恒等映射，不创建子模块
            return

        if mode == "full":
            if d_model % n_total != 0:
                raise ValueError(
                    f"d_model={d_model} must be divisible by T={n_total} for token mixing."
                )
            self.d_sub = d_model // n_total

        # 逐 token FFN（共享参数），同时用于 'full' 和 'ffn_only'
        self.norm = nn.LayerNorm(d_model)
        self.fc1 = nn.Linear(d_model, d_model * hidden_mult)
        self.fc2 = nn.Linear(d_model * hidden_mult, d_model)
        self.dropout = nn.Dropout(dropout)
        # 残差之后做 Post-LN，稳定堆叠 block 的输出
        self.post_norm = nn.LayerNorm(d_model)

    def token_mixing(self, Q: torch.Tensor) -> torch.Tensor:
        """通过 reshape 和 transpose 执行无参数 token mixing。

        步骤:
        1. 将通道切成 T 个子空间：(B, T, D) -> (B, T, T, d_sub)。
        2. 交换 token 轴与子空间轴：(B, token, h, d_sub) -> (B, h, token, d_sub)。
        3. 再展平成：(B, T, D)。

        参数:
            Q: (B, T, D)。

        返回:
            形状为 (B, T, D) 的混合后张量。
        """
        B, T, D = Q.shape

        # (B, T, D) -> (B, T, T, d_sub)
        Q_split = Q.view(B, T, self.T, self.d_sub)

        # (B, token, h, d_sub) -> (B, h, token, d_sub)
        Q_rewired = Q_split.transpose(1, 2).contiguous()

        # (B, T, T, d_sub) -> (B, T, D)
        Q_hat = Q_rewired.view(B, T, D)
        return Q_hat

    def forward(self, Q: torch.Tensor) -> torch.Tensor:
        """应用查询增强：token mixing、FFN 和残差连接。

        参数:
            Q: (B, T, D)，其中 T = Nq + Nns。

        返回:
            形状为 (B, T, D) 的增强后张量。
        """
        if self.mode == "none":
            return Q

        # Token Mixing（无参数重排）或恒等映射
        if self.mode == "full":
            Q_hat = self.token_mixing(Q)
        else:  # 'ffn_only' 模式
            Q_hat = Q

        # 逐 token FFN
        x = self.norm(Q_hat)
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.dropout(x)
        Q_e = self.fc2(x)

        # 来自原始 Q 的残差
        Q_boost = Q + Q_e
        Q_boost = self.post_norm(Q_boost)
        return Q_boost


class MultiSeqQueryGenerator(nn.Module):
    """多序列查询生成模块。

    为每条序列独立生成 Q token:
    对每条序列 i:
        GlobalInfo_i = Concat(F1..FM, MeanPool(Seq_i))
        Q_i = [FFN_{i,1}(GlobalInfo_i), ..., FFN_{i,N}(GlobalInfo_i)]
    """

    def __init__(
        self,
        d_model: int,
        num_ns: int,
        num_queries: int,
        num_sequences: int,
        hidden_mult: int = 4,
    ) -> None:
        super().__init__()
        self.num_queries = num_queries
        self.num_sequences = num_sequences
        self.d_model = d_model

        global_info_dim = (num_ns + 1) * d_model

        # 对 global_info 做 LayerNorm，避免大维度拼接导致梯度爆炸
        self.global_info_norm = nn.LayerNorm(global_info_dim)

        # 每条序列拥有 N 个独立 FFN
        self.query_ffns_per_seq = nn.ModuleList(
            [
                nn.ModuleList(
                    [
                        nn.Sequential(
                            nn.Linear(global_info_dim, d_model * hidden_mult),
                            nn.SiLU(),
                            nn.Linear(d_model * hidden_mult, d_model),
                            nn.LayerNorm(d_model),
                        )
                        for _ in range(num_queries)
                    ]
                )
                for _ in range(num_sequences)
            ]
        )

    def forward(
        self,
        ns_tokens: torch.Tensor,
        seq_tokens_list: list,
        seq_padding_masks: list,
    ) -> list:
        """为每条序列生成 query token。

        参数:
            ns_tokens: (B, M, D)，共享 NS token。
            seq_tokens_list: 长度为 S 的 (B, L_i, D) 张量列表。
            seq_padding_masks: 长度为 S 的 (B, L_i) mask 列表，True 表示 padding。

        返回:
            长度为 S 的 (B, Nq, D) query token 张量列表。
        """
        B = ns_tokens.shape[0]
        ns_flat = ns_tokens.view(B, -1)  # (B, M*D)

        q_tokens_list = []
        for i in range(self.num_sequences):
            # 对 Seq_i 做均值池化
            valid_mask = ~seq_padding_masks[i]  # True 表示有效
            valid_mask_expanded = valid_mask.unsqueeze(
                -1
            ).float()  # (B, L_i, 1)
            seq_sum = (seq_tokens_list[i] * valid_mask_expanded).sum(
                dim=1
            )  # (B, D)
            seq_count = valid_mask_expanded.sum(dim=1).clamp(min=1)  # (B, 1)
            seq_pooled = seq_sum / seq_count  # (B, D)

            # GlobalInfo_i = 拼接(NS_flat, seq_pooled_i)
            global_info = torch.cat(
                [ns_flat, seq_pooled], dim=-1
            )  # (B, (M+1)*D)
            global_info = self.global_info_norm(global_info)

            # 生成 N 个 query token
            queries = [ffn(global_info) for ffn in self.query_ffns_per_seq[i]]
            q_tokens = torch.stack(queries, dim=1)  # (B, Nq, D)
            q_tokens_list.append(q_tokens)

        return q_tokens_list


# ═══════════════════════════════════════════════════════════════════════════════
# 序列编码器
# ═══════════════════════════════════════════════════════════════════════════════


class SwiGLUEncoder(nn.Module):
    """高效的无注意力序列编码器。

    结构：x + Dropout(SwiGLU(LN(x)))。
    """

    def __init__(
        self, d_model: int, hidden_mult: int = 4, dropout: float = 0.0
    ) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.swiglu = SwiGLU(d_model, hidden_mult)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """应用带残差连接的 SwiGLU 编码器。

        参数:
            x: (B, L, D)。
            key_padding_mask: (B, L)，True 表示 padding；该编码器变体不使用它。
            **kwargs: 接收 rope_cos/rope_sin 以及其他未使用参数。

        返回:
            (形状为 (B, L, D) 的输出张量, key_padding_mask) 元组。
        """
        residual = x
        x = self.norm(x)
        x = self.swiglu(x)
        x = self.dropout(x)
        x = residual + x
        return x, key_padding_mask


class TransformerEncoder(nn.Module):
    """带自注意力和 RoPE 的高容量序列编码器。

    结构：标准 Transformer Encoder Layer（Pre-LN）。
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        hidden_mult: int = 4,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.self_attn = RoPEMultiheadAttention(
            d_model=d_model,
            num_heads=num_heads,
            dropout=dropout,
            rope_on_q=True,
        )

        hidden_dim = d_model * hidden_mult
        self.ffn = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, d_model),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        rope_cos: Optional[torch.Tensor] = None,
        rope_sin: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """应用一层 Transformer 编码器。

        参数:
            x: (B, L, D)。
            key_padding_mask: (B, L)，True 表示 padding 位置。
            rope_cos: (1, L, head_dim)，RoPE 余弦值。
            rope_sin: (1, L, head_dim)，RoPE 正弦值。

        返回:
            (形状为 (B, L, D) 的输出张量, key_padding_mask) 元组。
        """
        # 带 RoPE 的自注意力（Pre-LN）
        residual = x
        x = self.norm1(x)
        x, _ = self.self_attn(
            query=x,
            key=x,
            value=x,
            key_padding_mask=key_padding_mask,
            rope_cos=rope_cos,
            rope_sin=rope_sin,
        )
        x = residual + x

        # FFN（Pre-LN）
        residual = x
        x = self.norm2(x)
        x = self.ffn(x)
        x = residual + x

        return x, key_padding_mask


class LongerEncoder(nn.Module):
    """Top-K 压缩序列编码器。

    根据输入长度自适应行为:
    - L > top_k（第一个 MultiSeqHyFormerBlock）：Cross Attention。
      Q = 最近的 top_k 个 token，K/V = 全部 seq token -> 输出 (B, top_k, D)。
    - L <= top_k（后续 MultiSeqHyFormerBlock）：Self Attention。
      Q = K = V = top_k token -> 输出 (B, top_k, D)。

    因果 mask 只应用在 top_k token 之间（self-attention 层）；第一层 cross-attention 不使用因果 mask，因为 Q 和 K 长度不同。

    返回 (output, new_key_padding_mask)，以便下游更新 mask。
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        top_k: int = 50,
        hidden_mult: int = 4,
        dropout: float = 0.0,
        causal: bool = False,
    ) -> None:
        super().__init__()
        self.top_k = top_k
        self.causal = causal

        # 注意力前的 Pre-LN
        self.norm_q = nn.LayerNorm(d_model)
        self.norm_kv = nn.LayerNorm(d_model)

        # cross attention 与 self attention 共享 RoPEMHA
        self.attn = RoPEMultiheadAttention(
            d_model=d_model,
            num_heads=num_heads,
            dropout=dropout,
            rope_on_q=True,
        )

        # FFN（Pre-LN + 残差）
        self.ffn_norm = nn.LayerNorm(d_model)
        hidden_dim = d_model * hidden_mult
        self.ffn = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, d_model),
            nn.Dropout(dropout),
        )

    def _gather_top_k(
        self, x: torch.Tensor, key_padding_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """为每个样本选择最新的 top_k 个有效 token。

        参数:
            x: (B, L, D)。
            key_padding_mask: (B, L)，True 表示 padding。

        返回:
            top_k_tokens: (B, top_k, D)。
            new_padding_mask: (B, top_k)，True 表示 padding。
            position_indices: (B, top_k)，每个被选中 token 的原始位置索引，用于 Q 侧 RoPE。
        """
        B, L, D = x.shape
        device = x.device

        # 每个样本的有效长度
        valid_len = (~key_padding_mask).sum(dim=1)  # (B,)

        # 每个样本的起始位置：max(valid_len - top_k, 0)
        actual_k = torch.clamp(valid_len, max=self.top_k)  # (B,)
        start_pos = valid_len - actual_k  # (B,)

        # 构造 gather 索引：(B, top_k)
        offsets = (
            torch.arange(self.top_k, device=device).unsqueeze(0).expand(B, -1)
        )  # (B, top_k)
        indices = start_pos.unsqueeze(1) + offsets  # (B, top_k)

        # 对 valid_len < top_k 的样本，前部索引可能超过有效范围；
        # 将其 clamp 到 [0, L-1]，并在下面用 mask 处理
        indices = torch.clamp(indices, min=0, max=L - 1)

        # Gather 得到：(B, top_k, D)
        indices_expanded = indices.unsqueeze(-1).expand(
            -1, -1, D
        )  # (B, top_k, D)
        top_k_tokens = torch.gather(x, dim=1, index=indices_expanded)

        # 新填充 mask：前 (top_k - actual_k) 个位置为填充
        new_valid_len = actual_k  # (B,)
        pad_count = self.top_k - new_valid_len  # (B,)
        pos_indices = torch.arange(self.top_k, device=device).unsqueeze(
            0
        )  # (1, top_k)
        new_padding_mask = pos_indices < pad_count.unsqueeze(1)  # (B, top_k)

        # 将填充位置的 token 置零
        top_k_tokens = top_k_tokens * (~new_padding_mask).unsqueeze(-1).float()

        # Q 侧 RoPE 使用的 position_indices
        position_indices = indices  # (B, top_k)

        return top_k_tokens, new_padding_mask, position_indices

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        rope_cos: Optional[torch.Tensor] = None,
        rope_sin: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """应用带自适应 cross/self attention 的 LongerEncoder。

        参数:
            x: (B, L, D)，序列 token。
            key_padding_mask: (B, L)，True 表示 padding。
            rope_cos: (1, L, head_dim)，RoPE 余弦值（长度必须覆盖原始序列长度 L）。
            rope_sin: (1, L, head_dim)，RoPE 正弦值。

        返回:
            output: (B, top_k, D)，压缩后的序列。
            new_key_padding_mask: (B, top_k)，更新后的 padding mask。
        """
        B, L, D = x.shape

        if L > self.top_k:
            # === Cross Attention 模式（第一个 MultiSeqHyFormerBlock）===
            # 1. 提取最近的 top_k 个 token 作为 query
            q, new_mask, q_pos_indices = self._gather_top_k(x, key_padding_mask)

            # 2. Pre-LN
            q_normed = self.norm_q(q)
            kv_normed = self.norm_kv(x)

            # 3. 从全局 cos/sin 的 top_k 位置 gather，构造 Q 侧 RoPE cos/sin
            q_rope_cos = None
            q_rope_sin = None
            if rope_cos is not None and rope_sin is not None:
                # rope_cos: (1, L_max, head_dim)，q_pos_indices: (B, top_k)
                head_dim = rope_cos.shape[2]
                # 扩展到 batch 维度
                cos_expanded = rope_cos.expand(
                    B, -1, -1
                )  # (B, L_max, head_dim)
                sin_expanded = rope_sin.expand(B, -1, -1)
                idx = q_pos_indices.unsqueeze(-1).expand(
                    -1, -1, head_dim
                )  # (B, top_k, head_dim)
                q_rope_cos = torch.gather(
                    cos_expanded, 1, idx
                )  # (B, top_k, head_dim)
                q_rope_sin = torch.gather(sin_expanded, 1, idx)

            # 4. Cross Attention（Q 和 K 长度不同，因此不使用因果 mask）
            attn_out, _ = self.attn(
                query=q_normed,
                key=kv_normed,
                value=kv_normed,
                key_padding_mask=key_padding_mask,  # 原始 (B, L) mask
                rope_cos=rope_cos,
                rope_sin=rope_sin,
                q_rope_cos=q_rope_cos,
                q_rope_sin=q_rope_sin,
            )
            out = q + attn_out  # 基于 q 的残差
        else:
            # === Self Attention 模式（后续 MultiSeqHyFormerBlock）===
            new_mask = key_padding_mask

            # Pre-LN（Q 和 KV 共享 norm_q）
            x_normed = self.norm_q(x)

            # 因果 mask
            attn_mask = None
            if self.causal:
                attn_mask = nn.Transformer.generate_square_subsequent_mask(
                    L, device=x.device
                )

            attn_out, _ = self.attn(
                query=x_normed,
                key=x_normed,
                value=x_normed,
                key_padding_mask=key_padding_mask,
                attn_mask=attn_mask,
                rope_cos=rope_cos,
                rope_sin=rope_sin,
            )
            out = x + attn_out

        # FFN（Pre-LN + 残差）
        residual = out
        out = self.ffn_norm(out)
        out = self.ffn(out)
        out = residual + out

        return out, new_mask


def create_sequence_encoder(
    encoder_type: str,
    d_model: int,
    num_heads: int = 4,
    hidden_mult: int = 4,
    dropout: float = 0.0,
    top_k: int = 50,
    causal: bool = False,
) -> nn.Module:
    """创建指定类型的序列编码器。

    参数:
        encoder_type: 'swiglu'、'transformer' 或 'longer' 之一。
        d_model: 模型维度。
        num_heads: 注意力头数（transformer/longer 使用）。
        hidden_mult: FFN 扩展倍数。
        dropout: Dropout 比率。
        top_k: LongerEncoder 的压缩长度（仅 longer 使用）。
        causal: 是否在 LongerEncoder 中使用因果 mask（仅 longer 使用）。

    返回:
        序列编码器模块。
    """
    if encoder_type == "swiglu":
        return SwiGLUEncoder(d_model, hidden_mult, dropout)
    elif encoder_type == "transformer":
        return TransformerEncoder(d_model, num_heads, hidden_mult, dropout)
    elif encoder_type == "longer":
        return LongerEncoder(
            d_model, num_heads, top_k, hidden_mult, dropout, causal
        )
    else:
        raise ValueError(f"Unknown encoder type: {encoder_type}")


# ═══════════════════════════════════════════════════════════════════════════════
# HyFormer 块
# ═══════════════════════════════════════════════════════════════════════════════


class MultiSeqHyFormerBlock(nn.Module):
    """多序列 HyFormer 块。

    S 条序列各自独立执行 Sequence Evolution 和 Query Decoding，然后合并所有 Q token 与共享 NS token，做联合 Query Boosting。
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        num_queries: int,
        num_ns: int,
        num_sequences: int,
        seq_encoder_type: str = "swiglu",
        hidden_mult: int = 4,
        dropout: float = 0.0,
        top_k: int = 50,
        causal: bool = False,
        rank_mixer_mode: str = "full",
    ) -> None:
        super().__init__()
        self.num_sequences = num_sequences
        self.num_queries = num_queries
        self.num_ns = num_ns

        # 每条序列独立的序列编码器
        self.seq_encoders = nn.ModuleList(
            [
                create_sequence_encoder(
                    encoder_type=seq_encoder_type,
                    d_model=d_model,
                    num_heads=num_heads,
                    hidden_mult=hidden_mult,
                    dropout=dropout,
                    top_k=top_k,
                    causal=causal,
                )
                for _ in range(num_sequences)
            ]
        )

        # 每条序列独立的 cross-attention
        self.cross_attns = nn.ModuleList(
            [
                CrossAttention(
                    d_model=d_model,
                    num_heads=num_heads,
                    dropout=dropout,
                    ln_mode="pre",
                )
                for _ in range(num_sequences)
            ]
        )

        # RankMixer：输入 token 数 = Nq * S + Nns
        n_total = num_queries * num_sequences + num_ns
        self.mixer = RankMixerBlock(
            d_model=d_model,
            n_total=n_total,
            hidden_mult=hidden_mult,
            dropout=dropout,
            mode=rank_mixer_mode,
        )

    def forward(
        self,
        q_tokens_list: list,
        ns_tokens: torch.Tensor,
        seq_tokens_list: list,
        seq_padding_masks: list,
        rope_cos_list: Optional[List[torch.Tensor]] = None,
        rope_sin_list: Optional[List[torch.Tensor]] = None,
    ) -> Tuple[list, torch.Tensor, list, list]:
        """处理一步多序列 HyFormer 块。

        参数:
            q_tokens_list: 长度为 S 的 (B, Nq, D) 张量列表。
            ns_tokens: (B, Nns, D)。
            seq_tokens_list: 长度为 S 的 (B, L_i, D) 张量列表。
            seq_padding_masks: 长度为 S 的 (B, L_i) mask 列表。
            rope_cos_list: 长度为 S 的 (1, L_i, head_dim) 张量列表。
            rope_sin_list: 长度为 S 的 (1, L_i, head_dim) 张量列表。

        返回:
            元组 (next_q_list, next_ns, next_seq_list, next_masks)，其中 next_q_list 是更新后的 (B, Nq, D) query 张量列表，next_ns 是更新后的 (B, Nns, D) 非序列 token，next_seq_list 是编码后的 (B, L_i', D) 序列张量列表，next_masks 是更新后的 (B, L_i') padding mask 列表。
        """
        S = self.num_sequences
        Nq = self.num_queries

        # 1. 每条序列独立执行 Sequence Evolution
        next_seqs = []
        next_masks = []
        for i in range(S):
            rc = rope_cos_list[i] if rope_cos_list is not None else None
            rs = rope_sin_list[i] if rope_sin_list is not None else None
            result = self.seq_encoders[i](
                seq_tokens_list[i],
                seq_padding_masks[i],
                rope_cos=rc,
                rope_sin=rs,
            )
            next_seq_i, mask_i = result
            next_seqs.append(next_seq_i)
            next_masks.append(mask_i)

        # 2. 每条序列独立执行 Query Decoding
        decoded_qs = []
        for i in range(S):
            rc = rope_cos_list[i] if rope_cos_list is not None else None
            rs = rope_sin_list[i] if rope_sin_list is not None else None
            decoded_q_i = self.cross_attns[i](
                q_tokens_list[i],
                next_seqs[i],
                next_masks[i],
                rope_cos=rc,
                rope_sin=rs,
            )
            decoded_qs.append(decoded_q_i)

        # 3. Token Fusion：拼接所有 decoded_q + ns_tokens
        combined = torch.cat(
            decoded_qs + [ns_tokens], dim=1
        )  # (B, Nq*S + Nns, D)

        # 4. Query Boosting
        boosted = self.mixer(combined)  # (B, Nq*S + Nns, D)

        # 5. 拆回每条序列的 Q 和 NS
        next_q_list = []
        offset = 0
        for i in range(S):
            next_q_list.append(boosted[:, offset : offset + Nq, :])
            offset += Nq
        next_ns = boosted[:, offset:, :]

        return next_q_list, next_ns, next_seqs, next_masks


# ═══════════════════════════════════════════════════════════════════════════════
# PCVRHyFormer 主模型
# ═══════════════════════════════════════════════════════════════════════════════


class GroupNSTokenizer(nn.Module):
    """ns_tokenizer_type='group' 使用的 NS tokenizer。

    按 fid 对离散特征分组，对多值特征应用共享 Embedding 并做均值池化，然后把每组投影成一个 NS token（每组一个 token）。
    """

    def __init__(
        self,
        feature_specs: List[Tuple[int, int, int]],
        groups: List[List[int]],
        emb_dim: int,
        d_model: int,
        emb_skip_threshold: int = 0,
    ) -> None:
        super().__init__()
        self.feature_specs = feature_specs
        self.groups = groups
        self.emb_dim = emb_dim
        self.emb_skip_threshold = emb_skip_threshold

        # 每个 fid 一张 Embedding 表（若被 emb_skip_threshold 跳过
        # 或 vocab_size <= 0 / 无 vocab 信息，则为 None）。
        embs = []
        for vs, offset, length in feature_specs:
            skip = int(vs) <= 0 or (
                emb_skip_threshold > 0 and int(vs) > emb_skip_threshold
            )
            if skip:
                embs.append(None)
            else:
                embs.append(nn.Embedding(int(vs) + 1, emb_dim, padding_idx=0))
        self.embs = nn.ModuleList([e for e in embs if e is not None])
        # 将 fid 索引映射到 self.embs 中的位置（若被过滤则为 -1）
        self._emb_index = []
        real_idx = 0
        for e in embs:
            if e is not None:
                self._emb_index.append(real_idx)
                real_idx += 1
            else:
                self._emb_index.append(-1)

        # 每组投影：num_fids_in_group * emb_dim -> d_model（带 LayerNorm）
        self.group_projs = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(len(group) * emb_dim, d_model),
                    nn.LayerNorm(d_model),
                )
                for group in groups
            ]
        )

    def forward(self, int_feats: torch.Tensor) -> torch.Tensor:
        """将分组后的离散特征做 Embedding 并投影为 NS token。

        参数:
            int_feats: (B, total_int_dim)，拼接后的整数特征。

        返回:
            形状为 (B, num_groups, D) 的 token。
        """
        tokens = []
        for group, proj in zip(self.groups, self.group_projs):
            fid_embs = []
            for fid_idx in group:
                vs, offset, length = self.feature_specs[fid_idx]
                emb_real_idx = self._emb_index[fid_idx]
                if emb_real_idx == -1:
                    # 被过滤的高基数特征：输出零向量
                    fid_emb = int_feats.new_zeros(
                        int_feats.shape[0], self.emb_dim
                    )
                else:
                    emb_layer = self.embs[emb_real_idx]
                    if length == 1:
                        # 单值特征：直接查表
                        fid_emb = emb_layer(
                            int_feats[:, offset].long()
                        )  # (B, emb_dim)
                    else:
                        # 多值特征：查表后做均值池化（忽略填充值 0）
                        vals = int_feats[
                            :, offset : offset + length
                        ].long()  # (B, length)
                        emb_all = emb_layer(vals)  # (B, length, emb_dim)
                        mask = (
                            (vals != 0).float().unsqueeze(-1)
                        )  # (B, length, 1)
                        count = mask.sum(dim=1).clamp(min=1)  # (B, 1)
                        fid_emb = (emb_all * mask).sum(
                            dim=1
                        ) / count  # (B, emb_dim)
                fid_embs.append(fid_emb)
            cat_emb = torch.cat(fid_embs, dim=-1)  # (B, num_fids*emb_dim)
            tokens.append(F.silu(proj(cat_emb)).unsqueeze(1))  # (B, 1, D)
        return torch.cat(tokens, dim=1)  # (B, num_groups, D)


class RankMixerNSTokenizer(nn.Module):
    """遵循 RankMixer 论文方法的 NS Tokenizer。

    将所有组的 Embedding 向量拼接成一个长向量，然后均分为 num_ns_tokens 个片段，并将每个片段投影到 d_model。这样可以自由选择 num_ns_tokens（不依赖组数量）。
    """

    def __init__(
        self,
        feature_specs: List[Tuple[int, int, int]],
        groups: List[List[int]],
        emb_dim: int,
        d_model: int,
        num_ns_tokens: int,
        emb_skip_threshold: int = 0,
    ) -> None:
        """初始化 RankMixerNSTokenizer。

        参数:
            feature_specs: 每个特征的 [(vocab_size, offset, length), ...]。
            groups: 特征索引分组列表（定义语义顺序）。
            emb_dim: 每个特征的 Embedding 维度。
            d_model: 输出 token 维度。
            num_ns_tokens: 需要生成的 NS token 数量（T 个片段）。
            emb_skip_threshold: 跳过 vocab 大于阈值的特征 Embedding。
        """
        super().__init__()
        self.feature_specs = feature_specs
        self.groups = groups
        self.emb_dim = emb_dim
        self.num_ns_tokens = num_ns_tokens
        self.emb_skip_threshold = emb_skip_threshold

        # 每个 fid 一张 Embedding 表（若被 emb_skip_threshold 跳过
        # 或 vocab_size <= 0 / 无 vocab 信息，则为 None）。
        embs = []
        for vs, offset, length in feature_specs:
            skip = int(vs) <= 0 or (
                emb_skip_threshold > 0 and int(vs) > emb_skip_threshold
            )
            if skip:
                embs.append(None)
            else:
                embs.append(nn.Embedding(int(vs) + 1, emb_dim, padding_idx=0))
        self.embs = nn.ModuleList([e for e in embs if e is not None])
        # 将 fid 索引映射到 self.embs 中的位置（若被过滤则为 -1）
        self._emb_index = []
        real_idx = 0
        for e in embs:
            if e is not None:
                self._emb_index.append(real_idx)
                real_idx += 1
            else:
                self._emb_index.append(-1)

        # 计算总 embedding 维度：所有组中全部 fid 的维度之和
        total_num_fids = sum(len(g) for g in groups)
        total_emb_dim = total_num_fids * emb_dim

        # 对 total_emb_dim 做填充，使其能被 num_ns_tokens 整除
        self.chunk_dim = math.ceil(total_emb_dim / num_ns_tokens)
        self.padded_total_dim = self.chunk_dim * num_ns_tokens
        self._pad_size = self.padded_total_dim - total_emb_dim

        # 每个 chunk 的投影：chunk_dim -> d_model，并带 LayerNorm
        self.token_projs = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(self.chunk_dim, d_model), nn.LayerNorm(d_model)
                )
                for _ in range(num_ns_tokens)
            ]
        )

        logging.info(
            f"RankMixerNSTokenizer: {total_num_fids} fids, "
            f"total_emb_dim={total_emb_dim}, chunk_dim={self.chunk_dim}, "
            f"num_ns_tokens={num_ns_tokens}, pad={self._pad_size}"
        )

    def forward(self, int_feats: torch.Tensor) -> torch.Tensor:
        """对所有特征做 Embedding、拼接、切分并投影。

        参数:
            int_feats: (B, total_int_dim)，拼接后的整数特征。

        返回:
            (B, num_ns_tokens, d_model) 张量。
        """
        # 1. 按组顺序对所有 fid 做 Embedding，然后扁平拼接
        all_embs = []
        for group in self.groups:
            for fid_idx in group:
                vs, offset, length = self.feature_specs[fid_idx]
                emb_real_idx = self._emb_index[fid_idx]
                if emb_real_idx == -1:
                    fid_emb = int_feats.new_zeros(
                        int_feats.shape[0], self.emb_dim
                    )
                else:
                    emb_layer = self.embs[emb_real_idx]
                    if length == 1:
                        fid_emb = emb_layer(int_feats[:, offset].long())
                    else:
                        vals = int_feats[:, offset : offset + length].long()
                        emb_all = emb_layer(vals)
                        mask = (vals != 0).float().unsqueeze(-1)
                        count = mask.sum(dim=1).clamp(min=1)
                        fid_emb = (emb_all * mask).sum(dim=1) / count
                all_embs.append(fid_emb)

        cat_emb = torch.cat(all_embs, dim=-1)  # (B, total_emb_dim)

        # 2. 必要时填充
        if self._pad_size > 0:
            cat_emb = F.pad(
                cat_emb, (0, self._pad_size)
            )  # (B, padded_total_dim)

        # 3. 切分为 num_ns_tokens 个 chunk，并分别投影
        chunks = cat_emb.split(self.chunk_dim, dim=-1)  # (B, chunk_dim) 列表
        tokens = []
        for chunk, proj in zip(chunks, self.token_projs):
            tokens.append(F.silu(proj(chunk)).unsqueeze(1))  # (B, 1, d_model)

        return torch.cat(tokens, dim=1)  # (B, num_ns_tokens, d_model)


class PCVRHyFormer(nn.Module):
    """用于点击后转化率预测的 PCVRHyFormer 模型。

    结合 MultiSeqHyFormerBlock 与 MultiSeqQueryGenerator，处理多条输入序列和非序列特征。
    """

    def __init__(
        self,
        # 数据 schema
        user_int_feature_specs: List[Tuple[int, int, int]],
        item_int_feature_specs: List[Tuple[int, int, int]],
        user_dense_dim: int,
        item_dense_dim: int,
        seq_vocab_sizes: "dict[str, List[int]]",  # {domain: [每个 fid 的 vocab_size, ...]}
        # NS 分组配置（按 fid 索引分组）
        user_ns_groups: List[List[int]],
        item_ns_groups: List[List[int]],
        # 模型超参数
        d_model: int = 64,
        emb_dim: int = 64,
        num_queries: int = 1,
        num_hyformer_blocks: int = 2,
        num_heads: int = 4,
        seq_encoder_type: str = "transformer",
        hidden_mult: int = 4,
        dropout_rate: float = 0.01,
        seq_top_k: int = 50,
        seq_causal: bool = False,
        action_num: int = 1,
        num_time_buckets: int = 65,
        rank_mixer_mode: str = "full",
        use_rope: bool = False,
        rope_base: float = 10000.0,
        emb_skip_threshold: int = 0,
        seq_id_threshold: int = 10000,
        # NS tokenizer 变体
        ns_tokenizer_type: str = "rankmixer",
        user_ns_tokens: int = 0,
        item_ns_tokens: int = 0,
    ) -> None:
        super().__init__()

        self.d_model = d_model
        self.emb_dim = emb_dim
        self.action_num = action_num
        self.num_queries = num_queries
        self.seq_domains = sorted(seq_vocab_sizes.keys())  # 确定性顺序
        self.num_sequences = len(self.seq_domains)
        self.num_time_buckets = num_time_buckets
        self.rank_mixer_mode = rank_mixer_mode
        self.use_rope = use_rope
        self.emb_skip_threshold = emb_skip_threshold
        self.seq_id_threshold = seq_id_threshold
        self.ns_tokenizer_type = ns_tokenizer_type

        # ================== NS token 构造 ==================

        if ns_tokenizer_type == "group":
            # 原始方式：每组一个 NS token
            self.user_ns_tokenizer = GroupNSTokenizer(
                feature_specs=user_int_feature_specs,
                groups=user_ns_groups,
                emb_dim=emb_dim,
                d_model=d_model,
                emb_skip_threshold=emb_skip_threshold,
            )
            num_user_ns = len(user_ns_groups)

            self.item_ns_tokenizer = GroupNSTokenizer(
                feature_specs=item_int_feature_specs,
                groups=item_ns_groups,
                emb_dim=emb_dim,
                d_model=d_model,
                emb_skip_threshold=emb_skip_threshold,
            )
            num_item_ns = len(item_ns_groups)
        elif ns_tokenizer_type == "rankmixer":
            # RankMixer 论文风格：全部 embedding 拼接 -> 切分 -> 投影
            # 0 表示自动：回退为组数量
            if user_ns_tokens <= 0:
                user_ns_tokens = len(user_ns_groups)
            if item_ns_tokens <= 0:
                item_ns_tokens = len(item_ns_groups)
            self.user_ns_tokenizer = RankMixerNSTokenizer(
                feature_specs=user_int_feature_specs,
                groups=user_ns_groups,
                emb_dim=emb_dim,
                d_model=d_model,
                num_ns_tokens=user_ns_tokens,
                emb_skip_threshold=emb_skip_threshold,
            )
            num_user_ns = user_ns_tokens

            self.item_ns_tokenizer = RankMixerNSTokenizer(
                feature_specs=item_int_feature_specs,
                groups=item_ns_groups,
                emb_dim=emb_dim,
                d_model=d_model,
                num_ns_tokens=item_ns_tokens,
                emb_skip_threshold=emb_skip_threshold,
            )
            num_item_ns = item_ns_tokens
        else:
            raise ValueError(f"Unknown ns_tokenizer_type: {ns_tokenizer_type}")

        # 用户 dense 特征投影（若存在）
        self.has_user_dense = user_dense_dim > 0
        if self.has_user_dense:
            self.user_dense_proj = nn.Sequential(
                nn.Linear(user_dense_dim, d_model), nn.LayerNorm(d_model)
            )

        # 物品 dense 特征投影（若存在）
        self.has_item_dense = item_dense_dim > 0
        if self.has_item_dense:
            self.item_dense_proj = nn.Sequential(
                nn.Linear(item_dense_dim, d_model), nn.LayerNorm(d_model)
            )

        # NS token 总数
        self.num_ns = (
            num_user_ns
            + (1 if self.has_user_dense else 0)
            + num_item_ns
            + (1 if self.has_item_dense else 0)
        )

        # ================== 检查 d_model % T == 0 约束（仅 full 模式） ==================
        T = num_queries * self.num_sequences + self.num_ns
        if rank_mixer_mode == "full" and d_model % T != 0:
            valid_T_values = [
                t for t in range(1, d_model + 1) if d_model % t == 0
            ]
            raise ValueError(
                f"d_model={d_model} must be divisible by T=num_queries*num_sequences+num_ns="
                f"{num_queries}*{self.num_sequences}+{self.num_ns}={T}. "
                f"Valid T values for d_model={d_model}: {valid_T_values}"
            )

        # ================== 序列 token Embedding ==================
        # seq_id_threshold 决定序列 tokenizer 内哪些特征
        # 被视为 id 特征（会获得额外 dropout）。它完全
        # 独立于 emb_skip_threshold（后者用于跳过 Embedding 创建）。
        self.seq_id_emb_dropout = nn.Dropout(dropout_rate * 2)

        def _make_seq_embs(vocab_sizes):
            """创建 Embedding 列表；对因 emb_skip_threshold 或无 vocab 信息（vs<=0）而被跳过的特征返回 None。"""
            embs_raw = []
            for vs in vocab_sizes:
                skip = int(vs) <= 0 or (
                    emb_skip_threshold > 0 and int(vs) > emb_skip_threshold
                )
                if skip:
                    embs_raw.append(None)
                else:
                    embs_raw.append(
                        nn.Embedding(int(vs) + 1, emb_dim, padding_idx=0)
                    )
            module_list = nn.ModuleList([e for e in embs_raw if e is not None])
            # 将位置索引映射到 module_list 中的真实索引（若跳过则为 -1）
            index_map = []
            real_idx = 0
            for e in embs_raw:
                if e is not None:
                    index_map.append(real_idx)
                    real_idx += 1
                else:
                    index_map.append(-1)
            is_id = [int(vs) > seq_id_threshold for vs in vocab_sizes]
            return module_list, index_map, is_id

        # ================== 动态序列 Embedding ==================
        self._seq_embs = nn.ModuleDict()
        self._seq_emb_index = {}  # domain -> index_map
        self._seq_is_id = {}  # domain -> is_id 列表
        self._seq_vocab_sizes = {}  # domain -> vocab_sizes 列表
        self._seq_proj = nn.ModuleDict()

        for domain in self.seq_domains:
            vs = seq_vocab_sizes[domain]
            embs, idx_map, is_id = _make_seq_embs(vs)
            self._seq_embs[domain] = embs
            self._seq_emb_index[domain] = idx_map
            self._seq_is_id[domain] = is_id
            self._seq_vocab_sizes[domain] = vs
            self._seq_proj[domain] = nn.Sequential(
                nn.Linear(len(vs) * emb_dim, d_model), nn.LayerNorm(d_model)
            )

        # ================== 时间间隔桶 Embedding（可选） ==================
        if num_time_buckets > 0:
            self.time_embedding = nn.Embedding(
                num_time_buckets, d_model, padding_idx=0
            )

        # ================== HyFormer 组件 ==================
        # MultiSeqQueryGenerator 查询生成器
        self.query_generator = MultiSeqQueryGenerator(
            d_model=d_model,
            num_ns=self.num_ns,
            num_queries=num_queries,
            num_sequences=self.num_sequences,
            hidden_mult=hidden_mult,
        )

        # MultiSeqHyFormerBlock 堆叠
        self.blocks = nn.ModuleList(
            [
                MultiSeqHyFormerBlock(
                    d_model=d_model,
                    num_heads=num_heads,
                    num_queries=num_queries,
                    num_ns=self.num_ns,
                    num_sequences=self.num_sequences,
                    seq_encoder_type=seq_encoder_type,
                    hidden_mult=hidden_mult,
                    dropout=dropout_rate,
                    top_k=seq_top_k,
                    causal=seq_causal,
                    rank_mixer_mode=rank_mixer_mode,
                )
                for _ in range(num_hyformer_blocks)
            ]
        )

        # ================== RoPE ==================
        if use_rope:
            head_dim = d_model // num_heads
            self.rotary_emb = RotaryEmbedding(dim=head_dim, base=rope_base)
        else:
            self.rotary_emb = None

        # 输出投影
        self.output_proj = nn.Sequential(
            nn.Linear(num_queries * self.num_sequences * d_model, d_model),
            nn.LayerNorm(d_model),
        )

        # Dropout
        self.emb_dropout = nn.Dropout(dropout_rate)

        # 分类器
        self.clsfier = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.SiLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(d_model, action_num),
        )

        # 初始化参数
        self._init_params()

        # 记录 emb_skip_threshold 过滤统计
        if emb_skip_threshold > 0:

            def _count_filtered(vocab_sizes, emb_index):
                filtered = sum(1 for idx in emb_index if idx == -1)
                return filtered, len(vocab_sizes)

            for domain in self.seq_domains:
                f, t = _count_filtered(
                    self._seq_vocab_sizes[domain], self._seq_emb_index[domain]
                )
                if f > 0:
                    logging.info(
                        f"emb_skip_threshold={emb_skip_threshold}: {domain} skipped {f}/{t} features"
                    )
            for name, tokenizer in [
                ("user_ns", self.user_ns_tokenizer),
                ("item_ns", self.item_ns_tokenizer),
            ]:
                f = sum(1 for idx in tokenizer._emb_index if idx == -1)
                t = len(tokenizer._emb_index)
                if f > 0:
                    logging.info(
                        f"emb_skip_threshold={emb_skip_threshold}: {name} skipped {f}/{t} features"
                    )

    def _init_params(self) -> None:
        """对所有 Embedding 权重应用 Xavier 初始化。"""
        for domain in self.seq_domains:
            for emb in self._seq_embs[domain]:
                nn.init.xavier_normal_(emb.weight.data)
                emb.weight.data[0, :] = 0

        for tokenizer in [self.user_ns_tokenizer, self.item_ns_tokenizer]:
            for emb in tokenizer.embs:
                nn.init.xavier_normal_(emb.weight.data)
                emb.weight.data[0, :] = 0

        if self.num_time_buckets > 0:
            nn.init.xavier_normal_(self.time_embedding.weight.data)
            self.time_embedding.weight.data[0, :] = 0

    def reinit_high_cardinality_params(
        self, cardinality_threshold: int = 10000
    ) -> "set[int]":
        """只重新初始化高基数 Embedding。

        保留低基数 Embedding 和时间特征 Embedding。

        参数:
            cardinality_threshold: 只重新初始化 vocab_size 超过该值的 Embedding。

        返回:
            被重新初始化参数的 data_ptr() 值集合。
        """
        reinit_count = 0
        skip_count = 0
        reinit_ptrs = set()

        for emb_list, vocab_sizes, emb_index in [
            (
                self._seq_embs[d],
                self._seq_vocab_sizes[d],
                self._seq_emb_index[d],
            )
            for d in self.seq_domains
        ]:
            for i, vs in enumerate(vocab_sizes):
                real_idx = emb_index[i]
                if real_idx == -1:
                    # 已被 emb_skip_threshold 跳过，没有 Embedding 需要重新初始化
                    continue
                emb = emb_list[real_idx]
                if int(vs) > cardinality_threshold:
                    nn.init.xavier_normal_(emb.weight.data)
                    emb.weight.data[0, :] = 0
                    reinit_ptrs.add(emb.weight.data_ptr())
                    reinit_count += 1
                else:
                    skip_count += 1

        for tokenizer, specs in [
            (self.user_ns_tokenizer, self.user_ns_tokenizer.feature_specs),
            (self.item_ns_tokenizer, self.item_ns_tokenizer.feature_specs),
        ]:
            for i, (vs, offset, length) in enumerate(specs):
                real_idx = tokenizer._emb_index[i]
                if real_idx == -1:
                    continue
                emb = tokenizer.embs[real_idx]
                if int(vs) > cardinality_threshold:
                    nn.init.xavier_normal_(emb.weight.data)
                    emb.weight.data[0, :] = 0
                    reinit_ptrs.add(emb.weight.data_ptr())
                    reinit_count += 1
                else:
                    skip_count += 1

        # 始终保留 time_embedding
        if self.num_time_buckets > 0:
            skip_count += 1

        logging.info(
            f"Re-initialized {reinit_count} high-cardinality Embeddings "
            f"(vocab>{cardinality_threshold}), kept {skip_count}"
        )
        return reinit_ptrs

    def get_sparse_params(self) -> List[nn.Parameter]:
        """返回所有 Embedding 表参数（使用 Adagrad 优化）。"""
        sparse_params = set()
        for module in self.modules():
            if isinstance(module, nn.Embedding):
                sparse_params.add(module.weight.data_ptr())
        return [p for p in self.parameters() if p.data_ptr() in sparse_params]

    def get_dense_params(self) -> List[nn.Parameter]:
        """返回所有非 Embedding 参数（使用 AdamW 优化）。"""
        sparse_ptrs = {p.data_ptr() for p in self.get_sparse_params()}
        return [p for p in self.parameters() if p.data_ptr() not in sparse_ptrs]

    def _embed_seq_domain(
        self,
        seq: torch.Tensor,
        sideinfo_embs: nn.ModuleList,
        proj: nn.Module,
        is_id: List[bool],
        emb_index: List[int],
        time_bucket_ids: torch.Tensor,
    ) -> torch.Tensor:
        """拼接 sideinfo Embedding 并投影到 d_model，从而编码一个序列域。"""
        B, S, L = seq.shape
        emb_list = []
        for i in range(S):
            real_idx = emb_index[i] if i < len(emb_index) else -1
            if real_idx == -1:
                # 被 emb_skip_threshold 跳过的特征：输出零向量
                emb_list.append(
                    seq.new_zeros(B, L, self.emb_dim, dtype=torch.float)
                )
            else:
                emb = sideinfo_embs[real_idx]
                e = emb(seq[:, i, :])  # (B, L, emb_dim)
                if is_id[i] and self.training:
                    e = self.seq_id_emb_dropout(e)
                emb_list.append(e)
        cat_emb = torch.cat(emb_list, dim=-1)  # (B, L, S*emb_dim)
        token_emb = F.gelu(proj(cat_emb))  # (B, L, D)

        # 加入时间桶 Embedding（全 0 id 会通过 padding_idx=0 产生零向量）
        if self.num_time_buckets > 0:
            token_emb = token_emb + self.time_embedding(time_bucket_ids)

        return token_emb

    def _make_padding_mask(
        self, seq_len: torch.Tensor, max_len: int
    ) -> torch.Tensor:
        """根据序列长度生成 padding mask。"""
        device = seq_len.device
        idx = torch.arange(max_len, device=device).unsqueeze(0)  # (1, max_len)
        return idx >= seq_len.unsqueeze(1)  # (B, max_len)

    def _run_multi_seq_blocks(
        self,
        q_tokens_list: list,
        ns_tokens: torch.Tensor,
        seq_tokens_list: list,
        seq_masks_list: list,
        apply_dropout: bool = True,
    ) -> torch.Tensor:
        """运行带 dropout 和输出投影的多序列 block 堆叠。"""
        if apply_dropout:
            q_tokens_list = [self.emb_dropout(q) for q in q_tokens_list]
            ns_tokens = self.emb_dropout(ns_tokens)
            seq_tokens_list = [self.emb_dropout(s) for s in seq_tokens_list]

        curr_qs = q_tokens_list
        curr_ns = ns_tokens
        curr_seqs = seq_tokens_list
        curr_masks = seq_masks_list

        for block in self.blocks:
            # 为每条序列预计算 RoPE cos/sin
            rope_cos_list = None
            rope_sin_list = None
            if self.rotary_emb is not None:
                rope_cos_list = []
                rope_sin_list = []
                device = curr_seqs[0].device
                for seq_i in curr_seqs:
                    seq_len = seq_i.shape[1]
                    cos, sin = self.rotary_emb(seq_len, device)
                    rope_cos_list.append(cos)
                    rope_sin_list.append(sin)

            curr_qs, curr_ns, curr_seqs, curr_masks = block(
                q_tokens_list=curr_qs,
                ns_tokens=curr_ns,
                seq_tokens_list=curr_seqs,
                seq_padding_masks=curr_masks,
                rope_cos_list=rope_cos_list,
                rope_sin_list=rope_sin_list,
            )

        # 输出：拼接所有序列的 Q token，然后通过 MLP 投影
        B = curr_qs[0].shape[0]
        all_q = torch.cat(curr_qs, dim=1)  # (B, Nq*S, D)
        output = all_q.view(B, -1)  # (B, Nq*S*D)
        output = self.output_proj(output)  # (B, D)

        return output

    def forward(self, inputs: ModelInput) -> torch.Tensor:
        """执行 PCVRHyFormer 模型的 forward 过程。"""
        # 1. NS token：分组投影
        user_ns = self.user_ns_tokenizer(
            inputs.user_int_feats
        )  # (B, num_user_groups, D)
        item_ns = self.item_ns_tokenizer(
            inputs.item_int_feats
        )  # (B, num_item_groups, D)

        ns_parts = [user_ns]
        if self.has_user_dense:
            user_dense_tok = F.silu(
                self.user_dense_proj(inputs.user_dense_feats)
            ).unsqueeze(1)  # (B, 1, D)
            ns_parts.append(user_dense_tok)
        ns_parts.append(item_ns)
        if self.has_item_dense:
            item_dense_tok = F.silu(
                self.item_dense_proj(inputs.item_dense_feats)
            ).unsqueeze(1)  # (B, 1, D)
            ns_parts.append(item_dense_tok)

        ns_tokens = torch.cat(ns_parts, dim=1)  # (B, num_ns, D)

        # 2. 对每个序列 domain 做 Embedding（动态）
        seq_tokens_list = []
        seq_masks_list = []
        for domain in self.seq_domains:
            tokens = self._embed_seq_domain(
                inputs.seq_data[domain],
                self._seq_embs[domain],
                self._seq_proj[domain],
                self._seq_is_id[domain],
                self._seq_emb_index[domain],
                inputs.seq_time_buckets[domain],
            )
            seq_tokens_list.append(tokens)
            mask = self._make_padding_mask(
                inputs.seq_lens[domain], inputs.seq_data[domain].shape[2]
            )
            seq_masks_list.append(mask)

        # 3. 通过 MultiSeqQueryGenerator 为每条序列生成独立 Q token
        q_tokens_list = self.query_generator(
            ns_tokens, seq_tokens_list, seq_masks_list
        )

        # 4. Dropout + MultiSeqHyFormerBlock 堆叠 + 输出投影
        output = self._run_multi_seq_blocks(
            q_tokens_list,
            ns_tokens,
            seq_tokens_list,
            seq_masks_list,
            apply_dropout=self.training,
        )

        # 5. 分类器
        logits = self.clsfier(output)  # (B, action_num)
        return logits

    def predict(self, inputs: ModelInput) -> Tuple[torch.Tensor, torch.Tensor]:
        """在不使用 dropout 的情况下执行推理，并返回 logits 与 embedding。"""
        # 复用 forward 逻辑，但不使用 dropout
        user_ns = self.user_ns_tokenizer(inputs.user_int_feats)
        item_ns = self.item_ns_tokenizer(inputs.item_int_feats)

        ns_parts = [user_ns]
        if self.has_user_dense:
            user_dense_tok = F.silu(
                self.user_dense_proj(inputs.user_dense_feats)
            ).unsqueeze(1)
            ns_parts.append(user_dense_tok)
        ns_parts.append(item_ns)
        if self.has_item_dense:
            item_dense_tok = F.silu(
                self.item_dense_proj(inputs.item_dense_feats)
            ).unsqueeze(1)
            ns_parts.append(item_dense_tok)

        ns_tokens = torch.cat(ns_parts, dim=1)

        seq_tokens_list = []
        seq_masks_list = []
        for domain in self.seq_domains:
            tokens = self._embed_seq_domain(
                inputs.seq_data[domain],
                self._seq_embs[domain],
                self._seq_proj[domain],
                self._seq_is_id[domain],
                self._seq_emb_index[domain],
                inputs.seq_time_buckets[domain],
            )
            seq_tokens_list.append(tokens)
            mask = self._make_padding_mask(
                inputs.seq_lens[domain], inputs.seq_data[domain].shape[2]
            )
            seq_masks_list.append(mask)

        q_tokens_list = self.query_generator(
            ns_tokens, seq_tokens_list, seq_masks_list
        )

        output = self._run_multi_seq_blocks(
            q_tokens_list,
            ns_tokens,
            seq_tokens_list,
            seq_masks_list,
            apply_dropout=False,
        )

        logits = self.clsfier(output)
        return logits, output

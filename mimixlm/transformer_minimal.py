"""
Minimal Transformer Implementation
- Pure PyTorch, no dependency on transformers or similar libraries
- Includes: ModelConfig, model definition, training, sampling-based decoding
- Supports:
    pos_encoding : sinusoidal | rope
    ffn_type     : standard | glu
    activation   : relu | gelu | silu | swiglu
    norm_type    : pre | post
    model_type   : encoder_decoder | lm (decoder-only)
    tie_weights  : share decoder embedding and output projection weights
    kv_cache     : incremental KV cache for fast autoregressive decoding
    decoding     : greedy / top-k / top-p / temperature sampling
"""

import dataclasses
import json
import os
import math
from dataclasses import dataclass
from typing import Optional, Literal, Iterable, Generator

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ModelConfig:
    """Configuration for the Transformer model."""
    # Vocabulary & special tokens
    vocab_size: int = 32
    pad_idx: int = 0
    bos_idx: int = 1
    eos_idx: int = 2

    # Model architecture
    model_type: Literal["encoder_decoder", "lm"] = "encoder_decoder"
    d_model: int = 128
    num_heads: int = 4
    num_encoder_layers: int = 2
    num_decoder_layers: int = 2
    d_ff: int = 256
    max_len: int = 64
    dropout: float = 0.1

    # Positional encoding
    pos_encoding: Literal["sinusoidal", "rope"] = "rope"
    # Base frequency for RoPE; larger values extend effective context length (e.g. 500000 for LLaMA-3)
    rope_base: int = 10000

    # FFN type and activation function
    # When ffn_type="glu", activation can be "swiglu" / "gelu" / "silu"
    ffn_type: Literal["standard", "glu"] = "glu"
    activation: Literal["relu", "relu2", "gelu", "silu", "swiglu"] = "silu"

    # Number of KV heads for Grouped Query Attention (GQA) / Multi Query Attention (MQA).
    # 0 means equal to num_heads (standard MHA).
    # 1 means MQA (all query heads share a single KV head).
    # 1 < num_kv_heads < num_heads means GQA.
    num_kv_heads: int = 0

    # Normalization layer type used throughout the model
    norm_layer: Literal["layernorm", "rmsnorm"] = "rmsnorm"
    norm_type: Literal["pre", "post"] = "pre"
    # Epsilon for all normalization layers (LayerNorm / RMSNorm)
    norm_eps: float = 1e-6

    # Embedding output scale factor applied after the lookup.
    # None  : no scaling
    # float : multiply the embedding by this value (e.g. math.sqrt(d_model) for the
    #         original Transformer paper convention)
    embedding_scale: Optional[float] = None

    # Whether to apply a normalization layer immediately after the token embedding
    # (and after positional encoding).  Useful for stabilising deep models.
    embedding_norm: bool = False

    # Whether to apply per-head normalization to Q/K and V after projection
    qk_norm: bool = False
    v_norm: bool = False

    # Whether to use learnable scalar weights on each residual branch
    # Each sublayer gets an alpha parameter: output = x + alpha * sublayer(x)
    learnable_residual: bool = False

    # Whether to use tied weights for the decoder embedding and output projection
    tie_weights: bool = False

    # Whether this model is a chat/instruction-tuned model.
    # When True, chat() requires a ChatFormatter instance to format each turn
    # into the model's expected prompt template before encoding.
    is_chat: bool = False

    # Floating-point precision for model weights and inference.
    # "float32" : full precision (default, always safe)
    # "float16" : half precision (GPU only; saves memory, may need loss scaling)
    # "bfloat16": brain float (recommended for modern GPUs / Apple MPS)
    dtype: Literal["float32", "float16", "bfloat16"] = "float32"




# ─────────────────────────────────────────────
# 1. Normalization Layers
# ─────────────────────────────────────────────

class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization (RMSNorm).
    Simpler than LayerNorm: no mean subtraction, only RMS scaling.
    output = x / rms(x) * weight,  where rms(x) = sqrt(mean(x^2) + eps)
    """

    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Upcast to fp32 for numerically stable variance computation,
        # then cast the result back to the original dtype.
        x_fp32 = x.float()
        rms = x_fp32.pow(2).mean(dim=-1, keepdim=True).add(self.eps).sqrt()
        return (x_fp32 / rms).to(x.dtype) * self.weight


def build_norm(d_model: int, cfg: ModelConfig) -> nn.Module:
    """Return a normalization layer instance based on cfg.norm_layer and cfg.norm_eps."""
    if cfg.norm_layer == "rmsnorm":
        return RMSNorm(d_model, eps=cfg.norm_eps)
    return nn.LayerNorm(d_model, eps=cfg.norm_eps)


# ─────────────────────────────────────────────
# 2. Positional Encoding
# ─────────────────────────────────────────────

class SinusoidalPositionalEncoding(nn.Module):
    """Classic sinusoidal positional encoding added directly to embeddings."""

    def __init__(self, d_model: int, max_len: int, dropout: float):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # [1, max_len, d_model]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class RotaryEmbedding(nn.Module):
    """
    Rotary Position Embedding (RoPE).
    Applied to Q and K inside attention, not added to the token embeddings.
    The base frequency is configurable (cfg.rope_base).
    """

    def __init__(self, d_head: int, max_len: int, base: int = 10000):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, d_head, 2).float() / d_head))
        self.register_buffer("inv_freq", inv_freq)

        positions = torch.arange(max_len).float()
        freqs = torch.outer(positions, inv_freq)   # [max_len, d_head/2]
        emb = torch.cat([freqs, freqs], dim=-1)    # [max_len, d_head]
        self.register_buffer("cos_cache", emb.cos())
        self.register_buffer("sin_cache", emb.sin())

    @staticmethod
    def _rotate_half(x: torch.Tensor) -> torch.Tensor:
        """Negate the second half and concatenate with the first half to rotate."""
        half = x.shape[-1] // 2
        return torch.cat([-x[..., half:], x[..., :half]], dim=-1)

    def apply(self, x: torch.Tensor, offset: int = 0) -> torch.Tensor:
        """
        Apply RoPE to x: [batch, heads, seq, d_head].
        offset: starting position index (used with KV cache).
        """
        seq_len = x.size(2)
        cos = self.cos_cache[offset: offset + seq_len].unsqueeze(0).unsqueeze(0)
        sin = self.sin_cache[offset: offset + seq_len].unsqueeze(0).unsqueeze(0)
        return x * cos + self._rotate_half(x) * sin


# ─────────────────────────────────────────────
# 2. Activation Factory
# ─────────────────────────────────────────────

class _ReLU2(nn.Module):
    """Squared ReLU: f(x) = relu(x)^2. Faster convergence than plain ReLU."""
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(x).pow(2)


def build_activation(name: str) -> nn.Module:
    mapping = {
        "relu":   nn.ReLU(),
        "relu2":  _ReLU2(),
        "gelu":   nn.GELU(),
        "silu":   nn.SiLU(),
        "swiglu": nn.SiLU(),  # SwiGLU uses SiLU as the gate activation
    }
    if name not in mapping:
        raise ValueError(f"Unsupported activation: {name}. Choose from {list(mapping)}")
    return mapping[name]


# ─────────────────────────────────────────────
# 3. Feed-Forward Networks (Standard & GLU)
# ─────────────────────────────────────────────

class StandardFeedForward(nn.Module):
    """Standard two-layer FFN: Linear -> Activation -> Dropout -> Linear."""

    def __init__(self, d_model: int, d_ff: int, activation: str, dropout: float):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff, bias=False)
        self.w2 = nn.Linear(d_ff, d_model, bias=False)
        self.act = build_activation(activation)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(self.dropout(self.act(self.w1(x))))


class GluFeedForward(nn.Module):
    """
    GLU-family FFN (including SwiGLU used in LLaMA):
      gate   = activation(W_gate * x)
      value  = W_val * x
      output = W2 * (gate * value)
    """

    def __init__(self, d_model: int, d_ff: int, activation: str, dropout: float):
        super().__init__()
        self.w_gate = nn.Linear(d_model, d_ff, bias=False)
        self.w_val = nn.Linear(d_model, d_ff, bias=False)
        self.w2 = nn.Linear(d_ff, d_model, bias=False)
        self.act = build_activation(activation)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = self.act(self.w_gate(x))
        value = self.w_val(x)
        return self.w2(self.dropout(gate * value))


def build_ffn(cfg: ModelConfig) -> nn.Module:
    if cfg.ffn_type == "glu":
        return GluFeedForward(cfg.d_model, cfg.d_ff, cfg.activation, cfg.dropout)
    return StandardFeedForward(cfg.d_model, cfg.d_ff, cfg.activation, cfg.dropout)


# ─────────────────────────────────────────────
# 5. Multi-Head Attention (with RoPE, QK/V Norm & KV Cache)
# ─────────────────────────────────────────────

# Type alias for a single layer's KV cache: (key, value) tensors
KVCache = tuple[torch.Tensor, torch.Tensor]


class MultiHeadAttention(nn.Module):
    """
    Unified attention supporting MHA / GQA / MQA via cfg.num_kv_heads:
      - MHA : num_kv_heads == num_heads  (standard, default)
      - GQA : 1 < num_kv_heads < num_heads  (e.g. LLaMA-2/3, Mistral)
      - MQA : num_kv_heads == 1  (all query heads share one KV head)

    K/V projections output num_kv_heads * d_head dimensions.
    Before computing attention scores, K/V are expanded to num_heads via
    repeat_interleave so the rest of the computation is identical to MHA.
    The KV cache stores the compact (num_kv_heads) representation to save memory.
    """

    def __init__(
        self,
        cfg: ModelConfig,
        rope: Optional[RotaryEmbedding] = None,
    ):
        super().__init__()
        assert cfg.d_model % cfg.num_heads == 0, "d_model must be divisible by num_heads"

        self.num_heads    = cfg.num_heads
        self.num_kv_heads = cfg.num_kv_heads if cfg.num_kv_heads > 0 else cfg.num_heads
        self.d_head       = cfg.d_model // cfg.num_heads
        self.scale        = math.sqrt(self.d_head)
        self.rope         = rope

        assert self.num_heads % self.num_kv_heads == 0, (
            f"num_heads ({self.num_heads}) must be divisible by num_kv_heads ({self.num_kv_heads})"
        )
        # Number of query heads that share each KV head
        self.kv_groups = self.num_heads // self.num_kv_heads

        kv_dim = self.num_kv_heads * self.d_head  # reduced dim for K and V

        self.w_q = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.w_k = nn.Linear(cfg.d_model, kv_dim, bias=False)
        self.w_v = nn.Linear(cfg.d_model, kv_dim, bias=False)
        self.w_o = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.dropout = nn.Dropout(cfg.dropout)

        # Optional per-head normalization applied after splitting heads.
        # qk_norm: normalize Q and K independently (stabilizes attention logits).
        # v_norm:  normalize V (can improve gradient flow in deep models).
        self.norm_q = build_norm(self.d_head, cfg) if cfg.qk_norm else None
        self.norm_k = build_norm(self.d_head, cfg) if cfg.qk_norm else None
        self.norm_v = build_norm(self.d_head, cfg) if cfg.v_norm  else None

    def _split_heads_q(self, x: torch.Tensor) -> torch.Tensor:
        """[batch, seq, d_model] -> [batch, num_heads, seq, d_head]"""
        batch, seq, _ = x.shape
        return x.view(batch, seq, self.num_heads, self.d_head).transpose(1, 2)

    def _split_heads_kv(self, x: torch.Tensor) -> torch.Tensor:
        """[batch, seq, kv_dim] -> [batch, num_kv_heads, seq, d_head]"""
        batch, seq, _ = x.shape
        return x.view(batch, seq, self.num_kv_heads, self.d_head).transpose(1, 2)

    def _apply_head_norm(
        self, x: torch.Tensor, norm: Optional[nn.Module]
    ) -> torch.Tensor:
        """Apply per-head norm to x: [batch, heads, seq, d_head]."""
        if norm is None:
            return x
        batch, heads, seq, d_head = x.shape
        return norm(x.reshape(-1, d_head)).reshape(batch, heads, seq, d_head)

    def _full_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Standard full attention: O(seq²) memory and compute.

        Args:
            q : [batch, num_heads,    seq_q, d_head]
            k : [batch, num_heads,    seq_k, d_head]
            v : [batch, num_heads,    seq_k, d_head]
            mask : [batch, 1, seq_q, seq_k] or None, True = attend

        Returns:
            context : [batch, num_heads, seq_q, d_head]
        """
        scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))
        attn_weights = self.dropout(F.softmax(scores.float(), dim=-1).to(scores.dtype))
        return torch.matmul(attn_weights, v)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        past_kv: Optional[KVCache] = None,
    ) -> tuple[torch.Tensor, KVCache]:
        """
        Args:
            query, key, value : input tensors [batch, seq, d_model]
            mask              : attention mask [batch, 1, seq_q, seq_k], True = attend
            past_kv           : cached (key, value) from previous steps
                                shapes: [batch, num_kv_heads, past_len, d_head]

        Returns:
            output : [batch, seq_q, d_model]
            new_kv : updated compact KV cache [batch, num_kv_heads, total_len, d_head]
        """
        batch = query.size(0)

        q = self._split_heads_q(self.w_q(query))   # [batch, num_heads,    seq_q, d_head]
        k = self._split_heads_kv(self.w_k(key))    # [batch, num_kv_heads, seq_k, d_head]
        v = self._split_heads_kv(self.w_v(value))  # [batch, num_kv_heads, seq_k, d_head]

        # Optional per-head normalization (before RoPE and cache concat)
        q = self._apply_head_norm(q, self.norm_q)
        k = self._apply_head_norm(k, self.norm_k)
        v = self._apply_head_norm(v, self.norm_v)

        # Apply RoPE to Q and the *new* K only, using the correct position offset.
        # The cached K already has RoPE baked in from previous steps, so we must
        # NOT re-apply RoPE to the full concatenated K — only to the new tokens.
        past_len = past_kv[0].size(2) if past_kv is not None else 0
        if self.rope is not None:
            q = self.rope.apply(q, offset=past_len)
            k = self.rope.apply(k, offset=past_len)  # new K starts at position past_len

        # Concatenate new K/V with the cache *after* RoPE so cached positions are preserved
        if past_kv is not None:
            past_k, past_v = past_kv
            k = torch.cat([past_k, k], dim=2)
            v = torch.cat([past_v, v], dim=2)

        # GQA/MQA: expand K and V from num_kv_heads to num_heads
        # Each KV head is shared by kv_groups query heads.
        # For MHA kv_groups == 1, so this is a no-op.
        if self.kv_groups > 1:
            k = k.repeat_interleave(self.kv_groups, dim=1)  # [batch, num_heads, seq_k, d_head]
            v = v.repeat_interleave(self.kv_groups, dim=1)

        context = self._full_attention(q, k, v, mask)

        context = context.transpose(1, 2).contiguous()
        context = context.view(batch, -1, self.num_heads * self.d_head)

        # Store the compact (unexpanded) KV in the cache to save memory
        new_kv: KVCache = (
            k[:, :: self.kv_groups] if self.kv_groups > 1 else k,
            v[:, :: self.kv_groups] if self.kv_groups > 1 else v,
        )
        return self.w_o(context), new_kv


# ─────────────────────────────────────────────
# 6. Encoder Layer (Pre/Post Norm, Learnable Residual)
# ─────────────────────────────────────────────

class EncoderLayer(nn.Module):
    def __init__(self, cfg: ModelConfig, rope: Optional[RotaryEmbedding] = None):
        super().__init__()
        self.norm_type = cfg.norm_type
        self.self_attn = MultiHeadAttention(cfg, rope)
        self.ff = build_ffn(cfg)
        self.norm1 = build_norm(cfg.d_model, cfg)
        self.norm2 = build_norm(cfg.d_model, cfg)
        self.dropout = nn.Dropout(cfg.dropout)

        # Learnable residual scaling: output = x + alpha * sublayer(x).
        # Initialized to 1.0 so behavior matches standard residual at the start.
        if cfg.learnable_residual:
            self.alpha_attn = nn.Parameter(torch.ones(1))
            self.alpha_ff   = nn.Parameter(torch.ones(1))
        else:
            self.alpha_attn = None
            self.alpha_ff   = None

    def _residual(self, x: torch.Tensor, sublayer_out: torch.Tensor, alpha: Optional[nn.Parameter]) -> torch.Tensor:
        scaled = self.dropout(sublayer_out)
        return x + (alpha * scaled if alpha is not None else scaled)

    def forward(self, x: torch.Tensor, src_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if self.norm_type == "pre":
            normed = self.norm1(x)
            attn_out, _ = self.self_attn(normed, normed, normed, src_mask)
            x = self._residual(x, attn_out, self.alpha_attn)
            x = self._residual(x, self.ff(self.norm2(x)), self.alpha_ff)
        else:
            attn_out, _ = self.self_attn(x, x, x, src_mask)
            x = self.norm1(self._residual(x, attn_out, self.alpha_attn))
            x = self.norm2(self._residual(x, self.ff(x), self.alpha_ff))
        return x


# ─────────────────────────────────────────────
# 7. Decoder Layer (Pre/Post Norm, optional Cross-Attn, KV Cache, Learnable Residual)
# ─────────────────────────────────────────────

class DecoderLayer(nn.Module):
    def __init__(
        self,
        cfg: ModelConfig,
        rope: Optional[RotaryEmbedding] = None,
    ):
        super().__init__()
        self.norm_type = cfg.norm_type
        self.is_lm = cfg.model_type == "lm"

        self.self_attn = MultiHeadAttention(cfg, rope)
        self.ff = build_ffn(cfg)
        self.norm1 = build_norm(cfg.d_model, cfg)
        self.norm_ff = build_norm(cfg.d_model, cfg)
        self.dropout = nn.Dropout(cfg.dropout)

        # Cross-attention only exists in encoder_decoder mode
        if not self.is_lm:
            self.cross_attn = MultiHeadAttention(cfg, rope=None)  # no RoPE for cross-attn
            self.norm2 = build_norm(cfg.d_model, cfg)

        # Learnable residual scaling: output = x + alpha * sublayer(x).
        # Decoder has up to three residual branches: self-attn, cross-attn, ff.
        if cfg.learnable_residual:
            self.alpha_self_attn  = nn.Parameter(torch.ones(1))
            self.alpha_ff         = nn.Parameter(torch.ones(1))
            self.alpha_cross_attn = nn.Parameter(torch.ones(1)) if not self.is_lm else None
        else:
            self.alpha_self_attn  = None
            self.alpha_ff         = None
            self.alpha_cross_attn = None

    def _residual(self, x: torch.Tensor, sublayer_out: torch.Tensor, alpha: Optional[nn.Parameter]) -> torch.Tensor:
        scaled = self.dropout(sublayer_out)
        return x + (alpha * scaled if alpha is not None else scaled)

    def forward(
        self,
        x: torch.Tensor,
        encoder_output: Optional[torch.Tensor] = None,
        src_mask: Optional[torch.Tensor] = None,
        tgt_mask: Optional[torch.Tensor] = None,
        past_self_kv: Optional[KVCache] = None,
    ) -> tuple[torch.Tensor, KVCache]:
        """
        Returns:
            x: updated hidden states
            new_self_kv: updated self-attention KV cache for this layer
        """
        if self.norm_type == "pre":
            normed = self.norm1(x)
            self_attn_out, new_self_kv = self.self_attn(normed, normed, normed, tgt_mask, past_self_kv)

            x = self._residual(x, self_attn_out, self.alpha_self_attn)
            if not self.is_lm and encoder_output is not None:
                normed2 = self.norm2(x)
                cross_out, _ = self.cross_attn(normed2, encoder_output, encoder_output, src_mask)
                x = self._residual(x, cross_out, self.alpha_cross_attn)
            x = self._residual(x, self.ff(self.norm_ff(x)), self.alpha_ff)
        else:
            self_attn_out, new_self_kv = self.self_attn(x, x, x, tgt_mask, past_self_kv)
            x = self.norm1(self._residual(x, self_attn_out, self.alpha_self_attn))
            if not self.is_lm and encoder_output is not None:
                cross_out, _ = self.cross_attn(x, encoder_output, encoder_output, src_mask)
                x = self.norm2(self._residual(x, cross_out, self.alpha_cross_attn))
            x = self.norm_ff(self._residual(x, self.ff(x), self.alpha_ff))

        return x, new_self_kv


# ─────────────────────────────────────────────
# 7. Full Transformer Model
# ─────────────────────────────────────────────

class Transformer(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.d_model = cfg.d_model
        self.is_lm = cfg.model_type == "lm"

        # Token embeddings
        self.token_embedding = nn.Embedding(cfg.vocab_size, cfg.d_model, padding_idx=cfg.pad_idx)
        if not self.is_lm:
            # Separate source embedding for encoder_decoder mode
            self.src_embedding = nn.Embedding(cfg.vocab_size, cfg.d_model, padding_idx=cfg.pad_idx)

        # Optional normalization applied after token embedding (+ positional encoding)
        self.embedding_norm = build_norm(cfg.d_model, cfg) if cfg.embedding_norm else None

        # Positional encoding
        self.rope: Optional[RotaryEmbedding] = None
        self.pos_encoding: Optional[SinusoidalPositionalEncoding] = None

        if cfg.pos_encoding == "rope":
            d_head = cfg.d_model // cfg.num_heads
            self.rope = RotaryEmbedding(d_head, cfg.max_len, base=cfg.rope_base)
        else:
            self.pos_encoding = SinusoidalPositionalEncoding(cfg.d_model, cfg.max_len, cfg.dropout)

        # Encoder (encoder_decoder mode only)
        if not self.is_lm:
            self.encoder_layers = nn.ModuleList(
                [EncoderLayer(cfg, self.rope) for _ in range(cfg.num_encoder_layers)]
            )

        # Decoder
        self.decoder_layers = nn.ModuleList(
            [DecoderLayer(cfg, self.rope) for _ in range(cfg.num_decoder_layers)]
        )

        self.output_norm = build_norm(cfg.d_model, cfg)
        self.output_projection = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

        # Weight tying: share decoder token_embedding and output_projection weights
        if cfg.tie_weights:
            self.output_projection.weight = self.token_embedding.weight

        self._init_weights()

    def _init_weights(self):
        embedding_names = {"token_embedding.weight", "src_embedding.weight"}
        for name, param in self.named_parameters():
            if param.dim() <= 1:
                continue
            if name in embedding_names:
                # Embedding rows act as lookup vectors; initialize with small normal so that
                # the initial logits (hidden @ embedding.T) have unit-ish variance regardless
                # of d_model, preventing the initial loss from being far above log(vocab_size).
                nn.init.normal_(param, mean=0.0, std=self.d_model ** -0.5)
                # Zero out the padding row so it never contributes to the output
                if self.cfg.pad_idx is not None:
                    with torch.no_grad():
                        param[self.cfg.pad_idx].zero_()
            elif self.cfg.tie_weights and name == "output_projection.weight":
                # Already shares memory with token_embedding.weight; skip to avoid overwrite
                continue
            else:
                nn.init.xavier_uniform_(param)

    # ── Mask helpers ───────────────────────────

    def _make_pad_mask(self, seq: torch.Tensor) -> torch.Tensor:
        """[batch, seq] -> [batch, 1, 1, seq], False at pad positions."""
        return (seq != self.cfg.pad_idx).unsqueeze(1).unsqueeze(2)

    def _make_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Lower-triangular causal mask [1, 1, seq, seq]."""
        return torch.tril(
            torch.ones(seq_len, seq_len, device=device, dtype=torch.bool)
        ).unsqueeze(0).unsqueeze(0)

    def _make_single_step_causal_mask(self, past_len: int, device: torch.device) -> torch.Tensor:
        """
        Causal mask for a single new token attending to all past + current positions.
        Shape: [1, 1, 1, past_len + 1], all True (the new token can see everything before it).
        """
        return torch.ones(1, 1, 1, past_len + 1, device=device, dtype=torch.bool)

    def _make_prefill_causal_mask(
        self, seq_len: int, past_len: int, device: torch.device
    ) -> torch.Tensor:
        """
        Causal mask for prefilling *seq_len* new tokens when *past_len* tokens
        are already in the KV cache.

        Each new query token at position (past_len + i) can attend to all
        past tokens [0, past_len) and to new tokens [0, i] (lower-triangular
        within the new chunk).

        Shape: [1, 1, seq_len, past_len + seq_len]
        """
        total_len = past_len + seq_len
        # Full lower-triangular mask over total_len positions
        full_causal = torch.tril(
            torch.ones(total_len, total_len, device=device, dtype=torch.bool)
        )
        # Slice out the rows corresponding to the new tokens only
        new_token_rows = full_causal[past_len : past_len + seq_len, :]  # [seq_len, total_len]
        return new_token_rows.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, total_len]

    # ── Embedding helper ───────────────────────

    def _embed(self, tokens: torch.Tensor, embedding: nn.Embedding) -> torch.Tensor:
        x = embedding(tokens)
        if self.cfg.embedding_scale is not None:
            x = x * self.cfg.embedding_scale
        if self.pos_encoding is not None:
            x = self.pos_encoding(x)
        if self.embedding_norm is not None:
            x = self.embedding_norm(x)
        return x

    # ── Encode (encoder_decoder mode) ──────────

    def encode(self, src: torch.Tensor) -> tuple:
        src_mask = self._make_pad_mask(src)
        x = self._embed(src, self.src_embedding)
        for layer in self.encoder_layers:
            x = layer(x, src_mask)
        return x, src_mask

    # ── Decode ─────────────────────────────────

    def decode(
        self,
        tgt: torch.Tensor,
        encoder_output: Optional[torch.Tensor] = None,
        src_mask: Optional[torch.Tensor] = None,
        past_kvs: Optional[list[KVCache]] = None,
    ) -> tuple[torch.Tensor, list[KVCache]]:
        """
        Args:
            tgt: target token ids [batch, seq]
            encoder_output: encoder hidden states (encoder_decoder mode only)
            src_mask: encoder padding mask
            past_kvs: list of per-layer KV caches from previous decode steps

        Returns:
            hidden: decoder output [batch, seq, d_model]
            new_kvs: updated list of per-layer KV caches
        """
        seq_len = tgt.size(1)
        past_len = past_kvs[0][0].size(2) if past_kvs is not None else 0

        if past_kvs is not None and seq_len == 1:
            # Incremental decoding: single new token, attend to all past + current positions
            tgt_mask = self._make_single_step_causal_mask(past_len, tgt.device)
        elif past_kvs is not None and seq_len > 1:
            # Incremental prefill: multiple new tokens with existing KV cache.
            # Mask shape must be [1, 1, seq_len, past_len + seq_len] so each new
            # query can attend to all cached positions plus its causal prefix.
            tgt_mask = self._make_prefill_causal_mask(seq_len, past_len, tgt.device)
        elif self.is_lm:
            tgt_mask = self._make_causal_mask(seq_len, tgt.device)
        else:
            tgt_pad_mask = self._make_pad_mask(tgt)
            tgt_causal_mask = self._make_causal_mask(seq_len, tgt.device)
            tgt_mask = tgt_pad_mask & tgt_causal_mask

        x = self._embed(tgt, self.token_embedding)
        new_kvs: list[KVCache] = []

        for layer_idx, layer in enumerate(self.decoder_layers):
            layer_past_kv = past_kvs[layer_idx] if past_kvs is not None else None
            x, new_kv = layer(x, encoder_output, src_mask, tgt_mask, layer_past_kv)
            new_kvs.append(new_kv)

        return self.output_norm(x), new_kvs

    # ── Forward (training) ─────────────────────
    def forward(self, src: torch.Tensor, tgt: Optional[torch.Tensor] = None) -> torch.Tensor:
        if self.is_lm:
            # LM mode: src is the full input sequence (prompt + continuation)
            hidden, _ = self.decode(src)
            return self.output_projection(hidden)
        else:
            assert tgt is not None, "encoder_decoder mode requires tgt"
            encoder_output, src_mask = self.encode(src)
            hidden, _ = self.decode(tgt, encoder_output, src_mask)
            return self.output_projection(hidden)

    # ── Serialisation ──────────────────────────────────────────────────────

    def save(self, save_dir: str) -> None:
        """
        Save the model weights and config to *save_dir*.

        Creates two files:
          - ``config.json``  — all ModelConfig fields as JSON
          - ``model.pt``     — model state_dict (via torch.save)

        The directory is created automatically if it does not exist.

        Usage::

            model.save("checkpoints/my_model")
            # Later:
            model = Transformer.from_pretrained("checkpoints/my_model")
        """
        os.makedirs(save_dir, exist_ok=True)

        # Serialise ModelConfig as plain JSON (all fields are primitives).
        config_dict = dataclasses.asdict(self.cfg)
        config_path = os.path.join(save_dir, "config.json")
        with open(config_path, "w", encoding="utf-8") as config_file:
            json.dump(config_dict, config_file, indent=2)

        # Save model weights.
        weights_path = os.path.join(save_dir, "model.pt")
        # Unwrap DDP wrapper if present so we always save the raw state_dict.
        raw_model = self.module if hasattr(self, "module") else self
        torch.save(raw_model.state_dict(), weights_path)

        print(f"  Model saved → {save_dir}  (config.json + model.pt)")

    @classmethod
    def from_pretrained(cls, save_dir: str, device: str = "cpu") -> "Transformer":
        """
        Reconstruct a :class:`Transformer` from a directory created by :meth:`save`.

        Loads ``config.json`` to rebuild the :class:`ModelConfig`, instantiates
        the model, then loads the saved weights.  No model instance is required
        beforehand — the config is fully self-contained.

        Args:
            save_dir : path to the directory produced by :meth:`save`
            device   : torch device string to map weights onto (default ``"cpu"``)

        Returns:
            A :class:`Transformer` in eval mode with weights restored.

        Usage::

            model = Transformer.from_pretrained("checkpoints/my_model", device="cuda")
        """
        config_path = os.path.join(save_dir, "config.json")
        if not os.path.isfile(config_path):
            raise FileNotFoundError(f"config.json not found in {save_dir!r}")

        with open(config_path, "r", encoding="utf-8") as config_file:
            config_dict = json.load(config_file)

        cfg = ModelConfig(**config_dict)
        model = cls(cfg)

        weights_path = os.path.join(save_dir, "model.pt")
        if not os.path.isfile(weights_path):
            raise FileNotFoundError(f"model.pt not found in {save_dir!r}")

        state_dict = torch.load(weights_path, map_location=device, weights_only=True)
        model.load_state_dict(state_dict, strict=False)

        # Cast to the dtype recorded in the config
        dtype_map = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}
        model_dtype = dtype_map.get(cfg.dtype, torch.float32)
        model.to(device=torch.device(device), dtype=model_dtype)
        model.eval()

        print(f"  Model loaded ← {save_dir}  (vocab_size={cfg.vocab_size}, device={device}, dtype={cfg.dtype})")
        return model

# Training utilities (TransformerDataset, collate_fn, make_dataloader,
# TrainingCallback, CopyTokenizerCallback, train) live in train_pipeline.py.
# Import them directly from there to avoid circular imports.

# ─────────────────────────────────────────────
# 9. Sampling Decode (greedy / top-k / top-p / temperature, with KV cache)
# ─────────────────────────────────────────────

def _sample_next_token(
    logits: torch.Tensor,
    temperature: float,
    top_k: int,
    top_p: float,
    repetition_penalty: float = 1.0,
    generated_ids: "list[int] | None" = None,
) -> torch.Tensor:
    """
    Sample the next token from logits [vocab_size].
    temperature=1.0, top_k=1, top_p=1.0 is equivalent to greedy decoding.

    repetition_penalty: values > 1.0 discourage repeating tokens that have
        already appeared in generated_ids.  Positive logits are divided by
        the penalty; negative logits are multiplied by it (Ctrl-paper style).
        1.0 = no penalty.
    generated_ids: flat list of token ids generated so far (prompt excluded).
        Required when repetition_penalty != 1.0; ignored otherwise.
    """
    # Repetition penalty: penalise tokens already present in the output
    if repetition_penalty != 1.0 and generated_ids:
        penalty_mask = torch.zeros_like(logits, dtype=torch.bool)
        for token_id in set(generated_ids):
            penalty_mask[token_id] = True
        # Positive logits → divide (reduce probability); negative → multiply (push further down)
        logits = torch.where(
            penalty_mask,
            torch.where(logits > 0, logits / repetition_penalty, logits * repetition_penalty),
            logits,
        )

    # Temperature scaling
    if temperature != 1.0:
        logits = logits / max(temperature, 1e-8)

    # Top-k filtering: keep only the k highest-probability tokens
    if top_k > 0:
        top_k = min(top_k, logits.size(-1))
        threshold = logits.topk(top_k).values[..., -1, None]
        logits = logits.masked_fill(logits < threshold, float("-inf"))

    # Top-p (nucleus) filtering
    if 0.0 < top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        # Remove tokens whose cumulative probability exceeds top_p
        # (keep the token that just crosses the threshold)
        indices_to_remove = cumulative_probs - F.softmax(sorted_logits, dim=-1) > top_p
        sorted_logits = sorted_logits.masked_fill(indices_to_remove, float("-inf"))
        logits = torch.zeros_like(logits).scatter_(-1, sorted_indices, sorted_logits)

    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)


@torch.no_grad()
def decode(
    model: Transformer,
    src: torch.Tensor,
    max_decode_len: int = 20,
    temperature: float = 1.0,
    top_k: int = 1,
    top_p: float = 1.0,
    repetition_penalty: float = 1.0,
) -> torch.Tensor:
    """
    Autoregressive decoding with KV cache for efficiency.

    Args:
        src                : [1, src_len]
                             - encoder_decoder mode: source sequence
                             - lm mode: prompt tokens (should start with BOS)
        max_decode_len     : maximum number of new tokens to generate
        temperature        : softmax temperature; lower = more deterministic
        top_k              : sample from top-k tokens only; 1 = greedy
        top_p              : nucleus sampling threshold; 1.0 = no filtering
        repetition_penalty : > 1.0 penalises tokens already generated;
                             1.0 = no penalty (default)

    Returns:
        Generated token ids [decoded_len] (excludes BOS; includes EOS if generated).
    """
    model.eval()
    cfg    = model.cfg
    device = src.device

    if model.is_lm:
        # LM mode: prefill the full prompt in one pass to populate the KV cache.
        # Take the last hidden state directly — it already encodes "what comes next"
        # — so the first new token is sampled from it without re-feeding the last token.
        hidden, past_kvs = model.decode(src)
        last_hidden      = hidden[:, -1:, :]   # [1, 1, d_model]
        generated        = src.clone()
        generated_ids: list[int] = []

        prompt_len = src.size(1)
        for _ in range(max_decode_len):
            # Stop if the total sequence length (prompt + generated) would exceed max_len
            if prompt_len + len(generated_ids) >= cfg.max_len:
                break
            next_logits = model.output_projection(last_hidden)[0, -1, :]  # [vocab_size]
            next_token  = _sample_next_token(
                next_logits, temperature, top_k, top_p,
                repetition_penalty=repetition_penalty,
                generated_ids=generated_ids,
            )
            generated   = torch.cat([generated, next_token.unsqueeze(0)], dim=1)
            token_id    = next_token.item()
            if token_id == cfg.eos_idx:
                break
            generated_ids.append(token_id)
            # Feed only the newly generated token; KV cache carries all prior context
            hidden, past_kvs = model.decode(next_token.unsqueeze(0), past_kvs=past_kvs)
            last_hidden      = hidden   # [1, 1, d_model]

        return generated[0, src.size(1):]

    else:
        # encoder_decoder mode: encode source once, then decode autoregressively.
        # Start with BOS; first step has no cache yet so the full prefix is processed.
        encoder_output, src_mask = model.encode(src)
        generated    = torch.tensor([[cfg.bos_idx]], device=device)
        past_kvs     = None
        generated_ids: list[int] = []

        src_len = src.size(1)
        for _ in range(max_decode_len):
            # Stop if total sequence length (src + BOS + generated) would exceed max_len
            if src_len + len(generated_ids) >= cfg.max_len:
                break
            if past_kvs is not None:
                # Incremental step: only feed the last generated token
                hidden, past_kvs = model.decode(
                    generated[:, -1:], encoder_output, src_mask, past_kvs
                )
            else:
                # First step: process the initial BOS token and build the cache
                hidden, past_kvs = model.decode(generated, encoder_output, src_mask)

            next_logits = model.output_projection(hidden)[0, -1, :]  # [vocab_size]
            next_token  = _sample_next_token(
                next_logits, temperature, top_k, top_p,
                repetition_penalty=repetition_penalty,
                generated_ids=generated_ids,
            )
            token_id  = next_token.item()
            generated = torch.cat([generated, next_token.unsqueeze(0)], dim=1)
            if token_id == cfg.eos_idx:
                break
            generated_ids.append(token_id)

        # Strip BOS
        return generated[0, 1:]


@torch.no_grad()
def decode_stream(
    model: Transformer,
    src: torch.Tensor,
    max_decode_len: int = 256,
    temperature: float = 1.0,
    top_k: int = 1,
    top_p: float = 1.0,
    repetition_penalty: float = 1.0,
    past_kvs: list | None = None,
    last_hidden: "torch.Tensor | None" = None,
    out_state: "dict | None" = None,
) -> "Generator[int, None, None]":
    """
    Streaming autoregressive decoding — yields one token id at a time.

    Identical logic to :func:`decode` but implemented as a generator so the
    caller can process / display each token as soon as it is produced, without
    waiting for the full sequence to be generated.

    Args:
        model              : trained Transformer (lm or encoder_decoder)
        src                : [1, src_len] prompt / source tensor on the correct device
        max_decode_len     : maximum number of *new* tokens to generate
        temperature        : softmax temperature (lower = more deterministic)
        top_k              : keep only top-k logits; 1 = greedy
        top_p              : nucleus sampling threshold; 1.0 = disabled
        repetition_penalty : > 1.0 penalises tokens already generated; 1.0 = no penalty
        past_kvs           : (LM only) pre-built KV cache from a prior prefill step.
                             When provided together with ``last_hidden``, the initial
                             ``model.decode(src)`` prefill is skipped and generation
                             starts immediately from the cached state.
        last_hidden        : (LM only) hidden state ``[1, 1, d_model]`` corresponding
                             to the last token of the prefill.  Must be supplied
                             together with ``past_kvs``.
        out_state          : (LM only) optional dict that will be updated in-place
                             with ``{"past_kvs": ..., "generated_ids": [...]}`` after
                             generation finishes.  Lets callers retrieve the updated
                             KV cache without breaking the generator protocol.

    Yields:
        int — one token id per step, stopping before or at EOS.
        The EOS token itself is NOT yielded; the generator simply returns.
    """
    model.eval()
    cfg    = model.cfg
    device = src.device

    if model.is_lm:
        # If a pre-built KV cache and last hidden state are supplied, skip the
        # initial prefill and jump straight into the autoregressive loop.
        if past_kvs is not None and last_hidden is not None:
            current_past_kvs  = past_kvs
            current_hidden    = last_hidden
        else:
            hidden, current_past_kvs = model.decode(src)
            current_hidden           = hidden[:, -1:, :]

        generated_ids: list[int] = []
        prompt_len = src.size(1)

        for _ in range(max_decode_len):
            # Stop if total sequence length (prompt + generated) would exceed max_len
            if prompt_len + len(generated_ids) >= cfg.max_len:
                break
            next_logits = model.output_projection(current_hidden)[0, -1, :]
            next_token  = _sample_next_token(
                next_logits, temperature, top_k, top_p,
                repetition_penalty=repetition_penalty,
                generated_ids=generated_ids,
            )
            token_id    = next_token.item()
            if token_id == cfg.eos_idx:
                break
            generated_ids.append(token_id)
            yield token_id
            hidden, current_past_kvs = model.decode(next_token.unsqueeze(0), past_kvs=current_past_kvs)
            current_hidden           = hidden

        if out_state is not None:
            out_state["past_kvs"]      = current_past_kvs
            out_state["generated_ids"] = generated_ids

    else:
        encoder_output, src_mask = model.encode(src)
        generated     = torch.tensor([[cfg.bos_idx]], device=device)
        past_kvs      = None
        generated_ids = []
        src_len       = src.size(1)

        for _ in range(max_decode_len):
            # Stop if total sequence length (src + BOS + generated) would exceed max_len
            if src_len + len(generated_ids) >= cfg.max_len:
                return
            if past_kvs is not None:
                hidden, past_kvs = model.decode(
                    generated[:, -1:], encoder_output, src_mask, past_kvs
                )
            else:
                hidden, past_kvs = model.decode(generated, encoder_output, src_mask)

            next_logits = model.output_projection(hidden)[0, -1, :]
            next_token  = _sample_next_token(
                next_logits, temperature, top_k, top_p,
                repetition_penalty=repetition_penalty,
                generated_ids=generated_ids,
            )
            token_id  = next_token.item()
            if token_id == cfg.eos_idx:
                return
            generated_ids.append(token_id)
            yield token_id
            generated = torch.cat([generated, next_token.unsqueeze(0)], dim=1)

# ─────────────────────────────────────────────────────────────────────────────
# Quick smoke test (executed when this file is run directly)
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # ── Test 1: LM forward pass ───────────────────────────────────────────────
    cfg = ModelConfig(
        vocab_size=64, pad_idx=0, bos_idx=1, eos_idx=2,
        model_type="lm", d_model=64, num_heads=4,
        num_encoder_layers=0, num_decoder_layers=2,
        d_ff=128, max_len=32, dropout=0.0,
        pos_encoding="rope", ffn_type="glu", activation="silu",
    )
    model = Transformer(cfg)
    model.eval()

    src = torch.randint(3, cfg.vocab_size, (2, 16))  # [batch=2, seq=16]
    logits = model(src)
    assert logits.shape == (2, 16, cfg.vocab_size), logits.shape
    print(f"✅ Test 1: LM forward  logits={logits.shape}")

    # ── Test 2: greedy decode ─────────────────────────────────────────────────
    prompt = torch.tensor([[cfg.bos_idx]])
    generated_ids = decode(model, prompt, max_decode_len=10, top_k=1)
    assert generated_ids.ndim == 1
    print(f"✅ Test 2: greedy decode  tokens={generated_ids.tolist()}")

    # ── Test 3: streaming decode ──────────────────────────────────────────────
    stream_ids = list(decode_stream(model, prompt, max_decode_len=8, top_k=1))
    assert isinstance(stream_ids, list)
    print(f"✅ Test 3: streaming decode  tokens={stream_ids}")

    # ── Test 4: encoder-decoder forward pass ─────────────────────────────────
    cfg_encdec = ModelConfig(
        vocab_size=64, pad_idx=0, bos_idx=1, eos_idx=2,
        model_type="encoder_decoder", d_model=64, num_heads=4,
        num_encoder_layers=2, num_decoder_layers=2,
        d_ff=128, max_len=32, dropout=0.0,
    )
    encdec = Transformer(cfg_encdec)
    encdec.eval()
    enc_src = torch.randint(3, cfg_encdec.vocab_size, (1, 8))
    enc_tgt = torch.randint(3, cfg_encdec.vocab_size, (1, 6))
    enc_logits = encdec(enc_src, enc_tgt)
    assert enc_logits.shape == (1, 6, cfg_encdec.vocab_size)
    print(f"✅ Test 4: encoder-decoder forward  logits={enc_logits.shape}")

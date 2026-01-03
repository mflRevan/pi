"""
SwiGLU Transformer Baseline

A standard Transformer with SwiGLU FFN for fair comparison against RIN/Echo models.
Matches parameter count and layout for apples-to-apples comparison.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple


class SwiGLU(nn.Module):
    """SwiGLU activation: SiLU(xW) * xV"""
    
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.w = nn.Linear(d_model, d_ff, bias=False)
        self.v = nn.Linear(d_model, d_ff, bias=False)
        self.w2 = nn.Linear(d_ff, d_model, bias=False)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w(x)) * self.v(x))


class CausalSelfAttention(nn.Module):
    """Standard causal self-attention with rotary positional embeddings."""
    
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float = 0.0,
        max_seq_len: int = 2048,
    ):
        super().__init__()
        assert d_model % n_heads == 0
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        
        self.qkv_proj = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = 1.0 / math.sqrt(self.d_head)
        
        # Precompute rotary embeddings
        self.max_seq_len = max_seq_len
        self._init_rope(max_seq_len)
    
    def _init_rope(self, max_seq_len: int):
        """Initialize rotary position embeddings."""
        dim = self.d_head
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        t = torch.arange(max_seq_len).float()
        freqs = torch.einsum('i,j->ij', t, inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        self.register_buffer('cos_cached', emb.cos().unsqueeze(0).unsqueeze(0))
        self.register_buffer('sin_cached', emb.sin().unsqueeze(0).unsqueeze(0))
    
    def _apply_rope(self, x: torch.Tensor, seq_len: int) -> torch.Tensor:
        """Apply rotary position embeddings."""
        cos = self.cos_cached[:, :, :seq_len, :]
        sin = self.sin_cached[:, :, :seq_len, :]
        
        x1 = x[..., :x.shape[-1]//2]
        x2 = x[..., x.shape[-1]//2:]
        
        return torch.cat([
            x1 * cos[..., :x1.shape[-1]] - x2 * sin[..., :x1.shape[-1]],
            x2 * cos[..., :x1.shape[-1]] + x1 * sin[..., :x1.shape[-1]]
        ], dim=-1)
    
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        
        # QKV projection
        qkv = self.qkv_proj(x)
        q, k, v = qkv.chunk(3, dim=-1)
        
        # Reshape for multi-head attention
        q = q.view(batch_size, seq_len, self.n_heads, self.d_head).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.n_heads, self.d_head).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.n_heads, self.d_head).transpose(1, 2)
        
        # Apply rotary embeddings
        q = self._apply_rope(q, seq_len)
        k = self._apply_rope(k, seq_len)
        
        # Attention
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        # Causal mask
        if mask is None:
            mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device), diagonal=1).bool()
        scores = scores.masked_fill(mask.unsqueeze(0).unsqueeze(0), float('-inf'))
        
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        
        # Output
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        out = self.out_proj(out)
        
        return out


class TransformerBlock(nn.Module):
    """Standard Transformer block with pre-norm and SwiGLU FFN."""
    
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads, dropout)
        
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = SwiGLU(d_model, d_ff)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Attention with residual
        x = x + self.dropout(self.attn(self.ln1(x), mask))
        # FFN with residual
        x = x + self.dropout(self.ffn(self.ln2(x)))
        return x


class SwiGLUTransformer(nn.Module):
    """
    SwiGLU Transformer baseline for fair comparison.
    
    Standard Transformer architecture with:
    - Rotary position embeddings (RoPE)
    - Pre-normalization
    - SwiGLU FFN
    """
    
    def __init__(
        self,
        vocab_size: int = 50257,
        d_model: int = 128,
        num_layers: int = 2,
        n_heads: int = 8,
        d_ff: Optional[int] = None,
        dropout: float = 0.0,
        max_seq_len: int = 2048,
    ):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.num_layers = num_layers
        self.n_heads = n_heads
        self.d_ff = d_ff or d_model * 4
        
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, self.d_ff, dropout)
            for _ in range(num_layers)
        ])
        
        self.ln_f = nn.LayerNorm(d_model)
        self.output_proj = nn.Linear(d_model, vocab_size, bias=False)
        
        # Weight tying
        self.output_proj.weight = self.token_embedding.weight
        
        self._init_weights()
    
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, std=0.02)
    
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.token_embedding(input_ids)
        
        for block in self.blocks:
            x = block(x)
        
        x = self.ln_f(x)
        logits = self.output_proj(x)
        
        return logits
    
    def compute_loss(self, input_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        logits = self.forward(input_ids)
        
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = input_ids[:, 1:].contiguous()
        
        loss = F.cross_entropy(
            shift_logits.view(-1, self.vocab_size),
            shift_labels.view(-1),
        )
        
        return loss, logits
    
    def get_num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())
    
    def __repr__(self) -> str:
        return (
            f"SwiGLUTransformer(\n"
            f"  vocab_size={self.vocab_size},\n"
            f"  d_model={self.d_model},\n"
            f"  num_layers={self.num_layers},\n"
            f"  n_heads={self.n_heads},\n"
            f"  d_ff={self.d_ff},\n"
            f"  params={self.get_num_params():,}\n"
            f")"
        )

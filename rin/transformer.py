"""
Transformer Models - SwiGLU Baseline and Holographic Transformer

This module provides:
- SwiGLUTransformer: Standard Transformer with SwiGLU FFN (baseline)
- HolographicTransformer: Holographic Transformer with Resonant FFN

Usage:
    from rin.transformer import SwiGLUTransformer, HolographicTransformer
    
    # Baseline
    baseline = SwiGLUTransformer(vocab_size=50257, d_model=512, num_layers=6)
    
    # Holographic (with Omniware FFN, Triton acceleration)
    model = HolographicTransformer(
        vocab_size=50257,
        d_model=512,
        n_layers=6,
        gate_mode='omniware',
        use_triton=True,
    )
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


# =============================================================================
# HOLOGRAPHIC TRANSFORMER
# =============================================================================

class HolographicTransformer(nn.Module):
    """
    Full Holographic Transformer for sequence modeling.
    
    Architecture:
    - Embedding layer with Real/Imag split
    - Stack of HolographicBlocks
    - Output projection to vocabulary
    
    The key innovation is maintaining separate real (content) and imaginary (phase)
    streams throughout, with position information encoded additively in the phase space.
    
    FFN Gate Modes (gate_mode):
    - 'content': Original content-only gating
    - 'time': Position-aware gating (RoPE-style, position only)
    - 'parallel': Time × Content multiplicative gating (two separate activations)
    - 'omniware': Unified time × content theta (single activation, most expressive)
    
    For 'omniware' mode (default), log_grad (default True) enables logarithmic
    gradient scaling: ln(1 + |grad|) * sign(grad) applied to x_imag and w gradients.
    
    This provides natural geometric compression for multiscale time frequencies:
    - Gradient ratio reduction: ~40-500x (from ~2000-4000x to ~2-60x)
    - Compute overhead: <1% (identity forward, simple log backward)
    - Works at any sequence length without numerical instability
    
    Triton Acceleration:
    When use_triton=True (default) and gate_mode='omniware':
    - Uses V2 optimized kernels with autotuned block sizes
    - Two-pass backward without atomic operations
    - <2x overhead vs SwiGLU baseline
    
    Args:
        vocab_size: Vocabulary size
        d_model: Model dimension
        n_heads: Number of attention heads
        n_layers: Number of transformer layers
        n_phase: Phase features for attention (default: 8 * n_heads)
        expansion: FFN expansion factor
        dropout: Dropout rate
        causal: Whether to use causal masking
        max_seq_len: Maximum sequence length
        use_pure_interference: If True, use PureInterferenceAttention
        gate_mode: FFN gating mode ('content', 'time', 'parallel', 'omniware')
        use_triton: Whether to use Triton kernels
        log_grad: Enable log gradient scaling for 'omniware'
    """
    
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 512,
        n_heads: int = 8,
        n_layers: int = 6,
        n_phase: int = None,
        expansion: int = 4,
        dropout: float = 0.0,
        causal: bool = True,
        max_seq_len: int = 8192,
        use_pure_interference: bool = False,
        gate_mode: str = 'omniware',
        use_triton: bool = True,
        log_grad: bool = True,
    ):
        super().__init__()
        
        # Import here to avoid circular imports
        from .block import HolographicBlock
        
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_layers = n_layers
        self.gate_mode = gate_mode
        self.use_triton = use_triton
        self.log_grad = log_grad
        
        # Embeddings for real and imaginary streams
        self.embed_real = nn.Embedding(vocab_size, d_model)
        self.embed_imag = nn.Embedding(vocab_size, d_model)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            HolographicBlock(
                d_model=d_model,
                n_heads=n_heads,
                n_phase=n_phase,
                expansion=expansion,
                dropout=dropout,
                causal=causal,
                max_seq_len=max_seq_len,
                use_pure_interference=use_pure_interference,
                gate_mode=gate_mode,
                use_triton=use_triton,
                log_grad=log_grad,
            )
            for _ in range(n_layers)
        ])
        
        # Output
        self.norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        
        # Weight tying
        self.lm_head.weight = self.embed_real.weight
        
        self._init_weights()
    
    def _init_weights(self):
        nn.init.normal_(self.embed_real.weight, std=0.02)
        nn.init.normal_(self.embed_imag.weight, std=0.02)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            input_ids: Token IDs (B, L)
            mask: Optional attention mask
            
        Returns:
            Logits (B, L, vocab_size)
        """
        # Get embeddings
        x_real = self.embed_real(input_ids)
        x_imag = self.embed_imag(input_ids)
        
        # Process through blocks
        for block in self.blocks:
            x_real, x_imag = block(x_real, x_imag, mask)
        
        # Output
        x = self.norm(x_real)
        logits = self.lm_head(x)
        
        return logits
    
    def compute_loss(self, input_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute cross-entropy loss for language modeling."""
        logits = self.forward(input_ids)
        
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = input_ids[:, 1:].contiguous()
        
        loss = F.cross_entropy(
            shift_logits.view(-1, self.vocab_size),
            shift_labels.view(-1),
        )
        
        return loss, logits
    
    def get_num_params(self) -> int:
        """Return total number of parameters."""
        return sum(p.numel() for p in self.parameters())
    
    def __repr__(self) -> str:
        triton_info = ', triton=True' if self.use_triton else ''
        return (
            f"HolographicTransformer(\n"
            f"  vocab_size={self.vocab_size},\n"
            f"  d_model={self.d_model},\n"
            f"  n_layers={self.n_layers},\n"
            f"  gate_mode={self.gate_mode}{triton_info},\n"
            f"  params={self.get_num_params():,}\n"
            f")"
        )


__all__ = [
    'SwiGLU',
    'CausalSelfAttention',
    'TransformerBlock',
    'SwiGLUTransformer',
    'HolographicTransformer',
]

"""
Echo v2 - Parallel Architecture

Key insight: The Echo model should be parallel like Transformer, not recurrent like RIN.

Architecture:
    1. Token embedding (NOT split into w,b pairs)
    2. Add learnable position embeddings  
    3. Parallel Echo blocks: Attention || Resonant → Add
    4. Output projection

This removes the recurrent bottleneck and enables batch-parallel processing.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, List

from .lut import get_global_lut

PHI = (1 + math.sqrt(5)) / 2


class ResonantFFN(nn.Module):
    """
    Resonant Feed-Forward Network with per-neuron interference.
    """
    
    def __init__(self, d_model: int, num_neurons: int):
        super().__init__()
        self.d_model = d_model
        self.num_neurons = num_neurons
        
        self.W = nn.Parameter(torch.randn(num_neurons, d_model) * 0.02)
        self.B = nn.Parameter(torch.zeros(num_neurons, d_model))
        
        self.proj_cos = nn.Linear(num_neurons, d_model, bias=False)
        self.proj_sin = nn.Linear(num_neurons, d_model, bias=False)
        
        self._lut = None
        self._init_weights()
    
    def _init_weights(self):
        nn.init.xavier_uniform_(self.proj_cos.weight, gain=0.5)
        nn.init.xavier_uniform_(self.proj_sin.weight, gain=0.5)
    
    def _get_lut(self, device):
        if self._lut is None or self._lut.sin_table.device != device:
            self._lut = get_global_lut(4096, device)
        return self._lut
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, d_model)
        Returns:
            output: (batch, seq_len, d_model)
        """
        lut = self._get_lut(x.device)
        
        # x: (B, S, D) → (B, S, 1, D)
        x_exp = x.unsqueeze(-2)
        
        # Phase: θ[b,s,n,d] = x[b,s,d] / (1+|W[n,d]|) + B[n,d]
        wavelength = 1.0 + self.W.abs()
        theta = x_exp / wavelength + self.B  # (B, S, N, D)
        
        sin_theta, cos_theta = lut.lookup_sin_cos(theta)
        
        # Interference sum across d_model
        cos_sum = cos_theta.sum(dim=-1)  # (B, S, N)
        sin_sum = sin_theta.sum(dim=-1)  # (B, S, N)
        
        # Project back
        output = self.proj_cos(cos_sum) + self.proj_sin(sin_sum)  # (B, S, D)
        
        return F.silu(output)


class EulerCausalAttention(nn.Module):
    """
    Causal self-attention with Euler transform for Q/K.
    """
    
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        assert d_model % n_heads == 0
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        
        # Q/K use Euler, V is linear
        self.w_q = nn.Parameter(torch.randn(n_heads, self.d_head) * 0.02)
        self.b_q = nn.Parameter(torch.zeros(n_heads, self.d_head))
        
        self.w_k = nn.Parameter(torch.randn(n_heads, self.d_head) * 0.02)
        self.b_k = nn.Parameter(torch.zeros(n_heads, self.d_head))
        
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(2 * self.d_head)
        
        self._lut = None
    
    def _get_lut(self, device):
        if self._lut is None or self._lut.sin_table.device != device:
            self._lut = get_global_lut(4096, device)
        return self._lut
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, d_model)
        Returns:
            output: (batch, seq_len, d_model)
        """
        B, S, D = x.shape
        lut = self._get_lut(x.device)
        
        # Reshape for multi-head: (B, S, n_heads, d_head)
        x_heads = x.view(B, S, self.n_heads, self.d_head)
        
        # Euler transform for Q
        wl_q = 1.0 + self.w_q.abs()  # (n_heads, d_head)
        theta_q = x_heads / wl_q + self.b_q  # (B, S, H, d_head)
        sin_q, cos_q = lut.lookup_sin_cos(theta_q)
        Q = torch.cat([cos_q, sin_q], dim=-1)  # (B, S, H, 2*d_head)
        
        # Euler transform for K
        wl_k = 1.0 + self.w_k.abs()
        theta_k = x_heads / wl_k + self.b_k
        sin_k, cos_k = lut.lookup_sin_cos(theta_k)
        K = torch.cat([cos_k, sin_k], dim=-1)  # (B, S, H, 2*d_head)
        
        # V is standard linear
        V = self.v_proj(x).view(B, S, self.n_heads, self.d_head)  # (B, S, H, d_head)
        
        # Transpose for attention: (B, H, S, *)
        Q = Q.transpose(1, 2)  # (B, H, S, 2*d_head)
        K = K.transpose(1, 2)  # (B, H, S, 2*d_head)
        V = V.transpose(1, 2)  # (B, H, S, d_head)
        
        # Attention: Q @ K^T gives cos(θ_q - θ_k) implicitly
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale  # (B, H, S, S)
        
        # Causal mask
        if mask is None:
            mask = torch.triu(torch.ones(S, S, device=x.device), diagonal=1).bool()
        scores = scores.masked_fill(mask.unsqueeze(0).unsqueeze(0), float('-inf'))
        
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        
        out = torch.matmul(attn, V)  # (B, H, S, d_head)
        out = out.transpose(1, 2).contiguous().view(B, S, D)
        out = self.out_proj(out)
        
        return out


class EchoBlock(nn.Module):
    """
    Echo Block: Parallel Attention + Resonant with Additive Fusion
    
    Unlike Transformer's sequential Attn → FFN,
    Echo uses parallel: (Attn || Res) → Add
    """
    
    def __init__(
        self,
        d_model: int,
        num_neurons: int,
        n_heads: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        
        self.ln = nn.LayerNorm(d_model)
        
        self.attention = EulerCausalAttention(d_model, n_heads, dropout)
        self.resonant = ResonantFFN(d_model, num_neurons)
        
        # Learnable scales for additive fusion
        self.attn_scale = nn.Parameter(torch.tensor(0.5))
        self.res_scale = nn.Parameter(torch.tensor(0.5))
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pre-norm
        normed = self.ln(x)
        
        # Parallel computation
        attn_out = self.attention(normed)
        res_out = self.resonant(normed)
        
        # Additive fusion with residual
        output = x + self.dropout(self.attn_scale * attn_out + self.res_scale * res_out)
        
        return output


class EchoModelV2(nn.Module):
    """
    Echo Model V2 - Parallel Architecture
    
    Removes recurrent bottleneck. Processes full sequence in parallel.
    """
    
    def __init__(
        self,
        vocab_size: int = 50257,
        d_model: int = 128,
        num_layers: int = 2,
        num_neurons: int = 256,
        n_heads: int = 8,
        max_seq_len: int = 2048,
        dropout: float = 0.0,
    ):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.num_layers = num_layers
        
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Embedding(max_seq_len, d_model)
        
        self.blocks = nn.ModuleList([
            EchoBlock(d_model, num_neurons, n_heads, dropout)
            for _ in range(num_layers)
        ])
        
        self.ln_f = nn.LayerNorm(d_model)
        self.output_proj = nn.Linear(d_model, vocab_size, bias=False)
        
        # Weight tying
        self.output_proj.weight = self.token_embedding.weight
        
        self._init_weights()
    
    def _init_weights(self):
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.pos_embedding.weight, std=0.02)
    
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        B, S = input_ids.shape
        device = input_ids.device
        
        # Embeddings
        tok_emb = self.token_embedding(input_ids)
        pos = torch.arange(S, device=device)
        pos_emb = self.pos_embedding(pos)
        
        x = tok_emb + pos_emb
        
        # Echo blocks
        for block in self.blocks:
            x = block(x)
        
        x = self.ln_f(x)
        logits = self.output_proj(x)
        
        return logits
    
    def get_num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())
    
    def __repr__(self) -> str:
        return (
            f"EchoModelV2(\n"
            f"  vocab_size={self.vocab_size},\n"
            f"  d_model={self.d_model},\n"
            f"  num_layers={self.num_layers},\n"
            f"  params={self.get_num_params():,}\n"
            f")"
        )

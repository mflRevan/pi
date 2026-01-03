"""
Echo Chamber Block - Parallel Attention + Resonant with Additive Fusion

KEY INSIGHT: Additive fusion of attention and resonant outputs enables:
- Both pathways learn specialized representations independently
- Strong gradient flow to both components (no gating suppression)
- 95%+ accuracy across all distances vs 76% for multiplicative

Architecture:
    EchoBlock = Parallel(EulerAttention, ResonantLayer) + Add
    
    Input x → ┬─ EulerAttention(x, cache, t) ─┬─ + → output
              └─ ResonantLayer(x, t)        ──┘

This differs from Transformer which is sequential:
    Transformer = Attention → Add → FFN → Add
    
Echo is parallel:
    Echo = (Attention || Resonant) → Add

The resonant layer provides fixed-wavelength interference patterns while
attention provides learned content-based retrieval. Together they create
a rich representation space.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, List

from .lut import get_global_lut

# Golden ratio
PHI = (1 + math.sqrt(5)) / 2


class StateCache:
    """State history cache for echo attention."""
    
    def __init__(self, max_len: Optional[int] = None, training: bool = True):
        self.max_len = max_len
        self.training = training
        self.reset()
    
    def reset(self):
        self.states: List[torch.Tensor] = []
        self._stacked_states: Optional[torch.Tensor] = None
        self._stacked_valid = False
    
    def append(self, state: torch.Tensor):
        if self.training:
            self.states.append(state)
        else:
            self.states.append(state.detach())
        self._stacked_valid = False
        
        if self.max_len is not None and len(self.states) > self.max_len:
            self.states.pop(0)
    
    def get_stacked_states(self) -> Optional[torch.Tensor]:
        if len(self.states) == 0:
            return None
        if not self._stacked_valid:
            self._stacked_states = torch.stack(self.states, dim=1)
            self._stacked_valid = True
        return self._stacked_states
    
    def __len__(self) -> int:
        return len(self.states)


class EulerAttentionHead(nn.Module):
    """
    Single attention head using Euler transform for query/key projection.
    
    The Euler transform projects to phase space:
        θ = x / (1 + |w|) + b + t·φ
        query = [cos(θ), sin(θ)]
        
    Dot product computes phase alignment:
        score = q · k = cos(θ_q - θ_k)
    """
    
    def __init__(self, d_model: int, d_head: int, head_idx: int):
        super().__init__()
        self.d_head = d_head
        self.start_idx = head_idx * d_head
        self.end_idx = (head_idx + 1) * d_head
        
        # Query Euler parameters
        self.w_query = nn.Parameter(torch.randn(d_head) * 0.02)
        self.b_query = nn.Parameter(torch.zeros(d_head))
        
        # Key Euler parameters  
        self.w_key = nn.Parameter(torch.randn(d_head) * 0.02)
        self.b_key = nn.Parameter(torch.zeros(d_head))
        
        self.scale = math.sqrt(2 * d_head)
        self._lut = None
    
    def _get_lut(self, device):
        if self._lut is None or self._lut.sin_table.device != device:
            self._lut = get_global_lut(4096, device)
        return self._lut
    
    def forward(
        self,
        x: torch.Tensor,
        cached_states: torch.Tensor,
        t: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        lut = self._get_lut(x.device)
        
        # Query: Euler transform
        x_patch = x[:, self.start_idx:self.end_idx]
        wl_q = 1.0 + self.w_query.abs()
        t_phi = t.unsqueeze(-1) * PHI if t.dim() == 1 else t * PHI
        theta_q = x_patch / wl_q + self.b_query + t_phi
        sin_q, cos_q = lut.lookup_sin_cos(theta_q)
        query = torch.cat([cos_q, sin_q], dim=-1)
        
        # Keys: Euler transform (no time - static representations)
        k_patches = cached_states[:, :, self.start_idx:self.end_idx]
        wl_k = 1.0 + self.w_key.abs()
        theta_k = k_patches / wl_k + self.b_key
        sin_k, cos_k = lut.lookup_sin_cos(theta_k)
        keys = torch.cat([cos_k, sin_k], dim=-1)
        
        # Attention: score = cos(θ_q - θ_k)
        scores = torch.bmm(query.unsqueeze(1), keys.transpose(1, 2)).squeeze(1)
        scores = scores / self.scale
        weights = F.softmax(scores, dim=-1)
        
        # Retrieve full states
        output = torch.bmm(weights.unsqueeze(1), cached_states).squeeze(1)
        return output, weights


class EulerAttention(nn.Module):
    """
    Multi-head Euler Attention.
    
    Uses Euler transform for resonance-based query/key matching.
    Output is projected via Euler transform as well (not linear).
    """
    
    def __init__(
        self,
        d_model: int,
        n_heads: int = 8,
        dropout: float = 0.0,
    ):
        super().__init__()
        assert d_model % n_heads == 0
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        
        self.heads = nn.ModuleList([
            EulerAttentionHead(d_model, self.d_head, i)
            for i in range(n_heads)
        ])
        
        # Euler output projection (not linear!)
        self.w_out = nn.Parameter(torch.randn(d_model) * 0.02)
        self.b_out = nn.Parameter(torch.zeros(d_model))
        self.out_scale = nn.Parameter(torch.ones(d_model) * 0.5)
        
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self._lut = None
    
    def _get_lut(self, device):
        if self._lut is None or self._lut.sin_table.device != device:
            self._lut = get_global_lut(4096, device)
        return self._lut
    
    def forward(
        self,
        x: torch.Tensor,
        cache: StateCache,
        t: torch.Tensor,
        return_weights: bool = False,
    ) -> Tuple[torch.Tensor, Optional[List[torch.Tensor]]]:
        cached_states = cache.get_stacked_states()
        if cached_states is None or len(cache) == 0:
            return torch.zeros_like(x), None
        
        # Multi-head attention
        head_outputs = []
        all_weights = [] if return_weights else None
        
        for head in self.heads:
            output, weights = head(x, cached_states, t)
            head_outputs.append(output)
            if return_weights:
                all_weights.append(weights)
        
        # Sum heads (interference)
        context = torch.stack(head_outputs, dim=0).sum(dim=0)
        
        # Euler output projection
        lut = self._get_lut(x.device)
        wl_out = 1.0 + self.w_out.abs()
        t_phi = t.unsqueeze(-1) * PHI if t.dim() == 1 else t * PHI
        theta_out = context / wl_out + self.b_out + t_phi
        sin_out, cos_out = lut.lookup_sin_cos(theta_out)
        
        # Combine cos and sin with learned scale
        output = self.out_scale * (cos_out + sin_out)
        output = self.dropout(output)
        
        return output, all_weights


class ResonantFFN(nn.Module):
    """
    Resonant Feed-Forward Network with per-neuron interference.
    
    Like MLP but with interference analysis instead of weighted sums:
        θ[n,d] = x[d] / (1 + |W[n,d]|) + B[n,d] + t
        cos_sum[n] = Σ_d cos(θ[n,d])
        sin_sum[n] = Σ_d sin(θ[n,d])
    """
    
    def __init__(
        self,
        d_model: int,
        num_neurons: int,
        use_swish: bool = True,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_neurons = num_neurons
        self.use_swish = use_swish
        
        # Per-neuron, per-dimension parameters
        self.W = nn.Parameter(torch.randn(num_neurons, d_model) * 0.02)
        self.B = nn.Parameter(torch.zeros(num_neurons, d_model))
        
        # Output projections
        self.proj_real = nn.Linear(num_neurons, d_model, bias=False)
        self.proj_imag = nn.Linear(num_neurons, d_model, bias=False)
        
        self._lut = None
        self._init_weights()
    
    def _init_weights(self):
        nn.init.xavier_uniform_(self.proj_real.weight, gain=0.5)
        nn.init.xavier_uniform_(self.proj_imag.weight, gain=0.5)
    
    def _get_lut(self, device):
        if self._lut is None or self._lut.sin_table.device != device:
            self._lut = get_global_lut(4096, device)
        return self._lut
    
    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        lut = self._get_lut(x.device)
        
        x_expanded = x.unsqueeze(1)
        wavelength = 1.0 + self.W.abs()
        
        if t.dim() == 1:
            t = t.unsqueeze(-1).unsqueeze(-1)
        elif t.dim() == 2:
            t = t.unsqueeze(-1)
        
        theta = x_expanded / wavelength + self.B + t
        sin_theta, cos_theta = lut.lookup_sin_cos(theta)
        
        # Interference sum across d_model
        cos_sum = cos_theta.sum(dim=-1)
        sin_sum = sin_theta.sum(dim=-1)
        
        output = self.proj_real(cos_sum) + self.proj_imag(sin_sum)
        
        if self.use_swish:
            output = F.silu(output)
        
        return output


class EchoBlock(nn.Module):
    """
    Echo Chamber Block - Parallel Attention + Resonant with Additive Fusion
    
    Architecture:
        Input x → ┬─ EulerAttention(x, cache, t) ─┬─ + → output
                  └─ ResonantFFN(x, t)           ─┘
    
    Key insight: Additive fusion preserves gradient flow to both pathways
    equally, enabling each to develop specialized representations.
    """
    
    def __init__(
        self,
        d_model: int,
        num_neurons: int,
        n_heads: int = 8,
        use_swish: bool = True,
        dropout: float = 0.0,
    ):
        super().__init__()
        
        self.attention = EulerAttention(
            d_model=d_model,
            n_heads=n_heads,
            dropout=dropout,
        )
        
        self.resonant = ResonantFFN(
            d_model=d_model,
            num_neurons=num_neurons,
            use_swish=use_swish,
        )
        
        # Learnable mixing weights for additive fusion
        self.attn_scale = nn.Parameter(torch.ones(1))
        self.res_scale = nn.Parameter(torch.ones(1))
    
    def forward(
        self,
        x: torch.Tensor,
        cache: StateCache,
        t: torch.Tensor,
        return_weights: bool = False,
    ) -> Tuple[torch.Tensor, Optional[List[torch.Tensor]]]:
        """
        Parallel computation with additive fusion.
        """
        # Parallel pathways
        attn_out, weights = self.attention(x, cache, t, return_weights)
        res_out = self.resonant(x, t)
        
        # Additive fusion with learnable scales
        output = x + self.attn_scale * attn_out + self.res_scale * res_out
        
        return output, weights


class EchoModel(nn.Module):
    """
    Echo Chamber Model - RIN with Echo Blocks
    
    Full model using Euler state evolution + Echo blocks for processing.
    
    Architecture:
        1. Token embedding → (w, b) pairs
        2. Euler state transformation (recurrent)
        3. Cache state for attention
        4. Process through EchoBlocks (parallel attention + resonant)
        5. Output projection
    """
    
    def __init__(
        self,
        vocab_size: int = 50257,
        d_model: int = 128,
        num_layers: int = 2,
        num_neurons: int = 256,
        n_heads: int = 8,
        use_swish: bool = True,
        max_cache_len: Optional[int] = None,
        dropout: float = 0.0,
    ):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_neurons = num_neurons
        self.n_heads = n_heads
        self.max_cache_len = max_cache_len
        
        # Token embeddings: 2*d_model for (w, b) pairs
        self.token_embedding = nn.Embedding(vocab_size, 2 * d_model)
        
        # Echo blocks
        self.blocks = nn.ModuleList([
            EchoBlock(
                d_model=d_model,
                num_neurons=num_neurons,
                n_heads=n_heads,
                use_swish=use_swish,
                dropout=dropout,
            )
            for _ in range(num_layers)
        ])
        
        # Output projection
        self.output_proj = nn.Linear(d_model, vocab_size, bias=False)
        
        self._lut = None
        self._init_weights()
    
    def _init_weights(self):
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.output_proj.weight, std=0.02)
        with torch.no_grad():
            self.token_embedding.weight.mul_(0.5)
    
    def _get_lut(self, device):
        if self._lut is None or self._lut.sin_table.device != device:
            self._lut = get_global_lut(4096, device)
        return self._lut
    
    def init_hidden(self, batch_size: int, device: torch.device):
        h_real = torch.zeros(batch_size, self.d_model, device=device)
        h_imag = torch.zeros(batch_size, self.d_model, device=device)
        return h_real, h_imag
    
    def create_cache(self, training: Optional[bool] = None) -> StateCache:
        if training is None:
            training = self.training
        return StateCache(max_len=self.max_cache_len, training=training)
    
    def euler_transform(
        self,
        h_real: torch.Tensor,
        h_imag: torch.Tensor,
        w: torch.Tensor,
        b: torch.Tensor,
        t: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Euler-based state transformation."""
        lut = self._get_lut(h_real.device)
        
        wavelength = 1.0 + w.abs()
        t_phi = t.unsqueeze(-1) * PHI if t.dim() == 1 else t * PHI
        
        theta_real = h_real / wavelength + b + t_phi
        theta_imag = h_imag / wavelength + b + t_phi
        
        sin_real, cos_real = lut.lookup_sin_cos(theta_real)
        sin_imag, cos_imag = lut.lookup_sin_cos(theta_imag)
        
        h_real_new = cos_real * cos_imag - sin_real * sin_imag
        h_imag_new = cos_real * sin_imag + sin_real * cos_imag
        
        return h_real_new, h_imag_new
    
    def forward(
        self,
        input_ids: torch.Tensor,
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        cache: Optional[StateCache] = None,
        t_start: int = 0,
        return_attention: bool = False,
    ):
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        
        if hidden is None:
            h_real, h_imag = self.init_hidden(batch_size, device)
        else:
            h_real, h_imag = hidden
        
        if cache is None:
            cache = self.create_cache(training=self.training)
        
        embeddings = self.token_embedding(input_ids)
        w_emb = embeddings[:, :, :self.d_model]
        b_emb = embeddings[:, :, self.d_model:]
        
        t_indices = torch.arange(seq_len, device=device, dtype=torch.float32) + t_start
        
        all_logits = []
        all_attention = [] if return_attention else None
        
        for t_idx in range(seq_len):
            w_t = w_emb[:, t_idx]
            b_t = b_emb[:, t_idx]
            t_val = t_indices[t_idx].expand(batch_size)
            
            # Euler state transformation
            h_real, h_imag = self.euler_transform(h_real, h_imag, w_t, b_t, t_val)
            x = h_real + h_imag
            
            # Cache BEFORE attention
            cache.append(x)
            
            # Process through Echo blocks
            t_phi = t_val * PHI
            block_attention = []
            for block in self.blocks:
                x, weights = block(x, cache, t_phi, return_weights=return_attention)
                if return_attention and weights is not None:
                    block_attention.append(weights)
            
            if return_attention:
                all_attention.append(block_attention)
            
            logits = self.output_proj(x)
            all_logits.append(logits)
        
        output = torch.stack(all_logits, dim=1)
        
        if return_attention:
            return output, (h_real, h_imag), cache, all_attention
        return output, (h_real, h_imag), cache
    
    def compute_loss(self, input_ids: torch.Tensor, **kwargs):
        logits, hidden, cache = self.forward(input_ids, **kwargs)
        
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = input_ids[:, 1:].contiguous()
        
        loss = F.cross_entropy(
            shift_logits.view(-1, self.vocab_size),
            shift_labels.view(-1),
        )
        
        return loss, logits, hidden, cache
    
    def get_num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())
    
    def __repr__(self) -> str:
        return (
            f"EchoModel(\n"
            f"  vocab_size={self.vocab_size},\n"
            f"  d_model={self.d_model},\n"
            f"  num_layers={self.num_layers},\n"
            f"  num_neurons={self.num_neurons},\n"
            f"  n_heads={self.n_heads},\n"
            f"  max_cache_len={self.max_cache_len},\n"
            f"  params={self.get_num_params():,}\n"
            f")"
        )

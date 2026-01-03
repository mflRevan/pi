"""
Test Attention Variants for Echo Model

Tests:
1. Current: Q(t=0), K(t=0) - time invariant
2. Q(t), K(t=0) - query time-dependent, key static
3. Q(t), K(t) - both time-dependent
4. All with Euler Value projection (not linear)

Then: Episodic Context approach with state history attention
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
import math
import time
from typing import Optional, Tuple, List
import sys

sys.path.insert(0, '/home/aiman/pi')

from rin.lut import get_global_lut

PHI = (1 + math.sqrt(5)) / 2


# =============================================================================
# Attention Variants
# =============================================================================

class EulerAttentionVariant(nn.Module):
    """
    Euler Attention with configurable time dependence.
    
    Variants:
        mode='static': Q(t=0), K(t=0) - current broken implementation
        mode='query_time': Q(t), K(t=0) - query evolves, keys static
        mode='both_time': Q(t), K(t_cached) - both use their timesteps
    
    CRITICAL FIX: Value projection is now Euler transform, not linear!
    Values are phase states that should be transformed with respect to t.
    """
    
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        mode: str = 'query_time',  # 'static', 'query_time', 'both_time'
        dropout: float = 0.0,
    ):
        super().__init__()
        assert d_model % n_heads == 0
        assert mode in ('static', 'query_time', 'both_time')
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.mode = mode
        
        # Query Euler params (per head)
        self.w_q = nn.Parameter(torch.randn(n_heads, self.d_head) * 0.02)
        self.b_q = nn.Parameter(torch.zeros(n_heads, self.d_head))
        
        # Key Euler params
        self.w_k = nn.Parameter(torch.randn(n_heads, self.d_head) * 0.02)
        self.b_k = nn.Parameter(torch.zeros(n_heads, self.d_head))
        
        # Value Euler params - CRITICAL: values are phase states!
        self.w_v = nn.Parameter(torch.randn(n_heads, self.d_head) * 0.02)
        self.b_v = nn.Parameter(torch.zeros(n_heads, self.d_head))
        
        # Output projection (also Euler)
        self.w_out = nn.Parameter(torch.randn(d_model) * 0.02)
        self.b_out = nn.Parameter(torch.zeros(d_model))
        
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(2 * self.d_head)
        
        self._lut = None
    
    def _get_lut(self, device):
        if self._lut is None or self._lut.sin_table.device != device:
            self._lut = get_global_lut(4096, device)
        return self._lut
    
    def forward(
        self,
        x: torch.Tensor,
        positions: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, d_model)
            positions: (seq_len,) timestep positions for time-dependent modes
            mask: causal mask
        """
        B, S, D = x.shape
        lut = self._get_lut(x.device)
        
        if positions is None:
            positions = torch.arange(S, device=x.device, dtype=torch.float32)
        
        # Reshape for heads: (B, S, H, d_head)
        x_heads = x.view(B, S, self.n_heads, self.d_head)
        
        # === QUERY ===
        wl_q = 1.0 + self.w_q.abs()
        if self.mode in ('query_time', 'both_time'):
            # Time-dependent query: add t * φ
            t_q = (positions * PHI).view(1, S, 1, 1)
            theta_q = x_heads / wl_q + self.b_q + t_q
        else:
            # Static query (broken - t=0)
            theta_q = x_heads / wl_q + self.b_q
        
        sin_q, cos_q = lut.lookup_sin_cos(theta_q)
        Q = torch.cat([cos_q, sin_q], dim=-1)  # (B, S, H, 2*d_head)
        
        # === KEY ===
        wl_k = 1.0 + self.w_k.abs()
        if self.mode == 'both_time':
            # Each key uses its own timestep
            t_k = (positions * PHI).view(1, S, 1, 1)
            theta_k = x_heads / wl_k + self.b_k + t_k
        else:
            # Static keys
            theta_k = x_heads / wl_k + self.b_k
        
        sin_k, cos_k = lut.lookup_sin_cos(theta_k)
        K = torch.cat([cos_k, sin_k], dim=-1)  # (B, S, H, 2*d_head)
        
        # === VALUE (Euler transform!) ===
        wl_v = 1.0 + self.w_v.abs()
        # Values always use their timestep - they're phase states!
        t_v = (positions * PHI).view(1, S, 1, 1)
        theta_v = x_heads / wl_v + self.b_v + t_v
        sin_v, cos_v = lut.lookup_sin_cos(theta_v)
        # V keeps both cos and sin as complex representation
        V_cos = cos_v  # (B, S, H, d_head)
        V_sin = sin_v
        
        # Transpose for attention: (B, H, S, *)
        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2)
        V_cos = V_cos.transpose(1, 2)
        V_sin = V_sin.transpose(1, 2)
        
        # Attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        
        # Causal mask
        if mask is None:
            mask = torch.triu(torch.ones(S, S, device=x.device), diagonal=1).bool()
        scores = scores.masked_fill(mask.unsqueeze(0).unsqueeze(0), float('-inf'))
        
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        
        # Retrieve - keep complex structure
        out_cos = torch.matmul(attn, V_cos)  # (B, H, S, d_head)
        out_sin = torch.matmul(attn, V_sin)
        
        # Combine heads via sum (interference) then reshape
        out_cos = out_cos.transpose(1, 2).contiguous().view(B, S, D)
        out_sin = out_sin.transpose(1, 2).contiguous().view(B, S, D)
        
        # Output Euler projection
        wl_out = 1.0 + self.w_out.abs()
        # Combine cos and sin into output theta
        theta_out = (out_cos + out_sin) / wl_out + self.b_out
        sin_out, cos_out = lut.lookup_sin_cos(theta_out)
        
        output = cos_out + sin_out
        
        return output


class ResonantFFN(nn.Module):
    """Resonant FFN (unchanged from echo_v2)."""
    
    def __init__(self, d_model: int, num_neurons: int):
        super().__init__()
        self.d_model = d_model
        self.num_neurons = num_neurons
        
        self.W = nn.Parameter(torch.randn(num_neurons, d_model) * 0.02)
        self.B = nn.Parameter(torch.zeros(num_neurons, d_model))
        
        self.proj_cos = nn.Linear(num_neurons, d_model, bias=False)
        self.proj_sin = nn.Linear(num_neurons, d_model, bias=False)
        
        self._lut = None
        nn.init.xavier_uniform_(self.proj_cos.weight, gain=0.5)
        nn.init.xavier_uniform_(self.proj_sin.weight, gain=0.5)
    
    def _get_lut(self, device):
        if self._lut is None or self._lut.sin_table.device != device:
            self._lut = get_global_lut(4096, device)
        return self._lut
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lut = self._get_lut(x.device)
        x_exp = x.unsqueeze(-2)
        wavelength = 1.0 + self.W.abs()
        theta = x_exp / wavelength + self.B
        sin_theta, cos_theta = lut.lookup_sin_cos(theta)
        cos_sum = cos_theta.sum(dim=-1)
        sin_sum = sin_theta.sum(dim=-1)
        output = self.proj_cos(cos_sum) + self.proj_sin(sin_sum)
        return F.silu(output)


class EchoBlockVariant(nn.Module):
    """Echo Block with configurable attention variant."""
    
    def __init__(
        self,
        d_model: int,
        num_neurons: int,
        n_heads: int,
        attention_mode: str = 'query_time',
        dropout: float = 0.0,
    ):
        super().__init__()
        
        self.ln = nn.LayerNorm(d_model)
        self.attention = EulerAttentionVariant(d_model, n_heads, attention_mode, dropout)
        self.resonant = ResonantFFN(d_model, num_neurons)
        
        self.attn_scale = nn.Parameter(torch.tensor(0.5))
        self.res_scale = nn.Parameter(torch.tensor(0.5))
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, positions: torch.Tensor = None) -> torch.Tensor:
        normed = self.ln(x)
        attn_out = self.attention(normed, positions)
        res_out = self.resonant(normed)
        output = x + self.dropout(self.attn_scale * attn_out + self.res_scale * res_out)
        return output


class EchoModelVariant(nn.Module):
    """Echo Model with configurable attention mode."""
    
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        num_layers: int,
        num_neurons: int,
        n_heads: int,
        attention_mode: str = 'query_time',
        max_seq_len: int = 2048,
    ):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.attention_mode = attention_mode
        
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Embedding(max_seq_len, d_model)
        
        self.blocks = nn.ModuleList([
            EchoBlockVariant(d_model, num_neurons, n_heads, attention_mode)
            for _ in range(num_layers)
        ])
        
        self.ln_f = nn.LayerNorm(d_model)
        self.output_proj = nn.Linear(d_model, vocab_size, bias=False)
        self.output_proj.weight = self.token_embedding.weight
        
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.pos_embedding.weight, std=0.02)
    
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        B, S = input_ids.shape
        device = input_ids.device
        
        tok_emb = self.token_embedding(input_ids)
        pos = torch.arange(S, device=device)
        pos_emb = self.pos_embedding(pos)
        
        x = tok_emb + pos_emb
        positions = torch.arange(S, device=device, dtype=torch.float32)
        
        for block in self.blocks:
            x = block(x, positions)
        
        x = self.ln_f(x)
        return self.output_proj(x)


# =============================================================================
# Test Functions
# =============================================================================

def test_attention_variants():
    """Compare different attention time configurations."""
    print("="*70)
    print("ATTENTION VARIANT COMPARISON")
    print("="*70)
    print("\nTesting: Q/K time dependence + Euler Value projection")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    vocab_size = 64
    seq_len = 32
    batch_size = 32
    num_epochs = 300
    
    modes = ['static', 'query_time', 'both_time']
    results = {}
    
    for mode in modes:
        print(f"\n--- Mode: {mode} ---")
        
        model = EchoModelVariant(
            vocab_size=vocab_size,
            d_model=64,
            num_layers=2,
            num_neurons=32,
            n_heads=4,
            attention_mode=mode,
        ).to(device)
        
        optimizer = AdamW(model.parameters(), lr=1e-3)
        
        start = time.time()
        final_acc = 0.0
        
        # Retrieval task: [noise, MARKER, target, noise..., MARKER, ?]
        marker = vocab_size - 1
        
        for epoch in range(num_epochs):
            model.train()
            
            seq = torch.randint(0, vocab_size-2, (batch_size, seq_len), device=device)
            targets = torch.randint(0, vocab_size-2, (batch_size,), device=device)
            
            for i in range(batch_size):
                pos = torch.randint(2, seq_len//2, (1,)).item()
                seq[i, pos] = marker
                seq[i, pos+1] = targets[i]
            
            seq[:, -2] = marker
            
            logits = model(seq)
            loss = F.cross_entropy(logits[:, -1, :], targets)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if epoch % 100 == 99:
                model.eval()
                with torch.no_grad():
                    pred = logits[:, -1, :].argmax(dim=-1)
                    final_acc = (pred == targets).float().mean().item()
                print(f"  Epoch {epoch+1}: loss={loss.item():.4f}, acc={final_acc:.1%}")
        
        elapsed = time.time() - start
        
        # Check gradients
        model.train()
        seq = torch.randint(0, vocab_size, (2, 16), device=device)
        logits = model(seq)
        logits.sum().backward()
        
        grad_info = {}
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_info[name] = param.grad.norm().item()
        
        attn_grads = [v for k, v in grad_info.items() if 'attention' in k]
        res_grads = [v for k, v in grad_info.items() if 'resonant' in k]
        
        results[mode] = {
            'acc': final_acc,
            'time': elapsed,
            'attn_grad': sum(attn_grads) / len(attn_grads) if attn_grads else 0,
            'res_grad': sum(res_grads) / len(res_grads) if res_grads else 0,
        }
        
        model.zero_grad()
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"{'Mode':<15} {'Accuracy':<12} {'Time':<10} {'Attn Grad':<12} {'Res Grad':<12}")
    print("-"*60)
    for mode, r in results.items():
        print(f"{mode:<15} {r['acc']:.1%}        {r['time']:.1f}s    {r['attn_grad']:.4f}      {r['res_grad']:.4f}")
    
    return results


if __name__ == "__main__":
    test_attention_variants()

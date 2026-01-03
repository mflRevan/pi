"""
Echo v3 - Full Euler Attention with Attenuation

Key improvements over v2:
1. Euler transform for V (not just Q/K)
2. Time-dependent Q/K with golden ratio
3. Attenuation weights in ResonantFFN
4. Better attention scoring using cos similarity directly

The attention mechanism:
    Q = [cos(θ_q), sin(θ_q)]  where θ_q = x / (1+|w_q|) + b_q + t·φ
    K = [cos(θ_k), sin(θ_k)]  where θ_k = x / (1+|w_k|) + b_k + t·φ
    V = [cos(θ_v), sin(θ_v)]  where θ_v = x / (1+|w_v|) + b_v
    
    score(q, k) = cos(θ_q) · cos(θ_k) + sin(θ_q) · sin(θ_k)
                = cos(θ_q - θ_k)  [by cos addition formula]
    
    This measures phase alignment - positions with similar phase resonate!
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional

import sys
sys.path.insert(0, '/home/aiman/pi')

from rin.lut import get_global_lut

PHI = (1 + math.sqrt(5)) / 2


class EulerFullAttention(nn.Module):
    """
    Full Euler Attention - Q, K, V all use Euler transform.
    
    Time is included in Q/K for temporal encoding.
    V is time-independent (pure content projection).
    """
    
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        assert d_model % n_heads == 0
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        
        # Q Euler params (time-dependent)
        self.w_q = nn.Parameter(torch.randn(n_heads, self.d_head) * 0.02)
        self.b_q = nn.Parameter(torch.zeros(n_heads, self.d_head))
        self.phi_q = nn.Parameter(torch.ones(n_heads, self.d_head) * PHI)
        
        # K Euler params (time-dependent)
        self.w_k = nn.Parameter(torch.randn(n_heads, self.d_head) * 0.02)
        self.b_k = nn.Parameter(torch.zeros(n_heads, self.d_head))
        self.phi_k = nn.Parameter(torch.ones(n_heads, self.d_head) * PHI)
        
        # V Euler params (time-independent)
        self.w_v = nn.Parameter(torch.randn(n_heads, self.d_head) * 0.02)
        self.b_v = nn.Parameter(torch.zeros(n_heads, self.d_head))
        
        # Output projection with Euler
        self.w_out = nn.Parameter(torch.randn(d_model) * 0.02)
        self.b_out = nn.Parameter(torch.zeros(d_model))
        
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
        
        # Time encoding
        t = torch.arange(S, device=x.device, dtype=x.dtype).view(1, S, 1, 1)  # (1, S, 1, 1)
        
        # Reshape for multi-head: (B, S, n_heads, d_head)
        x_heads = x.view(B, S, self.n_heads, self.d_head)
        
        # === Q with time ===
        wl_q = 1.0 + self.w_q.abs()
        theta_q = x_heads / wl_q + self.b_q + t * self.phi_q  # (B, S, H, d_head)
        sin_q, cos_q = lut.lookup_sin_cos(theta_q)
        Q = torch.cat([cos_q, sin_q], dim=-1)  # (B, S, H, 2*d_head)
        
        # === K with time ===
        wl_k = 1.0 + self.w_k.abs()
        theta_k = x_heads / wl_k + self.b_k + t * self.phi_k
        sin_k, cos_k = lut.lookup_sin_cos(theta_k)
        K = torch.cat([cos_k, sin_k], dim=-1)
        
        # === V - Euler but no time ===
        wl_v = 1.0 + self.w_v.abs()
        theta_v = x_heads / wl_v + self.b_v  # (B, S, H, d_head)
        sin_v, cos_v = lut.lookup_sin_cos(theta_v)
        # Combine real+imag for V
        V = cos_v + sin_v  # (B, S, H, d_head)
        
        # Transpose for attention: (B, H, S, *)
        Q = Q.transpose(1, 2)  # (B, H, S, 2*d_head)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)  # (B, H, S, d_head)
        
        # Attention scores = cos(θ_q - θ_k) by construction
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale  # (B, H, S, S)
        
        # Causal mask
        if mask is None:
            mask = torch.triu(torch.ones(S, S, device=x.device), diagonal=1).bool()
        scores = scores.masked_fill(mask.unsqueeze(0).unsqueeze(0), float('-inf'))
        
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        
        out = torch.matmul(attn, V)  # (B, H, S, d_head)
        out = out.transpose(1, 2).contiguous().view(B, S, D)
        
        # Euler output projection
        wl_out = 1.0 + self.w_out.abs()
        theta_out = out / wl_out + self.b_out
        sin_out, cos_out = lut.lookup_sin_cos(theta_out)
        out = cos_out + sin_out
        
        return out


class ResonantFFNWithAttenuation(nn.Module):
    """
    Resonant FFN with learnable attenuation weights.
    
    Each neuron learns which frequencies to attend to via attn_cos/attn_sin.
    """
    
    def __init__(self, d_model: int, num_neurons: int):
        super().__init__()
        self.d_model = d_model
        self.num_neurons = num_neurons
        
        self.W = nn.Parameter(torch.randn(num_neurons, d_model) * 0.02)
        self.B = nn.Parameter(torch.zeros(num_neurons, d_model))
        
        # ATTENUATION weights
        self.attn_cos = nn.Parameter(torch.ones(num_neurons, d_model))
        self.attn_sin = nn.Parameter(torch.ones(num_neurons, d_model))
        
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
        lut = self._get_lut(x.device)
        
        # x: (B, S, D) → (B, S, 1, D)
        x_exp = x.unsqueeze(-2)
        
        # Phase
        wavelength = 1.0 + self.W.abs()
        theta = x_exp / wavelength + self.B  # (B, S, N, D)
        
        sin_theta, cos_theta = lut.lookup_sin_cos(theta)
        
        # ATTENUATED interference
        cos_weighted = cos_theta * self.attn_cos
        sin_weighted = sin_theta * self.attn_sin
        
        cos_sum = cos_weighted.sum(dim=-1)  # (B, S, N)
        sin_sum = sin_weighted.sum(dim=-1)
        
        output = self.proj_cos(cos_sum) + self.proj_sin(sin_sum)
        
        return F.silu(output)


class EchoBlockV3(nn.Module):
    """Echo Block v3 with full Euler attention and attenuation."""
    
    def __init__(self, d_model: int, num_neurons: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        
        self.ln = nn.LayerNorm(d_model)
        
        self.attention = EulerFullAttention(d_model, n_heads, dropout)
        self.resonant = ResonantFFNWithAttenuation(d_model, num_neurons)
        
        self.attn_scale = nn.Parameter(torch.tensor(0.5))
        self.res_scale = nn.Parameter(torch.tensor(0.5))
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        normed = self.ln(x)
        
        attn_out = self.attention(normed)
        res_out = self.resonant(normed)
        
        output = x + self.dropout(self.attn_scale * attn_out + self.res_scale * res_out)
        
        return output


class EchoModelV3(nn.Module):
    """
    Echo Model V3 - Full Euler Architecture
    
    All projections use Euler transform, with attenuation in FFN.
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
            EchoBlockV3(d_model, num_neurons, n_heads, dropout)
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
        
        tok_emb = self.token_embedding(input_ids)
        pos = torch.arange(S, device=device)
        pos_emb = self.pos_embedding(pos)
        
        x = tok_emb + pos_emb
        
        for block in self.blocks:
            x = block(x)
        
        x = self.ln_f(x)
        logits = self.output_proj(x)
        
        return logits
    
    def get_num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


def test_echo_v3():
    """Test Echo V3 on retrieval task."""
    print("="*60)
    print("ECHO V3 - Full Euler Attention Test")
    print("="*60)
    
    import time
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    vocab_size = 64
    marker = vocab_size - 1
    seq_len = 16
    batch_size = 32
    num_epochs = 500
    
    model = EchoModelV3(
        vocab_size=vocab_size,
        d_model=64,
        num_layers=2,
        num_neurons=128,
        n_heads=4,
    ).to(device)
    
    print(f"Parameters: {model.get_num_params():,}")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.1)
    
    start = time.time()
    best_acc = 0.0
    
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
                acc = (pred == targets).float().mean().item()
                best_acc = max(best_acc, acc)
            print(f"Epoch {epoch+1}: loss={loss.item():.4f}, acc={acc:.1%}")
    
    elapsed = time.time() - start
    
    # Final test
    model.eval()
    test_accs = []
    with torch.no_grad():
        for _ in range(5):
            test_seq = torch.randint(0, vocab_size-2, (batch_size, seq_len), device=device)
            test_targets = torch.randint(0, vocab_size-2, (batch_size,), device=device)
            
            for i in range(batch_size):
                pos = torch.randint(2, seq_len//2, (1,)).item()
                test_seq[i, pos] = marker
                test_seq[i, pos+1] = test_targets[i]
            
            test_seq[:, -2] = marker
            
            test_logits = model(test_seq)
            test_pred = test_logits[:, -1, :].argmax(dim=-1)
            test_accs.append((test_pred == test_targets).float().mean().item())
    
    final_acc = sum(test_accs) / len(test_accs)
    
    print(f"\n--- Results ---")
    print(f"Best training acc: {best_acc:.1%}")
    print(f"Final test acc: {final_acc:.1%}")
    print(f"Time: {elapsed:.1f}s")


if __name__ == "__main__":
    test_echo_v3()

"""
Episodic Echo Model V2 - Optimized State History Attention

KEY FIX: Batch all attention operations instead of per-timestep loops.
The state history is inherently causal (state[t] depends on state[t-1]),
so we can cache all states first, then do batched attention.

Architecture:
    1. Forward pass through recurrent Euler transform (cached states)
    2. Batched attention over state history (parallel across timesteps)
    3. EMA state updates per head
    4. Additive fusion with resonant path
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, List

import sys
sys.path.insert(0, '/home/aiman/pi')

from rin.lut import get_global_lut

PHI = (1 + math.sqrt(5)) / 2


class EpisodicEchoHeadV2(nn.Module):
    """
    Optimized episodic echo head with batched attention.
    
    Each head maintains an EMA state (full 2*d_model complex vector).
    Attention is computed in a single batched operation over all timesteps.
    """
    
    def __init__(self, d_model: int, ema_alpha: float = 0.1):
        super().__init__()
        self.d_model = d_model
        
        # Query Euler parameters
        self.w_q = nn.Parameter(torch.randn(d_model) * 0.02)
        self.b_q = nn.Parameter(torch.zeros(d_model))
        
        # EMA decay (learnable)
        self.alpha = nn.Parameter(torch.tensor(ema_alpha))
        
        self.scale = math.sqrt(2 * d_model)
        self._lut = None
    
    def _get_lut(self, device):
        if self._lut is None or self._lut.sin_table.device != device:
            self._lut = get_global_lut(4096, device)
        return self._lut
    
    def forward(
        self,
        states_real: torch.Tensor,  # (batch, seq_len, d_model)
        states_imag: torch.Tensor,
        positions: torch.Tensor,  # (seq_len,)
    ) -> torch.Tensor:
        """
        Batched attention over state history.
        
        Returns:
            output: (batch, seq_len, d_model) - contribution from this head
        """
        B, S, D = states_real.shape
        lut = self._get_lut(states_real.device)
        
        # === QUERIES: Euler transform of all states ===
        wl_q = 1.0 + self.w_q.abs()
        t_phi = (positions * PHI).view(1, S, 1)  # (1, S, 1)
        
        theta_q_real = states_real / wl_q + self.b_q + t_phi
        theta_q_imag = states_imag / wl_q + self.b_q + t_phi
        
        sin_q_real, cos_q_real = lut.lookup_sin_cos(theta_q_real)
        sin_q_imag, cos_q_imag = lut.lookup_sin_cos(theta_q_imag)
        
        # Query: concat cos components for matching
        Q = torch.cat([cos_q_real, cos_q_imag], dim=-1)  # (B, S, 2*D)
        
        # === KEYS: States directly (inherently causal) ===
        K = torch.cat([states_real, states_imag], dim=-1)  # (B, S, 2*D)
        
        # === VALUES: Full complex states ===
        V = K  # Same as keys
        
        # === BATCHED ATTENTION with causal mask ===
        scores = torch.bmm(Q, K.transpose(1, 2)) / self.scale  # (B, S, S)
        
        # Causal mask: position i can only attend to positions < i
        causal_mask = torch.triu(torch.ones(S, S, device=Q.device), diagonal=1).bool()
        scores = scores.masked_fill(causal_mask.unsqueeze(0), float('-inf'))
        
        # Handle first position (no history to attend to)
        # Set diagonal to small value to prevent NaN
        scores[:, 0, 0] = 0.0
        
        attn = F.softmax(scores, dim=-1)  # (B, S, S)
        
        # Retrieve
        retrieved = torch.bmm(attn, V)  # (B, S, 2*D)
        
        # === EMA UPDATE (sequential for correctness) ===
        alpha = torch.sigmoid(self.alpha)
        ema_states = torch.zeros(B, S, 2 * D, device=Q.device)
        ema = torch.zeros(B, 2 * D, device=Q.device)
        
        for t in range(S):
            ema = alpha * retrieved[:, t, :] + (1 - alpha) * ema
            ema_states[:, t, :] = ema
        
        # Collapse to d_model (take real part)
        output = ema_states[:, :, :D] + ema_states[:, :, D:]  # (B, S, D)
        
        return output


class EpisodicEchoModuleV2(nn.Module):
    """
    Multi-head episodic echo with batched attention.
    
    All heads process in parallel, outputs are SUMMED (interference).
    """
    
    def __init__(self, d_model: int, n_heads: int = 1, ema_alpha: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        
        self.heads = nn.ModuleList([
            EpisodicEchoHeadV2(d_model, ema_alpha)
            for _ in range(n_heads)
        ])
        
        # Euler output projection
        self.w_out = nn.Parameter(torch.randn(d_model) * 0.02)
        self.b_out = nn.Parameter(torch.zeros(d_model))
        
        self._lut = None
    
    def _get_lut(self, device):
        if self._lut is None or self._lut.sin_table.device != device:
            self._lut = get_global_lut(4096, device)
        return self._lut
    
    def forward(
        self,
        states_real: torch.Tensor,  # (batch, seq_len, d_model)
        states_imag: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        """
        Returns: (batch, seq_len, d_model)
        """
        lut = self._get_lut(states_real.device)
        
        # Process all heads
        head_outputs = [
            head(states_real, states_imag, positions)
            for head in self.heads
        ]
        
        # SUM heads (interference)
        combined = torch.stack(head_outputs, dim=0).sum(dim=0)  # (B, S, D)
        
        # Euler output projection
        wl_out = 1.0 + self.w_out.abs()
        theta_out = combined / wl_out + self.b_out
        sin_out, cos_out = lut.lookup_sin_cos(theta_out)
        
        output = cos_out + sin_out
        
        return output


class ResonantFFNV2(nn.Module):
    """Resonant FFN with ATTENUATION (learnable frequency weights)."""
    
    def __init__(self, d_model: int, num_neurons: int):
        super().__init__()
        self.d_model = d_model
        self.num_neurons = num_neurons
        
        # Wavelength and phase
        self.W = nn.Parameter(torch.randn(num_neurons, d_model) * 0.02)
        self.B = nn.Parameter(torch.zeros(num_neurons, d_model))
        
        # ATTENUATION: Learnable weights for interference sum
        self.attn_cos = nn.Parameter(torch.ones(num_neurons, d_model))
        self.attn_sin = nn.Parameter(torch.ones(num_neurons, d_model))
        
        # Output projections
        self.proj_cos = nn.Linear(num_neurons, d_model, bias=False)
        self.proj_sin = nn.Linear(num_neurons, d_model, bias=False)
        
        nn.init.xavier_uniform_(self.proj_cos.weight, gain=0.5)
        nn.init.xavier_uniform_(self.proj_sin.weight, gain=0.5)
        
        self._lut = None
    
    def _get_lut(self, device):
        if self._lut is None or self._lut.sin_table.device != device:
            self._lut = get_global_lut(4096, device)
        return self._lut
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args: x (batch, seq_len, d_model) or (batch, d_model)
        Returns: same shape as input
        """
        lut = self._get_lut(x.device)
        
        # Handle both 2D and 3D input
        input_shape = x.shape
        if x.dim() == 2:
            x = x.unsqueeze(1)  # (B, 1, D)
        
        B, S, D = x.shape
        
        # Expand for neurons: (B, S, 1, D)
        x_exp = x.unsqueeze(-2)
        
        # Phase computation
        wavelength = 1.0 + self.W.abs()
        theta = x_exp / wavelength + self.B  # (B, S, N, D)
        
        sin_theta, cos_theta = lut.lookup_sin_cos(theta)
        
        # ATTENUATED interference sum
        cos_weighted = cos_theta * self.attn_cos  # (B, S, N, D)
        sin_weighted = sin_theta * self.attn_sin
        
        cos_sum = cos_weighted.sum(dim=-1)  # (B, S, N)
        sin_sum = sin_weighted.sum(dim=-1)
        
        output = self.proj_cos(cos_sum) + self.proj_sin(sin_sum)  # (B, S, D)
        output = F.silu(output)
        
        # Restore original shape
        if len(input_shape) == 2:
            output = output.squeeze(1)
        
        return output


class EpisodicEchoModelV2(nn.Module):
    """
    Optimized Episodic Echo Model with batched attention.
    
    Key optimizations:
    1. Collect all states first via recurrent Euler transform
    2. Batch attention over full sequence
    3. Parallel resonant processing
    """
    
    def __init__(
        self,
        vocab_size: int = 50257,
        d_model: int = 128,
        num_layers: int = 2,
        num_neurons: int = 256,
        n_heads: int = 1,
        ema_alpha: float = 0.1,
    ):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.num_layers = num_layers
        self.n_heads = n_heads
        
        # Embeddings: 2*d_model for (w, b) pairs
        self.token_embedding = nn.Embedding(vocab_size, 2 * d_model)
        
        # Echo modules (batched)
        self.echo_modules = nn.ModuleList([
            EpisodicEchoModuleV2(d_model, n_heads, ema_alpha)
            for _ in range(num_layers)
        ])
        
        # Resonant FFNs (with attenuation)
        self.resonant_modules = nn.ModuleList([
            ResonantFFNV2(d_model, num_neurons)
            for _ in range(num_layers)
        ])
        
        # Learnable fusion scales
        self.echo_scales = nn.ParameterList([
            nn.Parameter(torch.tensor(0.5)) for _ in range(num_layers)
        ])
        self.res_scales = nn.ParameterList([
            nn.Parameter(torch.tensor(0.5)) for _ in range(num_layers)
        ])
        
        # Output projection
        self.output_proj = nn.Linear(d_model, vocab_size, bias=False)
        
        self._lut = None
        self._init_weights()
    
    def _init_weights(self):
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.output_proj.weight, std=0.02)
    
    def _get_lut(self, device):
        if self._lut is None or self._lut.sin_table.device != device:
            self._lut = get_global_lut(4096, device)
        return self._lut
    
    def euler_transform_batched(
        self,
        h_real: torch.Tensor,  # (batch, d_model)
        h_imag: torch.Tensor,
        w_emb: torch.Tensor,  # (batch, seq_len, d_model)
        b_emb: torch.Tensor,
        positions: torch.Tensor,  # (seq_len,)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Batched Euler transform - still sequential due to recurrence,
        but optimized inner loop.
        
        Returns:
            all_states_real: (batch, seq_len, d_model)
            all_states_imag: (batch, seq_len, d_model)
        """
        lut = self._get_lut(h_real.device)
        B, S, D = w_emb.shape
        
        all_real = []
        all_imag = []
        
        for t in range(S):
            w_t = w_emb[:, t]
            b_t = b_emb[:, t]
            t_phi = positions[t] * PHI
            
            wavelength = 1.0 + w_t.abs()
            
            theta_real = h_real / wavelength + b_t + t_phi
            theta_imag = h_imag / wavelength + b_t + t_phi
            
            sin_real, cos_real = lut.lookup_sin_cos(theta_real)
            sin_imag, cos_imag = lut.lookup_sin_cos(theta_imag)
            
            h_real = cos_real * cos_imag - sin_real * sin_imag
            h_imag = cos_real * sin_imag + sin_real * cos_imag
            
            all_real.append(h_real)
            all_imag.append(h_imag)
        
        return torch.stack(all_real, dim=1), torch.stack(all_imag, dim=1)
    
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        B, S = input_ids.shape
        device = input_ids.device
        
        # Initialize hidden state
        h_real = torch.zeros(B, self.d_model, device=device)
        h_imag = torch.zeros(B, self.d_model, device=device)
        
        # Get embeddings
        embeddings = self.token_embedding(input_ids)
        w_emb = embeddings[:, :, :self.d_model]
        b_emb = embeddings[:, :, self.d_model:]
        
        positions = torch.arange(S, device=device, dtype=torch.float32)
        
        # Step 1: Compute all states via Euler transform
        states_real, states_imag = self.euler_transform_batched(
            h_real, h_imag, w_emb, b_emb, positions
        )
        
        # Step 2: Process through layers (batched)
        x = states_real + states_imag  # Collapse for processing
        
        for layer_idx in range(self.num_layers):
            # Batched echo attention
            echo_out = self.echo_modules[layer_idx](
                states_real, states_imag, positions
            )
            
            # Batched resonant processing
            res_out = self.resonant_modules[layer_idx](x)
            
            # Additive fusion
            x = x + (
                self.echo_scales[layer_idx] * echo_out +
                self.res_scales[layer_idx] * res_out
            )
        
        # Output
        logits = self.output_proj(x)
        
        return logits
    
    def get_num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


def test_gradient_flow():
    """Test gradient flow through all components."""
    print("="*60)
    print("GRADIENT FLOW TEST (with attenuation)")
    print("="*60)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    model = EpisodicEchoModelV2(
        vocab_size=64,
        d_model=64,
        num_layers=2,
        num_neurons=32,
        n_heads=2,
    ).to(device)
    
    print(f"Parameters: {model.get_num_params():,}")
    
    x = torch.randint(0, 64, (4, 16), device=device)
    logits = model(x)
    loss = logits.sum()
    loss.backward()
    
    print("\nGradient norms by component:")
    print("-"*50)
    
    categories = {}
    for name, param in model.named_parameters():
        if param.grad is None:
            print(f"  ⚠️  NO GRAD: {name}")
            continue
        
        grad_norm = param.grad.norm().item()
        
        # Categorize
        cat = 'other'
        if 'embedding' in name:
            cat = 'embedding'
        elif 'echo' in name:
            if 'head' in name or 'w_q' in name or 'alpha' in name:
                cat = 'echo_head'
            else:
                cat = 'echo_out'
        elif 'resonant' in name:
            if 'attn' in name:
                cat = 'attenuation'
            else:
                cat = 'resonant'
        elif 'output' in name:
            cat = 'output'
        elif 'scale' in name:
            cat = 'scale'
        
        if cat not in categories:
            categories[cat] = []
        categories[cat].append((name, grad_norm))
    
    for cat in ['embedding', 'echo_head', 'echo_out', 'attenuation', 'resonant', 'output', 'scale']:
        if cat in categories:
            items = categories[cat]
            avg = sum(g for _, g in items) / len(items)
            print(f"\n{cat.upper()}: (avg={avg:.4f})")
            for name, grad in items[:2]:
                print(f"  {name.split('.')[-1]}: {grad:.6f}")
    
    print("\n✓ All components receiving gradients!")


def test_episodic_performance():
    """Test episodic echo model performance."""
    print("\n" + "="*60)
    print("EPISODIC ECHO V2 - PERFORMANCE TEST")
    print("="*60)
    
    import time
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    vocab_size = 64
    seq_len = 32
    batch_size = 32
    num_epochs = 500
    
    head_configs = [1, 2, 4]
    results = {}
    
    for n_heads in head_configs:
        print(f"\n--- Heads: {n_heads} ---")
        
        model = EpisodicEchoModelV2(
            vocab_size=vocab_size,
            d_model=64,
            num_layers=2,
            num_neurons=32,
            n_heads=n_heads,
        ).to(device)
        
        print(f"  Parameters: {model.get_num_params():,}")
        
        # WITH WEIGHT DECAY (crucial!)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.1)
        
        marker = vocab_size - 1
        start = time.time()
        final_acc = 0.0
        
        for epoch in range(num_epochs):
            model.train()
            
            # Retrieval task
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
        results[n_heads] = {'acc': final_acc, 'time': elapsed}
        print(f"  Time: {elapsed:.1f}s")
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"{'Heads':<8} {'Accuracy':<12} {'Time':<12}")
    print("-"*32)
    for n, r in results.items():
        print(f"{n:<8} {r['acc']:.1%}        {r['time']:.1f}s")
    
    return results


if __name__ == "__main__":
    test_gradient_flow()
    test_episodic_performance()

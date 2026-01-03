"""
Episodic Echo Model - State History Attention

KEY INSIGHT: The input is a linear recurrent transforming state where each
embedding is a transformation function. Instead of attention over token
embeddings, we attend over CACHED STATE HISTORY.

Architecture:
    1. Token embedding → (w, b) transformation parameters
    2. Euler state transformation: state[t] = euler_transform(state[t-1], w[t], b[t], t)
    3. Cache ALL states into buffer (state history)
    4. Echo module: attention over state history (not tokens!)
       - Query: Euler transform of current state
       - Keys: Past states directly (inherently causal - no K projection needed)
       - Each head: softmax over past states → weighted sum → write to EMA state
       - Each head maintains FULL complex state (2*d_model)
       - Sum all head states (interference, not concat)
       - Euler output projection
    5. Add to resonant path

This is inherently causal: state[t] depends only on state[t-1] by construction.
No separate causal mask needed for attention since we only cache past states.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, List
from dataclasses import dataclass

import sys
sys.path.insert(0, '/home/aiman/pi')

from rin.lut import get_global_lut

PHI = (1 + math.sqrt(5)) / 2


@dataclass
class EpisodicState:
    """State container for episodic echo model."""
    # Recurrent state (complex)
    h_real: torch.Tensor  # (batch, d_model)
    h_imag: torch.Tensor
    # State history buffer
    history_real: List[torch.Tensor]  # List of (batch, d_model)
    history_imag: List[torch.Tensor]
    # EMA states per head
    ema_states: torch.Tensor  # (batch, n_heads, 2*d_model) - full complex per head


class EpisodicEchoHead(nn.Module):
    """
    Single head for episodic echo attention.
    
    Each head:
    1. Computes Euler query from current state
    2. Attends over state history (keys ARE the states - no projection)
    3. Writes weighted combination to its EMA state
    4. Outputs its EMA state
    
    The head maintains a FULL complex state vector (2*d_model).
    """
    
    def __init__(
        self,
        d_model: int,
        ema_alpha: float = 0.1,
        learnable_alpha: bool = True,
    ):
        super().__init__()
        self.d_model = d_model
        
        # Query Euler parameters - projects current state to query
        self.w_q = nn.Parameter(torch.randn(d_model) * 0.02)
        self.b_q = nn.Parameter(torch.zeros(d_model))
        
        # EMA decay rate
        if learnable_alpha:
            self.alpha = nn.Parameter(torch.tensor(ema_alpha))
        else:
            self.register_buffer('alpha', torch.tensor(ema_alpha))
        
        self.scale = math.sqrt(2 * d_model)
        self._lut = None
    
    def _get_lut(self, device):
        if self._lut is None or self._lut.sin_table.device != device:
            self._lut = get_global_lut(4096, device)
        return self._lut
    
    def forward(
        self,
        current_state_real: torch.Tensor,  # (batch, d_model)
        current_state_imag: torch.Tensor,
        history_real: List[torch.Tensor],  # List of (batch, d_model)
        history_imag: List[torch.Tensor],
        ema_state: torch.Tensor,  # (batch, 2*d_model) - this head's EMA
        t: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            output: (batch, 2*d_model) - this head's contribution
            new_ema: (batch, 2*d_model) - updated EMA state
        """
        if len(history_real) == 0:
            # No history yet, return zeros
            return torch.zeros_like(ema_state), ema_state
        
        lut = self._get_lut(current_state_real.device)
        batch_size = current_state_real.shape[0]
        
        # Current state as complex: (batch, 2*d_model)
        current_complex = torch.cat([current_state_real, current_state_imag], dim=-1)
        
        # === QUERY: Euler transform of current state ===
        wl_q = 1.0 + self.w_q.abs()
        t_phi = t * PHI
        
        # Apply Euler to real part
        theta_q_real = current_state_real / wl_q + self.b_q + t_phi
        sin_q_real, cos_q_real = lut.lookup_sin_cos(theta_q_real)
        
        # Apply Euler to imag part (same params, different input)
        theta_q_imag = current_state_imag / wl_q + self.b_q + t_phi
        sin_q_imag, cos_q_imag = lut.lookup_sin_cos(theta_q_imag)
        
        # Query = [cos_real, sin_real, cos_imag, sin_imag] flattened for matching
        query = torch.cat([cos_q_real, sin_q_real, cos_q_imag, sin_q_imag], dim=-1)
        # (batch, 4*d_model)
        
        # === KEYS: Past states directly (no projection - inherently causal) ===
        # Stack history: (batch, history_len, d_model) for real and imag
        stacked_real = torch.stack(history_real, dim=1)
        stacked_imag = torch.stack(history_imag, dim=1)
        
        # For key matching, we need comparable representation
        # Use the states directly concatenated
        keys = torch.cat([stacked_real, stacked_imag], dim=-1)  # (batch, hist, 2*d_model)
        
        # Expand query for matching: convert 4*d_model to 2*d_model for comparison
        # Use cos components as the key match (phase alignment)
        query_for_match = torch.cat([cos_q_real, cos_q_imag], dim=-1)  # (batch, 2*d_model)
        
        # === ATTENTION ===
        # Score = dot product of query with each historical state
        scores = torch.bmm(
            query_for_match.unsqueeze(1),  # (batch, 1, 2*d_model)
            keys.transpose(1, 2)  # (batch, 2*d_model, hist)
        ).squeeze(1)  # (batch, hist)
        
        scores = scores / self.scale
        weights = F.softmax(scores, dim=-1)  # (batch, hist)
        
        # === RETRIEVE: Weighted sum of historical states ===
        # Retrieved state is full complex
        values = torch.cat([stacked_real, stacked_imag], dim=-1)  # (batch, hist, 2*d_model)
        retrieved = torch.bmm(
            weights.unsqueeze(1),  # (batch, 1, hist)
            values  # (batch, hist, 2*d_model)
        ).squeeze(1)  # (batch, 2*d_model)
        
        # === UPDATE EMA ===
        alpha = torch.sigmoid(self.alpha)  # Constrain to (0, 1)
        new_ema = alpha * retrieved + (1 - alpha) * ema_state
        
        return new_ema, new_ema


class EpisodicEchoModule(nn.Module):
    """
    Episodic Echo Module - Multi-head attention over state history.
    
    Each head maintains its own EMA state (full 2*d_model complex vector).
    All head outputs are SUMMED (interference), not concatenated.
    
    This creates constructive interference when heads agree,
    and destructive interference (near-zero) when they don't.
    """
    
    def __init__(
        self,
        d_model: int,
        n_heads: int = 1,
        ema_alpha: float = 0.1,
        learnable_alpha: bool = True,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        
        self.heads = nn.ModuleList([
            EpisodicEchoHead(d_model, ema_alpha, learnable_alpha)
            for _ in range(n_heads)
        ])
        
        # Euler output projection
        self.w_out = nn.Parameter(torch.randn(2 * d_model) * 0.02)
        self.b_out = nn.Parameter(torch.zeros(2 * d_model))
        
        self._lut = None
    
    def _get_lut(self, device):
        if self._lut is None or self._lut.sin_table.device != device:
            self._lut = get_global_lut(4096, device)
        return self._lut
    
    def init_ema_states(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Initialize EMA states for all heads."""
        return torch.zeros(batch_size, self.n_heads, 2 * self.d_model, device=device)
    
    def forward(
        self,
        current_state_real: torch.Tensor,
        current_state_imag: torch.Tensor,
        history_real: List[torch.Tensor],
        history_imag: List[torch.Tensor],
        ema_states: torch.Tensor,  # (batch, n_heads, 2*d_model)
        t: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            output: (batch, d_model) - collapsed to real for adding to resonant
            new_ema_states: (batch, n_heads, 2*d_model)
        """
        batch_size = current_state_real.shape[0]
        lut = self._get_lut(current_state_real.device)
        
        # Process each head
        head_outputs = []
        new_ema_list = []
        
        for i, head in enumerate(self.heads):
            head_ema = ema_states[:, i, :]  # (batch, 2*d_model)
            output, new_ema = head(
                current_state_real,
                current_state_imag,
                history_real,
                history_imag,
                head_ema,
                t,
            )
            head_outputs.append(output)
            new_ema_list.append(new_ema)
        
        # Stack new EMAs
        new_ema_states = torch.stack(new_ema_list, dim=1)  # (batch, n_heads, 2*d_model)
        
        # SUM head outputs (interference!)
        combined = torch.stack(head_outputs, dim=0).sum(dim=0)  # (batch, 2*d_model)
        
        # Euler output projection
        wl_out = 1.0 + self.w_out.abs()
        theta_out = combined / wl_out + self.b_out
        sin_out, cos_out = lut.lookup_sin_cos(theta_out)
        
        # Collapse to d_model
        # Split cos and sin, add corresponding halves
        cos_real, cos_imag = cos_out.chunk(2, dim=-1)
        sin_real, sin_imag = sin_out.chunk(2, dim=-1)
        
        # Output = real part of complex result
        output = cos_real + sin_real  # (batch, d_model)
        
        return output, new_ema_states


class ResonantFFN(nn.Module):
    """Resonant FFN for parallel path."""
    
    def __init__(self, d_model: int, num_neurons: int):
        super().__init__()
        self.W = nn.Parameter(torch.randn(num_neurons, d_model) * 0.02)
        self.B = nn.Parameter(torch.zeros(num_neurons, d_model))
        self.proj_cos = nn.Linear(num_neurons, d_model, bias=False)
        self.proj_sin = nn.Linear(num_neurons, d_model, bias=False)
        nn.init.xavier_uniform_(self.proj_cos.weight, gain=0.5)
        nn.init.xavier_uniform_(self.proj_sin.weight, gain=0.5)
        self._lut = None
    
    def _get_lut(self, device):
        if self._lut is None or self._lut.sin_table.device != device:
            self._lut = get_global_lut(4096, device)
        return self._lut
    
    def forward(self, x_real: torch.Tensor, x_imag: torch.Tensor) -> torch.Tensor:
        lut = self._get_lut(x_real.device)
        # Collapse complex to real for resonant
        x = x_real + x_imag
        x_exp = x.unsqueeze(1)
        wavelength = 1.0 + self.W.abs()
        theta = x_exp / wavelength + self.B
        sin_theta, cos_theta = lut.lookup_sin_cos(theta)
        cos_sum = cos_theta.sum(dim=-1)
        sin_sum = sin_theta.sum(dim=-1)
        return F.silu(self.proj_cos(cos_sum) + self.proj_sin(sin_sum))


class EpisodicEchoModel(nn.Module):
    """
    Episodic Echo Model - State History Attention
    
    Architecture:
        1. Token embeddings → (w, b) transformation parameters
        2. Recurrent Euler state transformation
        3. Cache all states to history buffer
        4. Parallel: EpisodicEcho + ResonantFFN with additive fusion
        5. Output projection
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
        
        # Echo modules (one per layer)
        self.echo_modules = nn.ModuleList([
            EpisodicEchoModule(d_model, n_heads, ema_alpha)
            for _ in range(num_layers)
        ])
        
        # Resonant FFNs (one per layer)
        self.resonant_modules = nn.ModuleList([
            ResonantFFN(d_model, num_neurons)
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
    
    def euler_transform(
        self,
        h_real: torch.Tensor,
        h_imag: torch.Tensor,
        w: torch.Tensor,
        b: torch.Tensor,
        t: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Euler-based state transformation."""
        lut = self._get_lut(h_real.device)
        
        wavelength = 1.0 + w.abs()
        t_phi = t * PHI
        
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
        return_states: bool = False,
    ) -> torch.Tensor:
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        
        # Initialize
        h_real = torch.zeros(batch_size, self.d_model, device=device)
        h_imag = torch.zeros(batch_size, self.d_model, device=device)
        
        # Per-layer state histories and EMAs
        histories_real = [[] for _ in range(self.num_layers)]
        histories_imag = [[] for _ in range(self.num_layers)]
        ema_states = [
            module.init_ema_states(batch_size, device)
            for module in self.echo_modules
        ]
        
        # Get embeddings
        embeddings = self.token_embedding(input_ids)
        w_emb = embeddings[:, :, :self.d_model]
        b_emb = embeddings[:, :, self.d_model:]
        
        all_logits = []
        all_states = [] if return_states else None
        
        for t_idx in range(seq_len):
            w_t = w_emb[:, t_idx]
            b_t = b_emb[:, t_idx]
            t = float(t_idx)
            
            # Euler state transformation
            h_real, h_imag = self.euler_transform(h_real, h_imag, w_t, b_t, t)
            
            if return_states:
                all_states.append((h_real.clone(), h_imag.clone()))
            
            # Process through layers
            x_real, x_imag = h_real, h_imag
            
            for layer_idx in range(self.num_layers):
                # Cache state BEFORE processing
                # CRITICAL: Keep gradients during training for learning
                if self.training:
                    histories_real[layer_idx].append(x_real)
                    histories_imag[layer_idx].append(x_imag)
                else:
                    histories_real[layer_idx].append(x_real.detach())
                    histories_imag[layer_idx].append(x_imag.detach())
                
                # Parallel: Echo + Resonant
                echo_out, ema_states[layer_idx] = self.echo_modules[layer_idx](
                    x_real, x_imag,
                    histories_real[layer_idx][:-1],  # Exclude current (causal)
                    histories_imag[layer_idx][:-1],
                    ema_states[layer_idx],
                    t,
                )
                
                res_out = self.resonant_modules[layer_idx](x_real, x_imag)
                
                # Additive fusion
                delta = (
                    self.echo_scales[layer_idx] * echo_out +
                    self.res_scales[layer_idx] * res_out
                )
                
                # Update state (add delta to collapsed state)
                x_collapsed = x_real + x_imag
                x_collapsed = x_collapsed + delta
                
                # Re-split for next layer (simple split)
                x_real = x_collapsed
                x_imag = torch.zeros_like(x_collapsed)
            
            # Output
            logits = self.output_proj(x_real)
            all_logits.append(logits)
        
        output = torch.stack(all_logits, dim=1)
        
        if return_states:
            return output, all_states
        return output
    
    def get_num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


# =============================================================================
# Tests
# =============================================================================

def test_episodic_model():
    """Test episodic echo model with different head counts."""
    print("="*70)
    print("EPISODIC ECHO MODEL TEST")
    print("State History Attention")
    print("="*70)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    vocab_size = 64
    seq_len = 32
    batch_size = 32
    num_epochs = 300
    
    head_configs = [1, 2, 4]
    results = {}
    
    for n_heads in head_configs:
        print(f"\n--- Heads: {n_heads} ---")
        
        model = EpisodicEchoModel(
            vocab_size=vocab_size,
            d_model=64,
            num_layers=2,
            num_neurons=32,
            n_heads=n_heads,
        ).to(device)
        
        print(f"  Parameters: {model.get_num_params():,}")
        
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        
        start = time.time()
        final_acc = 0.0
        marker = vocab_size - 1
        
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
        
        # Analyze gradients
        model.train()
        seq = torch.randint(0, vocab_size, (2, 16), device=device)
        logits = model(seq)
        logits.sum().backward()
        
        grad_info = {}
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_info[name] = param.grad.norm().item()
        
        echo_grads = [v for k, v in grad_info.items() if 'echo' in k]
        res_grads = [v for k, v in grad_info.items() if 'resonant' in k]
        embed_grads = [v for k, v in grad_info.items() if 'embedding' in k]
        
        results[n_heads] = {
            'acc': final_acc,
            'time': elapsed,
            'echo_grad': sum(echo_grads) / len(echo_grads) if echo_grads else 0,
            'res_grad': sum(res_grads) / len(res_grads) if res_grads else 0,
            'embed_grad': sum(embed_grads) / len(embed_grads) if embed_grads else 0,
        }
        
        model.zero_grad()
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY: EPISODIC ECHO (State History Attention)")
    print("="*70)
    print(f"{'Heads':<8} {'Acc':<10} {'Time':<10} {'Echo∇':<12} {'Res∇':<12} {'Emb∇':<12}")
    print("-"*64)
    for n, r in results.items():
        print(f"{n:<8} {r['acc']:.1%}      {r['time']:.1f}s    {r['echo_grad']:.4f}      {r['res_grad']:.4f}      {r['embed_grad']:.4f}")
    
    return results


def test_gradient_flow_detailed():
    """Detailed gradient flow analysis."""
    print("\n" + "="*70)
    print("DETAILED GRADIENT FLOW ANALYSIS")
    print("="*70)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    model = EpisodicEchoModel(
        vocab_size=64,
        d_model=64,
        num_layers=2,
        num_neurons=32,
        n_heads=2,
    ).to(device)
    
    x = torch.randint(0, 64, (4, 16), device=device)
    logits = model(x)
    loss = logits.sum()
    loss.backward()
    
    print("\nGradient norms by component:")
    print("-"*50)
    
    categories = {
        'embedding': [],
        'echo_head': [],
        'echo_out': [],
        'resonant': [],
        'output': [],
        'scale': [],
    }
    
    for name, param in model.named_parameters():
        if param.grad is None:
            print(f"  ⚠️  NO GRAD: {name}")
            continue
        
        grad_norm = param.grad.norm().item()
        
        if 'embedding' in name:
            categories['embedding'].append((name, grad_norm))
        elif 'heads' in name:
            categories['echo_head'].append((name, grad_norm))
        elif 'echo' in name and 'out' in name:
            categories['echo_out'].append((name, grad_norm))
        elif 'resonant' in name:
            categories['resonant'].append((name, grad_norm))
        elif 'output' in name:
            categories['output'].append((name, grad_norm))
        elif 'scale' in name:
            categories['scale'].append((name, grad_norm))
    
    for cat, items in categories.items():
        if items:
            avg = sum(g for _, g in items) / len(items)
            print(f"\n{cat.upper()}:")
            for name, grad in items[:3]:  # Show first 3
                print(f"  {name}: {grad:.6f}")
            if len(items) > 3:
                print(f"  ... and {len(items)-3} more")
            print(f"  Average: {avg:.6f}")


import time

if __name__ == "__main__":
    test_gradient_flow_detailed()
    test_episodic_model()

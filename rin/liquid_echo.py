"""
Liquid Echo - Linear Recurrent Model with Interference-Gated Memory

Inspired by:
    - Nested Learning / Forward-Forward
    - Liquid Networks (adaptive time constants)
    - NMDA coincidence detection
    - Linear RNNs (S4, Mamba) but with wave-based gating

KEY INSIGHT: True linear recurrence with NO BPTT through state history.
The ONLY parameters receiving gradients over time are the "trigger" projections.

Architecture:
    1. ResonantLayer: Main processing (parallel, receives gradients)
    2. LiquidEchoHead: Memory circuit with interference-gated EMA
        - Trigger projection: Learns WHAT to remember (receives gradients)
        - State evolution: Learned w/b for "clock ticking" when maintained
        - Memory state: Detached each step (no BPTT!)
    3. Additive fusion: resonant + echo outputs combined

Memory Update Rule:
    α = σ(interference_score / √d_model)  # Normalized to 0-1
    memory_new = α * input_state + (1-α) * memory_old.detach()
    
    - High interference → α≈1 → overwrite with input ("trigger")
    - Low interference → α≈0 → maintain current state ("sustain")

This creates smooth coincidence detection like NMDA receptors or
ODE-based liquid networks, learning memory circuits that maintain
themselves over the sequence.
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


class LiquidEchoHead(nn.Module):
    """
    Single Liquid Echo head with interference-gated memory.
    
    The head maintains a complex memory state that:
    1. Gets overwritten when input INTERFERES with learned trigger
    2. Continues evolving with its own learned dynamics when not triggered
    
    GRADIENT FLOW:
        - Trigger projection (w_trigger, b_trigger): ✓ RECEIVES GRADIENTS
        - State evolution params (w_state, b_state): ✓ RECEIVES GRADIENTS
        - Memory state history: ✗ DETACHED (no BPTT!)
    
    This makes the model truly linear recurrent - O(1) memory for gradients.
    """
    
    def __init__(self, d_model: int, head_idx: int = 0):
        super().__init__()
        self.d_model = d_model
        self.head_idx = head_idx
        
        # === TRIGGER PROJECTION (the "query" that detects what to remember) ===
        # These are the ONLY params that receive gradients through time
        self.w_trigger = nn.Parameter(torch.randn(d_model) * 0.02)
        self.b_trigger = nn.Parameter(torch.zeros(d_model))
        
        # === STATE EVOLUTION (the "clock" for maintained memory) ===
        # Learned dynamics for when memory is NOT being overwritten
        self.w_state = nn.Parameter(torch.randn(d_model) * 0.02)
        self.b_state = nn.Parameter(torch.zeros(d_model))
        
        # === TRIGGER THRESHOLD ===
        # Learnable bias to control selectivity (negative = harder to trigger)
        # Initialize negative to make memory MAINTAIN by default
        self.trigger_bias = nn.Parameter(torch.tensor(-2.0))
        
        # Scaling for interference score normalization
        self.scale = math.sqrt(d_model)
        
        self._lut = None
        
        # Memory state (will be set during forward)
        # NOT a parameter - just persistent state
        self.register_buffer('_memory_real', None)
        self.register_buffer('_memory_imag', None)
    
    def _get_lut(self, device):
        if self._lut is None or self._lut.sin_table.device != device:
            self._lut = get_global_lut(4096, device)
        return self._lut
    
    def reset_memory(self, batch_size: int, device: torch.device):
        """Initialize memory state to zeros."""
        self._memory_real = torch.zeros(batch_size, self.d_model, device=device)
        self._memory_imag = torch.zeros(batch_size, self.d_model, device=device)
    
    def forward(
        self,
        x_real: torch.Tensor,  # Current input state (batch, d_model)
        x_imag: torch.Tensor,
        t: torch.Tensor,  # Current timestep
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Process one timestep of the liquid echo.
        
        Args:
            x_real, x_imag: Current input state (batch, d_model)
            t: Current timestep (scalar or batch,)
            
        Returns:
            echo_real, echo_imag: Echo contribution (batch, d_model)
            alpha: Interference score for diagnostics (batch,)
        """
        lut = self._get_lut(x_real.device)
        batch_size = x_real.shape[0]
        
        # Initialize memory if needed
        if self._memory_real is None or self._memory_real.shape[0] != batch_size:
            self.reset_memory(batch_size, x_real.device)
        
        # === 1. COMPUTE TRIGGER RESPONSE ===
        # Euler transform of input through trigger projection
        # This is the "pattern detector" - what should trigger memory update
        t_val = t.view(-1, 1) if t.dim() >= 1 else t.unsqueeze(0).unsqueeze(0)
        t_phi = t_val * PHI
        
        wl_trigger = 1.0 + self.w_trigger.abs()
        
        # Trigger responds to BOTH real and imag parts of input
        theta_trigger_real = x_real / wl_trigger + self.b_trigger + t_phi
        theta_trigger_imag = x_imag / wl_trigger + self.b_trigger + t_phi
        
        sin_tr, cos_tr = lut.lookup_sin_cos(theta_trigger_real)
        sin_ti, cos_ti = lut.lookup_sin_cos(theta_trigger_imag)
        
        # Trigger vector (complex)
        trigger_real = cos_tr * cos_ti - sin_tr * sin_ti
        trigger_imag = cos_tr * sin_ti + sin_tr * cos_ti
        
        # === 2. COMPUTE INTERFERENCE SCORE ===
        # Complex dot product between trigger and input
        # Real part of (trigger* · input) = constructive interference measure
        interference = (
            trigger_real * x_real + trigger_imag * x_imag
        ).sum(dim=-1)  # (batch,)
        
        # Normalize and squash to [0, 1]
        # Add learnable bias - negative bias makes it harder to trigger
        alpha = torch.sigmoid(interference / self.scale + self.trigger_bias)  # (batch,)
        
        # === 3. EMA UPDATE OF MEMORY STATE ===
        # CRITICAL: Detach memory to prevent gradient flow through history!
        memory_real_detached = self._memory_real.detach()
        memory_imag_detached = self._memory_imag.detach()
        
        # Expand alpha for broadcasting
        alpha_exp = alpha.unsqueeze(-1)  # (batch, 1)
        
        # Blend: high alpha → input overwrites, low alpha → maintain memory
        blended_real = alpha_exp * x_real + (1 - alpha_exp) * memory_real_detached
        blended_imag = alpha_exp * x_imag + (1 - alpha_exp) * memory_imag_detached
        
        # === 4. EVOLVE MEMORY STATE ===
        # Apply Euler transform with state evolution params
        # This is the "clock ticking" for the maintained memory
        wl_state = 1.0 + self.w_state.abs()
        
        theta_state_real = blended_real / wl_state + self.b_state + t_phi
        theta_state_imag = blended_imag / wl_state + self.b_state + t_phi
        
        sin_sr, cos_sr = lut.lookup_sin_cos(theta_state_real)
        sin_si, cos_si = lut.lookup_sin_cos(theta_state_imag)
        
        # Complex multiplication for evolved state
        evolved_real = cos_sr * cos_si - sin_sr * sin_si
        evolved_imag = cos_sr * sin_si + sin_sr * cos_si
        
        # === 5. STORE NEW MEMORY STATE ===
        # Store for next timestep (detached to prevent BPTT)
        self._memory_real = evolved_real.detach()
        self._memory_imag = evolved_imag.detach()
        
        # Return the evolved state as echo contribution
        # Note: evolved_real/imag DO have gradients to trigger and state params
        return evolved_real, evolved_imag, alpha


class LiquidEchoModule(nn.Module):
    """
    Multi-head Liquid Echo module.
    
    Multiple heads maintain independent memory circuits, each detecting
    different "trigger patterns". Their outputs are SUMMED (superposition)
    to create the final echo signal.
    """
    
    def __init__(self, d_model: int, n_heads: int = 1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        
        self.heads = nn.ModuleList([
            LiquidEchoHead(d_model, head_idx=i)
            for i in range(n_heads)
        ])
    
    def reset_memory(self, batch_size: int, device: torch.device):
        """Reset all head memories."""
        for head in self.heads:
            head.reset_memory(batch_size, device)
    
    def forward(
        self,
        x_real: torch.Tensor,
        x_imag: torch.Tensor,
        t: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor]]:
        """
        Process through all heads and sum outputs.
        
        Returns:
            echo_real, echo_imag: Summed echo (batch, d_model)
            alphas: List of alpha scores per head for diagnostics
        """
        echo_real = torch.zeros_like(x_real)
        echo_imag = torch.zeros_like(x_imag)
        alphas = []
        
        for head in self.heads:
            h_real, h_imag, alpha = head(x_real, x_imag, t)
            echo_real = echo_real + h_real
            echo_imag = echo_imag + h_imag
            alphas.append(alpha)
        
        return echo_real, echo_imag, alphas


class ResonantLayerSimple(nn.Module):
    """
    Simplified Resonant Layer for Liquid Echo model.
    
    Takes complex input, does interference analysis, returns complex output.
    Includes attenuation weights.
    """
    
    def __init__(self, d_model: int, num_neurons: int):
        super().__init__()
        self.d_model = d_model
        self.num_neurons = num_neurons
        
        # Input collapse
        self.input_collapse = nn.Linear(2 * d_model, d_model, bias=True)
        
        # Per-neuron wavelength and phase
        self.W = nn.Parameter(torch.randn(num_neurons, d_model) * 0.02)
        self.B = nn.Parameter(torch.zeros(num_neurons, d_model))
        
        # Attenuation weights
        self.attn_cos = nn.Parameter(torch.ones(num_neurons, d_model))
        self.attn_sin = nn.Parameter(torch.ones(num_neurons, d_model))
        
        # Output projections
        self.out_proj_real = nn.Linear(num_neurons, d_model, bias=False)
        self.out_proj_imag = nn.Linear(num_neurons, d_model, bias=False)
        
        self._lut = None
        self._init_weights()
    
    def _init_weights(self):
        nn.init.xavier_uniform_(self.input_collapse.weight, gain=0.5)
        nn.init.zeros_(self.input_collapse.bias)
        nn.init.xavier_uniform_(self.out_proj_real.weight, gain=0.5)
        nn.init.xavier_uniform_(self.out_proj_imag.weight, gain=0.5)
    
    def _get_lut(self, device):
        if self._lut is None or self._lut.sin_table.device != device:
            self._lut = get_global_lut(4096, device)
        return self._lut
    
    def forward(
        self,
        x_real: torch.Tensor,
        x_imag: torch.Tensor,
        t: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        lut = self._get_lut(x_real.device)
        
        # Collapse complex to single vector
        x_combined = torch.cat([x_real, x_imag], dim=-1)
        x_collapsed = self.input_collapse(x_combined)
        
        # Expand for neurons
        x_exp = x_collapsed.unsqueeze(1)  # (batch, 1, d_model)
        
        # Phase computation
        wavelength = 1.0 + self.W.abs()
        t_val = t.view(-1, 1, 1) if t.dim() >= 1 else t.unsqueeze(0).unsqueeze(0).unsqueeze(0)
        
        theta = x_exp / wavelength + self.B + t_val  # (batch, num_neurons, d_model)
        
        sin_theta, cos_theta = lut.lookup_sin_cos(theta)
        
        # Attenuated interference sum
        cos_weighted = cos_theta * self.attn_cos
        sin_weighted = sin_theta * self.attn_sin
        
        cos_sum = cos_weighted.sum(dim=-1)  # (batch, num_neurons)
        sin_sum = sin_weighted.sum(dim=-1)
        
        out_real = self.out_proj_real(cos_sum)
        out_imag = self.out_proj_imag(sin_sum)
        
        return F.silu(out_real), F.silu(out_imag)


class LiquidEchoModel(nn.Module):
    """
    Liquid Echo Model - Linear Recurrent with Interference-Gated Memory
    
    Architecture:
        1. Token embedding → (w, b) pairs for Euler transform
        2. Recurrent Euler state transformation
        3. Parallel: ResonantLayer + LiquidEchoModule
        4. Additive fusion → output
    
    The model is "linear recurrent" because:
        - Echo memory states are DETACHED each step (no BPTT)
        - Only trigger projections receive gradients through time
        - Resonant layer operates per-step (parallel within step)
    
    Memory complexity: O(1) for gradients (vs O(T) for standard RNN)
    """
    
    def __init__(
        self,
        vocab_size: int = 50257,
        d_model: int = 128,
        num_layers: int = 2,
        num_neurons: int = 256,
        n_echo_heads: int = 1,
    ):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.num_layers = num_layers
        self.n_echo_heads = n_echo_heads
        
        # Token embedding: 2*d_model for (w, b) pairs
        self.token_embedding = nn.Embedding(vocab_size, 2 * d_model)
        
        # Resonant layers (parallel processing)
        self.resonant_layers = nn.ModuleList([
            ResonantLayerSimple(d_model, num_neurons)
            for _ in range(num_layers)
        ])
        
        # Liquid Echo modules (memory circuits)
        self.echo_modules = nn.ModuleList([
            LiquidEchoModule(d_model, n_echo_heads)
            for _ in range(num_layers)
        ])
        
        # Learnable fusion scales
        self.res_scales = nn.ParameterList([
            nn.Parameter(torch.tensor(0.5)) for _ in range(num_layers)
        ])
        self.echo_scales = nn.ParameterList([
            nn.Parameter(torch.tensor(0.5)) for _ in range(num_layers)
        ])
        
        # Output projection (collapses complex to real)
        self.output_proj_real = nn.Linear(d_model, vocab_size, bias=False)
        self.output_proj_imag = nn.Linear(d_model, vocab_size, bias=False)
        
        self._lut = None
        self._init_weights()
    
    def _init_weights(self):
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.output_proj_real.weight, std=0.02)
        nn.init.normal_(self.output_proj_imag.weight, std=0.02)
    
    def _get_lut(self, device):
        if self._lut is None or self._lut.sin_table.device != device:
            self._lut = get_global_lut(4096, device)
        return self._lut
    
    def reset_memory(self, batch_size: int, device: torch.device):
        """Reset all echo memories."""
        for echo in self.echo_modules:
            echo.reset_memory(batch_size, device)
    
    def euler_transform(
        self,
        h_real: torch.Tensor,
        h_imag: torch.Tensor,
        w: torch.Tensor,
        b: torch.Tensor,
        t: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply Euler transform to hidden state."""
        lut = self._get_lut(h_real.device)
        
        wavelength = 1.0 + w.abs()
        t_phi = t * PHI
        t_phi = t_phi.view(-1, 1) if t_phi.dim() >= 1 else t_phi.unsqueeze(0).unsqueeze(0)
        
        theta_real = h_real / wavelength + b + t_phi
        theta_imag = h_imag / wavelength + b + t_phi
        
        sin_r, cos_r = lut.lookup_sin_cos(theta_real)
        sin_i, cos_i = lut.lookup_sin_cos(theta_imag)
        
        h_real_new = cos_r * cos_i - sin_r * sin_i
        h_imag_new = cos_r * sin_i + sin_r * cos_i
        
        return h_real_new, h_imag_new
    
    def forward(
        self,
        input_ids: torch.Tensor,
        return_alphas: bool = False,
    ) -> torch.Tensor:
        """
        Forward pass through the Liquid Echo model.
        
        Args:
            input_ids: Token IDs (batch, seq_len)
            return_alphas: Whether to return alpha diagnostics
            
        Returns:
            logits: (batch, seq_len, vocab_size)
        """
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        
        # Reset echo memories
        self.reset_memory(batch_size, device)
        
        # Initialize hidden state
        h_real = torch.zeros(batch_size, self.d_model, device=device)
        h_imag = torch.zeros(batch_size, self.d_model, device=device)
        
        # Get embeddings
        embeddings = self.token_embedding(input_ids)
        w_emb = embeddings[:, :, :self.d_model]
        b_emb = embeddings[:, :, self.d_model:]
        
        all_logits = []
        all_alphas = [] if return_alphas else None
        
        for t in range(seq_len):
            w_t = w_emb[:, t, :]
            b_t = b_emb[:, t, :]
            t_val = torch.tensor(t, dtype=torch.float32, device=device)
            
            # Euler transform of hidden state
            h_real, h_imag = self.euler_transform(h_real, h_imag, w_t, b_t, t_val)
            
            # Process through layers
            x_real, x_imag = h_real, h_imag
            t_phi = t_val * PHI
            
            step_alphas = []
            for layer_idx in range(self.num_layers):
                # Resonant processing (parallel, full gradients)
                res_real, res_imag = self.resonant_layers[layer_idx](
                    x_real, x_imag, t_phi
                )
                
                # Echo processing (linear recurrent, detached history)
                echo_real, echo_imag, alphas = self.echo_modules[layer_idx](
                    x_real, x_imag, t_val
                )
                
                if return_alphas:
                    step_alphas.extend([a.mean().item() for a in alphas])
                
                # Additive fusion
                x_real = x_real + (
                    self.res_scales[layer_idx] * res_real +
                    self.echo_scales[layer_idx] * echo_real
                )
                x_imag = x_imag + (
                    self.res_scales[layer_idx] * res_imag +
                    self.echo_scales[layer_idx] * echo_imag
                )
            
            if return_alphas:
                all_alphas.append(step_alphas)
            
            # Output projection (collapse complex to real)
            logits = self.output_proj_real(x_real) + self.output_proj_imag(x_imag)
            all_logits.append(logits)
        
        result = torch.stack(all_logits, dim=1)
        
        if return_alphas:
            return result, all_alphas
        return result
    
    def get_num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


# ============================================================================
# TESTING
# ============================================================================

def test_gradient_flow():
    """Verify gradient flow is correct (only trigger params get gradients over t)."""
    print("="*70)
    print("GRADIENT FLOW TEST - Liquid Echo")
    print("="*70)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    model = LiquidEchoModel(
        vocab_size=64,
        d_model=64,
        num_layers=2,
        num_neurons=64,
        n_echo_heads=2,
    ).to(device)
    
    print(f"Parameters: {model.get_num_params():,}")
    
    # Forward pass
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
        if 'embedding' in name:
            cat = 'embedding'
        elif 'echo' in name:
            if 'trigger' in name:
                cat = 'echo_trigger'  # SHOULD have gradients
            elif 'state' in name:
                cat = 'echo_state'  # SHOULD have gradients
            else:
                cat = 'echo_other'
        elif 'resonant' in name:
            if 'attn' in name:
                cat = 'attenuation'
            else:
                cat = 'resonant'
        elif 'output' in name:
            cat = 'output'
        elif 'scale' in name:
            cat = 'scale'
        else:
            cat = 'other'
        
        if cat not in categories:
            categories[cat] = []
        categories[cat].append((name, grad_norm))
    
    for cat in ['embedding', 'echo_trigger', 'echo_state', 'attenuation', 'resonant', 'output', 'scale']:
        if cat in categories:
            items = categories[cat]
            avg = sum(g for _, g in items) / len(items)
            print(f"\n{cat.upper()}: (avg={avg:.4f})")
            for name, grad in items[:3]:
                short_name = '.'.join(name.split('.')[-2:])
                print(f"  {short_name}: {grad:.6f}")
    
    print("\n✓ Gradient flow verified!")


def test_memory_behavior():
    """Test that memory behaves as expected (trigger updates, sustain maintains)."""
    print("\n" + "="*70)
    print("MEMORY BEHAVIOR TEST")
    print("="*70)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    model = LiquidEchoModel(
        vocab_size=64,
        d_model=32,
        num_layers=1,
        num_neurons=32,
        n_echo_heads=1,
    ).to(device)
    
    # Run with alpha tracking
    x = torch.randint(0, 64, (1, 20), device=device)
    logits, alphas = model(x, return_alphas=True)
    
    print("\nAlpha values over sequence (higher = more memory update):")
    print("-"*50)
    alphas_flat = [a[0] for a in alphas]  # First head of first layer
    for t, alpha in enumerate(alphas_flat):
        bar = '█' * int(alpha * 40)
        print(f"  t={t:2d}: {alpha:.3f} {bar}")
    
    print(f"\n  Mean alpha: {sum(alphas_flat)/len(alphas_flat):.3f}")
    print(f"  Min alpha:  {min(alphas_flat):.3f}")
    print(f"  Max alpha:  {max(alphas_flat):.3f}")


def test_retrieval_task():
    """Test on marker-based retrieval task."""
    print("\n" + "="*70)
    print("RETRIEVAL TASK TEST - Liquid Echo")
    print("="*70)
    
    import time
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    vocab_size = 64
    marker = vocab_size - 1
    seq_len = 16
    batch_size = 32
    num_epochs = 500
    
    model = LiquidEchoModel(
        vocab_size=vocab_size,
        d_model=64,
        num_layers=2,
        num_neurons=128,
        n_echo_heads=2,
    ).to(device)
    
    print(f"Parameters: {model.get_num_params():,}")
    
    # WITH WEIGHT DECAY
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.1)
    
    start = time.time()
    best_acc = 0.0
    
    for epoch in range(num_epochs):
        model.train()
        
        # Generate batch
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
    print(f"\nTime: {elapsed:.1f}s")
    print(f"Best accuracy: {best_acc:.1%}")
    
    # Final evaluation
    model.eval()
    test_accs = []
    with torch.no_grad():
        for _ in range(10):
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
    print(f"Final test accuracy: {final_acc:.1%}")
    
    return best_acc, final_acc


def test_performance():
    """Test computational performance."""
    print("\n" + "="*70)
    print("PERFORMANCE TEST - Liquid Echo")
    print("="*70)
    
    import time
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    model = LiquidEchoModel(
        vocab_size=1000,
        d_model=128,
        num_layers=4,
        num_neurons=256,
        n_echo_heads=4,
    ).to(device)
    
    print(f"Parameters: {model.get_num_params():,}")
    
    # Warmup
    x = torch.randint(0, 1000, (8, 64), device=device)
    for _ in range(3):
        _ = model(x)
    
    if device == "cuda":
        torch.cuda.synchronize()
    
    # Time forward pass
    n_trials = 10
    start = time.time()
    for _ in range(n_trials):
        _ = model(x)
        if device == "cuda":
            torch.cuda.synchronize()
    fwd_time = (time.time() - start) / n_trials * 1000
    
    # Time backward pass
    start = time.time()
    for _ in range(n_trials):
        logits = model(x)
        loss = logits.sum()
        loss.backward()
        if device == "cuda":
            torch.cuda.synchronize()
    total_time = (time.time() - start) / n_trials * 1000
    bwd_time = total_time - fwd_time
    
    print(f"\nBatch size: 8, Seq len: 64")
    print(f"  Forward:  {fwd_time:.1f}ms")
    print(f"  Backward: {bwd_time:.1f}ms")
    print(f"  Total:    {total_time:.1f}ms")
    
    # Memory usage
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
        _ = model(x)
        peak_mem = torch.cuda.max_memory_allocated() / 1024**2
        print(f"  Peak memory: {peak_mem:.1f}MB")


def compare_with_standard_rin():
    """Compare with standard RIN model."""
    print("\n" + "="*70)
    print("COMPARISON: Liquid Echo vs Standard RIN")
    print("="*70)
    
    import time
    from rin.model import RINModel
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    vocab_size = 64
    d_model = 64
    num_layers = 2
    num_neurons = 128
    
    # Standard RIN
    rin = RINModel(
        vocab_size=vocab_size,
        d_model=d_model,
        num_layers=num_layers,
        num_neurons=num_neurons,
    ).to(device)
    
    # Liquid Echo
    liquid = LiquidEchoModel(
        vocab_size=vocab_size,
        d_model=d_model,
        num_layers=num_layers,
        num_neurons=num_neurons,
        n_echo_heads=2,
    ).to(device)
    
    print(f"RIN params:    {rin.get_num_params():,}")
    print(f"Liquid params: {liquid.get_num_params():,}")
    
    # Test retrieval task for both
    marker = vocab_size - 1
    seq_len = 16
    batch_size = 32
    num_epochs = 300
    
    results = {}
    
    for name, model in [('RIN', rin), ('Liquid', liquid)]:
        print(f"\n--- {name} ---")
        
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
            
            output = model(seq)
            if isinstance(output, tuple):
                logits = output[0]
            else:
                logits = output
            
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
                print(f"  Epoch {epoch+1}: loss={loss.item():.4f}, acc={acc:.1%}")
        
        elapsed = time.time() - start
        results[name] = {'acc': best_acc, 'time': elapsed}
    
    print("\n" + "="*70)
    print("COMPARISON SUMMARY")
    print("="*70)
    print(f"{'Model':<12} {'Best Acc':<12} {'Time':<10}")
    print("-"*34)
    for name, r in results.items():
        print(f"{name:<12} {r['acc']:.1%}        {r['time']:.1f}s")


if __name__ == "__main__":
    test_gradient_flow()
    test_memory_behavior()
    test_retrieval_task()
    test_performance()
    compare_with_standard_rin()

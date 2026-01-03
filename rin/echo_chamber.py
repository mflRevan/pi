"""
Echo Chamber - Sparse Memory with Frequency-Gated Updates

Architecture:
    - Multiple heads, each with d_head = d_model // n_heads
    - Each head learns a "frequency query" (Euler transform with w, b)
    - Query computes interference score with input patch
    - sigmoid(score / sqrt(d_head)) determines write strength
    - ALL heads write to ONE SHARED complex memory state (d_model)
    - Single Euler output projection transforms memory → additive to datapath

Key insight: The query learns to recognize specific frequency patterns.
When input patch "resonates" with query frequency → high score → overwrite memory.
When no resonance → low score → preserve memory.

This creates a frequency-selective latch that can:
    - Lock onto specific patterns
    - Maintain state across long sequences
    - Output through learned frequency projection
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, List, Dict
import time

import sys
sys.path.insert(0, '/home/aiman/pi')

from rin.lut import get_global_lut

PHI = (1 + math.sqrt(5)) / 2


class EchoHead(nn.Module):
    """
    Single echo head - operates on d_head sized patch.
    
    Key insight: We compare input against a LEARNED TRIGGER, not memory.
    The trigger is what this head is "looking for" - a pattern detector.
    When input resonates with trigger → high score → write to memory.
    Memory is INDEPENDENT - fluid state that persists across time.
    """
    
    def __init__(self, d_head: int, head_idx: int, d_model: int):
        super().__init__()
        self.d_head = d_head
        self.head_idx = head_idx
        self.d_model = d_model
        
        # LEARNED TRIGGER - the pattern this head detects
        # This is what we compare input against (NOT memory)
        self.trigger_real = nn.Parameter(torch.randn(d_head) * 0.1)
        self.trigger_imag = nn.Parameter(torch.randn(d_head) * 0.1)
        
        # Query projection - transforms input for comparison
        self.w_query = nn.Parameter(torch.randn(d_head) * 0.02)
        self.b_query = nn.Parameter(torch.zeros(d_head))
        
        # Learned temperature for sigmoid
        self.temperature = nn.Parameter(torch.tensor(1.0))
        
        self._lut = None
    
    def _get_lut(self, device):
        if self._lut is None or self._lut.sin_table.device != device:
            self._lut = get_global_lut(4096, device)
        return self._lut
    
    def forward(
        self,
        x_real: torch.Tensor,  # (batch, d_head) - input patch real
        x_imag: torch.Tensor,  # (batch, d_head) - input patch imag
        t: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Compute write score by comparing input to learned trigger.
        
        Returns:
            score: (batch,) sigmoid score in [0, 1]
            diagnostics: debug info including RAW scores
        """
        lut = self._get_lut(x_real.device)
        batch_size = x_real.shape[0]
        
        t_val = t.view(-1, 1) if t.dim() >= 1 else t.unsqueeze(0).unsqueeze(0)
        t_phi = t_val * PHI
        
        # === QUERY: Transform input through Euler ===
        wavelength = 1.0 + self.w_query.abs()
        
        theta_real = x_real / wavelength + self.b_query + t_phi
        theta_imag = x_imag / wavelength + self.b_query + t_phi
        
        sin_r, cos_r = lut.lookup_sin_cos(theta_real)
        sin_i, cos_i = lut.lookup_sin_cos(theta_imag)
        
        # Query = Euler transform of input
        query_real = cos_r * cos_i - sin_r * sin_i
        query_imag = cos_r * sin_i + sin_r * cos_i
        
        # === CONJUGATE INTERFERENCE: query* · trigger ===
        # Compare transformed input against LEARNED TRIGGER
        # Re(z1* · z2) = Re(z1)·Re(z2) + Im(z1)·Im(z2)  <- SIGNED (constructive/destructive)
        # Im(z1* · z2) = Re(z1)·Im(z2) - Im(z1)·Re(z2)  <- SIGNED (phase rotation)
        
        interference_real = (query_real * self.trigger_real + query_imag * self.trigger_imag).sum(dim=-1)
        interference_imag = (query_real * self.trigger_imag - query_imag * self.trigger_real).sum(dim=-1)
        
        # Normalize by magnitudes for scale invariance
        trigger_mag = torch.sqrt((self.trigger_real**2 + self.trigger_imag**2).sum() + 1e-8)
        query_mag = torch.sqrt((query_real**2 + query_imag**2).sum(dim=-1) + 1e-8)
        
        # Normalized complex interference (both components in [-1, 1])
        norm_int_real = interference_real / (query_mag * trigger_mag + 1e-8)
        norm_int_imag = interference_imag / (query_mag * trigger_mag + 1e-8)
        
        # CONSTRUCTIVE INTERFERENCE ONLY:
        # Perfect match: norm_int_real = +1, norm_int_imag = 0
        # Destructive:   norm_int_real = -1, norm_int_imag = 0
        # Distance from perfect constructive interference
        # Using exp(-k * distance) ensures only constructive interference scores high
        distance_from_ideal = torch.sqrt((1.0 - norm_int_real)**2 + norm_int_imag**2 + 1e-8)
        
        # Exponential decay: exp(-k * distance)
        # When distance=0 (perfect match) → exp(0) = 1
        # When distance=2 (destructive) → exp(-2k) ≈ 0
        temp = self.temperature.abs() + 0.1  # Learnable decay rate
        score = torch.exp(-temp * distance_from_ideal)
        
        diagnostics = {
            'interference_real': interference_real.detach(),
            'interference_imag': interference_imag.detach(),
            'norm_int_real': norm_int_real.detach(),
            'norm_int_imag': norm_int_imag.detach(),
            'distance_from_ideal': distance_from_ideal.detach(),
            'temperature': temp.detach(),
            'score': score.detach(),
            'query_mag': query_mag.detach(),
            'trigger_mag': trigger_mag.detach(),
        }
        
        return score, diagnostics


class EchoChamber(nn.Module):
    """
    Echo Chamber - Multiple heads writing to ONE shared memory.
    
    Architecture:
        - n_heads, each with d_head = d_model // n_heads
        - Input split into patches: x[:, head_idx*d_head : (head_idx+1)*d_head]
        - Each head computes write score via frequency matching
        - Scores combined → single alpha for memory update
        - Memory is (batch, d_model) complex
        - Output: Euler-projected memory, additive to datapath
    """
    
    def __init__(self, d_model: int, n_heads: int = 4):
        super().__init__()
        
        assert d_model % n_heads == 0, f"d_model {d_model} must be divisible by n_heads {n_heads}"
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        
        # Echo heads - each operates on a patch
        self.heads = nn.ModuleList([
            EchoHead(self.d_head, head_idx=i, d_model=d_model)
            for i in range(n_heads)
        ])
        
        # Output Euler projection - transforms memory for datapath
        # Learns the frequency at which memory influences computation
        self.w_out = nn.Parameter(torch.randn(d_model) * 0.02)
        self.b_out = nn.Parameter(torch.zeros(d_model))
        
        # Memory state (stored as real, imag separately)
        self._memory_real = None
        self._memory_imag = None
        
        self._lut = None
    
    def _get_lut(self, device):
        if self._lut is None or self._lut.sin_table.device != device:
            self._lut = get_global_lut(4096, device)
        return self._lut
    
    def reset_memory(self, batch_size: int, device: torch.device):
        """Initialize shared memory to zeros."""
        self._memory_real = torch.zeros(batch_size, self.d_model, device=device)
        self._memory_imag = torch.zeros(batch_size, self.d_model, device=device)
    
    def forward(
        self,
        x_real: torch.Tensor,  # (batch, d_model)
        x_imag: torch.Tensor,  # (batch, d_model)
        t: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """
        Process input and update shared memory.
        
        Returns:
            out_real, out_imag: Euler-projected memory output
            diagnostics: per-head and aggregate debug info
        """
        lut = self._get_lut(x_real.device)
        batch_size = x_real.shape[0]
        
        if self._memory_real is None or self._memory_real.shape[0] != batch_size:
            self.reset_memory(batch_size, x_real.device)
        
        t_val = t.view(-1, 1) if t.dim() >= 1 else t.unsqueeze(0).unsqueeze(0)
        t_phi = t_val * PHI
        
        # === COMPUTE SCORES FROM ALL HEADS ===
        head_scores = []
        head_diagnostics = []
        
        for head_idx, head in enumerate(self.heads):
            # Extract patch for this head
            start = head_idx * self.d_head
            end = start + self.d_head
            
            patch_real = x_real[:, start:end]
            patch_imag = x_imag[:, start:end]
            
            score, diag = head(patch_real, patch_imag, t)
            head_scores.append(score)
            head_diagnostics.append(diag)
        
        # === COMBINE SCORES ===
        # Stack: (batch, n_heads)
        scores = torch.stack(head_scores, dim=-1)
        
        # Mean across heads - if ANY head triggers strongly, we write
        # Alternative: max, or learned combination
        alpha = scores.mean(dim=-1)  # (batch,)
        
        # === MEMORY UPDATE ===
        # Detach old memory (no BPTT through history)
        memory_real_det = self._memory_real.detach()
        memory_imag_det = self._memory_imag.detach()
        
        alpha_exp = alpha.unsqueeze(-1)  # (batch, 1)
        
        # EMA blend: high alpha → overwrite with input, low alpha → keep memory
        new_memory_real = alpha_exp * x_real + (1 - alpha_exp) * memory_real_det
        new_memory_imag = alpha_exp * x_imag + (1 - alpha_exp) * memory_imag_det
        
        # Store updated memory (detached for next step)
        self._memory_real = new_memory_real.detach()
        self._memory_imag = new_memory_imag.detach()
        
        # === OUTPUT: EULER PROJECTION OF MEMORY ===
        # Transform memory through learned frequency for output
        wavelength = 1.0 + self.w_out.abs()
        
        theta_out_real = new_memory_real / wavelength + self.b_out + t_phi
        theta_out_imag = new_memory_imag / wavelength + self.b_out + t_phi
        
        sin_or, cos_or = lut.lookup_sin_cos(theta_out_real)
        sin_oi, cos_oi = lut.lookup_sin_cos(theta_out_imag)
        
        # Complex multiplication for output
        out_real = cos_or * cos_oi - sin_or * sin_oi
        out_imag = cos_or * sin_oi + sin_or * cos_oi
        
        diagnostics = {
            'head_scores': scores.detach(),
            'alpha': alpha.detach(),
            'memory_mag': (new_memory_real**2 + new_memory_imag**2).sum(-1).sqrt().detach(),
            'output_mag': (out_real**2 + out_imag**2).sum(-1).sqrt().detach(),
            'head_details': head_diagnostics,
        }
        
        return out_real, out_imag, diagnostics


class ResonantLayer(nn.Module):
    """Resonant layer with attenuation (from model.py)."""
    
    def __init__(self, d_model: int, num_neurons: int):
        super().__init__()
        self.d_model = d_model
        self.num_neurons = num_neurons
        
        self.input_collapse = nn.Linear(2 * d_model, d_model, bias=True)
        
        self.W = nn.Parameter(torch.randn(num_neurons, d_model) * 0.02)
        self.B = nn.Parameter(torch.zeros(num_neurons, d_model))
        
        self.attn_cos = nn.Parameter(torch.ones(num_neurons, d_model))
        self.attn_sin = nn.Parameter(torch.ones(num_neurons, d_model))
        
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
    
    def forward(self, x_real: torch.Tensor, x_imag: torch.Tensor, t: torch.Tensor):
        lut = self._get_lut(x_real.device)
        
        x_combined = torch.cat([x_real, x_imag], dim=-1)
        x_collapsed = self.input_collapse(x_combined)
        
        x_exp = x_collapsed.unsqueeze(1)
        wavelength = 1.0 + self.W.abs()
        t_val = t.view(-1, 1, 1) if t.dim() >= 1 else t.unsqueeze(0).unsqueeze(0).unsqueeze(0)
        
        theta = x_exp / wavelength + self.B + t_val
        
        sin_theta, cos_theta = lut.lookup_sin_cos(theta)
        
        cos_weighted = cos_theta * self.attn_cos
        sin_weighted = sin_theta * self.attn_sin
        
        cos_sum = cos_weighted.sum(dim=-1)
        sin_sum = sin_weighted.sum(dim=-1)
        
        out_real = self.out_proj_real(cos_sum)
        out_imag = self.out_proj_imag(sin_sum)
        
        return F.silu(out_real), F.silu(out_imag)


class EchoChamberModel(nn.Module):
    """
    Model with Echo Chamber for frequency-gated memory.
    
    Architecture:
        - Token embedding → Euler state evolution
        - Per-layer: Resonant processing + Echo Chamber output
        - Echo Chamber output added to datapath (additive interference)
        - Output projection to logits
    """
    
    def __init__(
        self,
        vocab_size: int = 50257,
        d_model: int = 128,
        num_layers: int = 2,
        num_neurons: int = 256,
        n_echo_heads: int = 4,
        fusion_mode: str = "additive",  # or "multiplicative"
    ):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.num_layers = num_layers
        self.n_echo_heads = n_echo_heads
        self.fusion_mode = fusion_mode
        
        self.token_embedding = nn.Embedding(vocab_size, 2 * d_model)
        
        self.resonant_layers = nn.ModuleList([
            ResonantLayer(d_model, num_neurons)
            for _ in range(num_layers)
        ])
        
        # ONE echo chamber per layer (each has shared memory across its heads)
        self.echo_chambers = nn.ModuleList([
            EchoChamber(d_model, n_echo_heads)
            for _ in range(num_layers)
        ])
        
        # Scale for echo contribution
        self.echo_scales = nn.ParameterList([
            nn.Parameter(torch.tensor(0.1))
            for _ in range(num_layers)
        ])
        
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
        for chamber in self.echo_chambers:
            chamber.reset_memory(batch_size, device)
    
    def euler_transform(self, h_real, h_imag, w, b, t):
        lut = self._get_lut(h_real.device)
        
        wavelength = 1.0 + w.abs()
        t_phi = t * PHI
        t_phi = t_phi.view(-1, 1) if t_phi.dim() >= 1 else t_phi.unsqueeze(0).unsqueeze(0)
        
        theta_real = h_real / wavelength + b + t_phi
        theta_imag = h_imag / wavelength + b + t_phi
        
        sin_r, cos_r = lut.lookup_sin_cos(theta_real)
        sin_i, cos_i = lut.lookup_sin_cos(theta_imag)
        
        return cos_r * cos_i - sin_r * sin_i, cos_r * sin_i + sin_r * cos_i
    
    def forward(self, input_ids: torch.Tensor, return_diagnostics: bool = False):
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        
        self.reset_memory(batch_size, device)
        
        h_real = torch.zeros(batch_size, self.d_model, device=device)
        h_imag = torch.zeros(batch_size, self.d_model, device=device)
        
        embeddings = self.token_embedding(input_ids)
        w_emb = embeddings[:, :, :self.d_model]
        b_emb = embeddings[:, :, self.d_model:]
        
        all_logits = []
        all_diagnostics = [] if return_diagnostics else None
        
        for t in range(seq_len):
            w_t = w_emb[:, t, :]
            b_t = b_emb[:, t, :]
            t_val = torch.tensor(t, dtype=torch.float32, device=device)
            
            h_real, h_imag = self.euler_transform(h_real, h_imag, w_t, b_t, t_val)
            
            x_real, x_imag = h_real, h_imag
            t_phi = t_val * PHI
            
            step_diag = {'t': t} if return_diagnostics else None
            
            for layer_idx in range(self.num_layers):
                # Resonant processing
                res_real, res_imag = self.resonant_layers[layer_idx](x_real, x_imag, t_phi)
                
                # Echo chamber - frequency-gated memory output
                echo_real, echo_imag, echo_diag = self.echo_chambers[layer_idx](
                    x_real, x_imag, t_val
                )
                
                if return_diagnostics:
                    step_diag[f'layer{layer_idx}_echo'] = echo_diag
                
                # Fusion
                scale = self.echo_scales[layer_idx]
                
                if self.fusion_mode == "additive":
                    # Echo adds to resonant output
                    combined_real = res_real + scale * echo_real
                    combined_imag = res_imag + scale * echo_imag
                else:  # multiplicative
                    combined_real = res_real * (1.0 + scale * echo_real)
                    combined_imag = res_imag * (1.0 + scale * echo_imag)
                
                if return_diagnostics:
                    step_diag[f'layer{layer_idx}_res_mag'] = (res_real**2 + res_imag**2).sum(-1).sqrt().mean().item()
                    step_diag[f'layer{layer_idx}_echo_mag'] = (echo_real**2 + echo_imag**2).sum(-1).sqrt().mean().item()
                
                # Residual
                x_real = x_real + combined_real
                x_imag = x_imag + combined_imag
            
            if return_diagnostics:
                all_diagnostics.append(step_diag)
            
            logits = self.output_proj_real(x_real) + self.output_proj_imag(x_imag)
            all_logits.append(logits)
        
        result = torch.stack(all_logits, dim=1)
        
        if return_diagnostics:
            return result, all_diagnostics
        return result
    
    def get_num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


# ============================================================================
# TESTING
# ============================================================================

def test_gradient_flow():
    """Verify gradients flow to all echo components."""
    print("="*80)
    print("GRADIENT FLOW TEST")
    print("="*80)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    model = EchoChamberModel(
        vocab_size=64,
        d_model=64,
        num_layers=2,
        num_neurons=64,
        n_echo_heads=4,
    ).to(device)
    
    print(f"Parameters: {model.get_num_params():,}")
    print(f"d_head = {64 // 4} = 16")
    
    x = torch.randint(0, 64, (4, 12), device=device)
    logits = model(x)
    loss = F.cross_entropy(logits[:, -1, :], torch.zeros(4, dtype=torch.long, device=device))
    loss.backward()
    
    print("\nGradient norms by component:")
    print("-"*80)
    
    cats = {'query': [], 'output_euler': [], 'resonant': [], 'echo_scale': [], 'output': []}
    
    for name, param in model.named_parameters():
        if param.grad is None or param.grad.norm().item() == 0:
            continue
        
        grad = param.grad.norm().item()
        
        if 'w_query' in name or 'b_query' in name:
            cats['query'].append((name, grad))
        elif 'w_out' in name or 'b_out' in name:
            cats['output_euler'].append((name, grad))
        elif 'echo_scale' in name:
            cats['echo_scale'].append((name, grad))
        elif 'output_proj' in name:
            cats['output'].append((name, grad))
        else:
            cats['resonant'].append((name, grad))
    
    for cat, items in cats.items():
        if items:
            avg = sum(g for _, g in items) / len(items)
            print(f"\n{cat.upper()} (avg={avg:.6f}, count={len(items)}):")
            for name, g in items[:3]:
                short = '.'.join(name.split('.')[-2:])
                print(f"  {short}: {g:.6f}")
    
    print("\n✓ Gradient flow test complete")


def test_alpha_dynamics():
    """Test write score dynamics over sequence - with detailed interference analysis."""
    print("\n" + "="*80)
    print("ALPHA (WRITE SCORE) DYNAMICS - DETAILED INTERFERENCE ANALYSIS")
    print("="*80)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    model = EchoChamberModel(
        vocab_size=64,
        d_model=64,
        num_layers=1,
        num_neurons=64,
        n_echo_heads=4,
    ).to(device)
    
    x = torch.randint(0, 64, (1, 20), device=device)
    _, diag = model(x, return_diagnostics=True)
    
    print(f"\nd_head = {64 // 4} = 16 per head")
    print(f"n_heads = 4, all writing to shared memory")
    print(f"Comparing input to LEARNED TRIGGER (not memory)")
    
    # Collect all scores for analysis
    all_interference_mags = []
    all_normalized = []
    all_raw = []
    all_temps = []
    all_alphas = []
    
    print("\n" + "-"*100)
    print(f"{'t':>3} | {'α':>6} | {'int_mag':>10} | {'normalized':>10} | {'raw':>10} | {'temp':>6} | Head Scores")
    print("-"*100)
    
    for step in diag:
        t = step['t']
        echo_diag = step['layer0_echo']
        alpha = echo_diag['alpha'].mean().item()
        head_scores = echo_diag['head_scores'].squeeze().tolist()
        
        # Get detailed per-head diagnostics
        head_details = echo_diag['head_details']
        
        # Average across heads for summary
        int_mag = sum(h['interference_mag'].mean().item() for h in head_details) / len(head_details)
        norm = sum(h['normalized_score'].mean().item() for h in head_details) / len(head_details)
        raw = sum(h['raw_score'].mean().item() for h in head_details) / len(head_details)
        temp = head_details[0]['temperature'].item()
        
        all_interference_mags.append(int_mag)
        all_normalized.append(norm)
        all_raw.append(raw)
        all_temps.append(temp)
        all_alphas.append(alpha)
        
        head_str = ' '.join([f'{s:.2f}' for s in head_scores])
        bar = '█' * int(alpha * 20)
        print(f"{t:3d} | {alpha:.4f} | {int_mag:10.4f} | {norm:10.4f} | {raw:10.4f} | {temp:6.2f} | [{head_str}] {bar}")
    
    print("-"*100)
    
    # Statistics
    def stats(name, vals):
        mean = sum(vals) / len(vals)
        std = (sum((v - mean)**2 for v in vals) / len(vals)) ** 0.5
        return f"{name}: mean={mean:.4f}, std={std:.4f}, min={min(vals):.4f}, max={max(vals):.4f}"
    
    print(f"\n{stats('interference_mag', all_interference_mags)}")
    print(f"{stats('normalized_score', all_normalized)}")
    print(f"{stats('raw_score', all_raw)}")
    print(f"{stats('alpha', all_alphas)}")
    print(f"temperature: {all_temps[0]:.4f}")
    
    # Analyze dynamic range
    print("\n" + "="*80)
    print("DYNAMIC RANGE ANALYSIS")
    print("="*80)
    
    # Show per-head trigger magnitudes
    print("\nPer-head trigger magnitudes (learned patterns):")
    for i, head in enumerate(model.echo_chambers[0].heads):
        t_mag = (head.trigger_real**2 + head.trigger_imag**2).sum().sqrt().item()
        print(f"  Head {i}: trigger_mag = {t_mag:.4f}")
    
    # Analyze what sigmoid needs
    print(f"\nSigmoid analysis:")
    print(f"  sigmoid(-2) = {torch.sigmoid(torch.tensor(-2.0)).item():.4f}")
    print(f"  sigmoid(-1) = {torch.sigmoid(torch.tensor(-1.0)).item():.4f}")
    print(f"  sigmoid(0) = {torch.sigmoid(torch.tensor(0.0)).item():.4f}")
    print(f"  sigmoid(1) = {torch.sigmoid(torch.tensor(1.0)).item():.4f}")
    print(f"  sigmoid(2) = {torch.sigmoid(torch.tensor(2.0)).item():.4f}")
    
    print(f"\nFor sparse triggering, raw_score should be:")
    print(f"  - Mostly NEGATIVE (< -1) → α < 0.27 (no write)")
    print(f"  - Occasionally POSITIVE (> 1) → α > 0.73 (write)")
    print(f"  Current raw_score range: [{min(all_raw):.4f}, {max(all_raw):.4f}]")


def test_retrieval_task():
    """Test on retrieval task."""
    print("\n" + "="*80)
    print("RETRIEVAL TASK TEST")
    print("="*80)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    vocab_size = 64
    marker = vocab_size - 1
    seq_len = 16
    batch_size = 32
    num_epochs = 500
    
    model = EchoChamberModel(
        vocab_size=vocab_size,
        d_model=64,
        num_layers=2,
        num_neurons=128,
        n_echo_heads=4,
        fusion_mode="additive",
    ).to(device)
    
    print(f"Parameters: {model.get_num_params():,}")
    print(f"Config: 4 heads, d_head=16, shared memory per layer")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.1)
    
    best_acc = 0.0
    best_epoch = 0
    start = time.time()
    
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
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        if epoch % 50 == 49:
            model.eval()
            with torch.no_grad():
                pred = logits[:, -1, :].argmax(dim=-1)
                acc = (pred == targets).float().mean().item()
                if acc > best_acc:
                    best_acc = acc
                    best_epoch = epoch + 1
            
            print(f"Epoch {epoch+1:3d}: loss={loss.item():.4f}, acc={acc:5.1%}, best={best_acc:5.1%} @{best_epoch}")
    
    elapsed = time.time() - start
    
    # Final diagnostics
    model.eval()
    with torch.no_grad():
        _, diag = model(seq, return_diagnostics=True)
    
    alphas = [d['layer0_echo']['alpha'].mean().item() for d in diag]
    alpha_mean = sum(alphas) / len(alphas)
    alpha_std = (sum((a - alpha_mean)**2 for a in alphas) / len(alphas)) ** 0.5
    write_70 = sum(1 for a in alphas if a > 0.7) / len(alphas)
    
    print(f"\nFinal alpha: mean={alpha_mean:.3f}, std={alpha_std:.3f}, write>70%={write_70*100:.1f}%")
    print(f"Time: {elapsed:.1f}s")
    print(f"Best accuracy: {best_acc:.1%} @epoch {best_epoch}")
    
    # Show alpha at each position
    print("\nFinal alpha sequence:")
    for t, a in enumerate(alphas):
        bar = '█' * int(a * 30)
        print(f"  t={t:2d}: α={a:.3f} {bar}")
    
    return best_acc


def test_comparison_heads():
    """Compare different head counts."""
    print("\n" + "="*80)
    print("HEAD COUNT COMPARISON")
    print("="*80)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    vocab_size = 64
    marker = vocab_size - 1
    seq_len = 16
    batch_size = 32
    num_epochs = 300
    
    results = {}
    
    for n_heads in [1, 2, 4, 8]:
        print(f"\n--- n_heads = {n_heads}, d_head = {64 // n_heads} ---")
        
        model = EchoChamberModel(
            vocab_size=vocab_size,
            d_model=64,
            num_layers=2,
            num_neurons=128,
            n_echo_heads=n_heads,
            fusion_mode="additive",
        ).to(device)
        
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.1)
        
        best_acc = 0.0
        start = time.time()
        
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
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            if epoch % 100 == 99:
                model.eval()
                with torch.no_grad():
                    pred = logits[:, -1, :].argmax(dim=-1)
                    acc = (pred == targets).float().mean().item()
                    best_acc = max(best_acc, acc)
                
                print(f"  Epoch {epoch+1}: loss={loss.item():.4f}, acc={acc:.1%}")
        
        elapsed = time.time() - start
        
        # Get final alpha stats
        model.eval()
        with torch.no_grad():
            _, diag = model(seq, return_diagnostics=True)
        
        alphas = [d['layer0_echo']['alpha'].mean().item() for d in diag]
        alpha_mean = sum(alphas) / len(alphas)
        
        results[n_heads] = {
            'acc': best_acc,
            'time': elapsed,
            'alpha_mean': alpha_mean,
            'd_head': 64 // n_heads,
        }
    
    print("\n" + "="*80)
    print("COMPARISON SUMMARY")
    print("="*80)
    print(f"{'n_heads':<10} {'d_head':<10} {'Best Acc':<12} {'Time':<10} {'α mean':<10}")
    print("-"*80)
    for n, r in results.items():
        print(f"{n:<10} {r['d_head']:<10} {r['acc']:5.1%}        {r['time']:<10.1f} {r['alpha_mean']:.3f}")


if __name__ == "__main__":
    test_gradient_flow()
    test_alpha_dynamics()
    # test_retrieval_task()
    # test_comparison_heads()

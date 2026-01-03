#!/usr/bin/env python3
"""
Deep Gradient Flow Analysis for RIN Architecture

This script conducts a thorough analysis of gradient flow through the RIN model,
focusing on:
1. Embedding gradients (w, b pairs) - do they receive meaningful gradients?
2. Euler transform gradient flow - how do gradients flow through sin/cos?
3. ResonantLayer gradient analysis
4. Echo Chamber Q/K/V gradient flow, entropy, and specialization
5. Complex state preservation vs information loss

Key hypothesis: The embedding states (w, b) which model time-invariant transforms
may not be receiving proper gradients.
"""

import argparse
import random
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from collections import defaultdict
import sys
sys.path.insert(0, '/home/aiman/pi')

from rin import PHI, get_global_lut
from rin.utils import wrap_time_periodic
from rin.model import ComplexLinear, ResonantLayer


# ============================================================
# Gradient Tracking Utilities
# ============================================================

class GradientTracker:
    """Track gradients across training for analysis."""
    
    def __init__(self):
        self.history = defaultdict(list)
        self.hooks = []
    
    def register_hook(self, module: nn.Module, name: str):
        """Register gradient hooks on a module."""
        def hook_fn(grad):
            if grad is not None:
                self.history[f"{name}_grad_norm"].append(grad.norm().item())
                self.history[f"{name}_grad_mean"].append(grad.mean().item())
                self.history[f"{name}_grad_std"].append(grad.std().item())
                self.history[f"{name}_grad_max"].append(grad.abs().max().item())
                zero_frac = (grad.abs() < 1e-8).float().mean().item()
                self.history[f"{name}_grad_zero_frac"].append(zero_frac)
        
        for pname, param in module.named_parameters():
            if param.requires_grad:
                handle = param.register_hook(
                    lambda g, n=f"{name}.{pname}": self._param_hook(g, n)
                )
                self.hooks.append(handle)
    
    def _param_hook(self, grad, name):
        if grad is not None:
            self.history[f"{name}_grad_norm"].append(grad.norm().item())
            self.history[f"{name}_grad_mean"].append(grad.mean().item())
            self.history[f"{name}_grad_std"].append(grad.std().item())
            if grad.dim() >= 2:
                per_dim_norm = grad.norm(dim=-1).mean().item()
                self.history[f"{name}_per_dim_norm"].append(per_dim_norm)
    
    def clear_hooks(self):
        for h in self.hooks:
            h.remove()
        self.hooks = []
    
    def get_summary(self) -> Dict:
        """Get summary statistics."""
        summary = {}
        for key, values in self.history.items():
            if values:
                summary[key] = {
                    'mean': sum(values) / len(values),
                    'min': min(values),
                    'max': max(values),
                    'last': values[-1],
                }
        return summary


@dataclass
class ActivationStats:
    """Statistics for activations at a specific point."""
    mean: float
    std: float
    min_val: float
    max_val: float
    dead_frac: float
    saturated_frac: float


def compute_activation_stats(x: torch.Tensor, saturation_thresh: float = 0.99) -> ActivationStats:
    """Compute activation statistics."""
    with torch.no_grad():
        return ActivationStats(
            mean=x.mean().item(),
            std=x.std().item(),
            min_val=x.min().item(),
            max_val=x.max().item(),
            dead_frac=(x.abs() < 1e-6).float().mean().item(),
            saturated_frac=(x.abs() > saturation_thresh).float().mean().item(),
        )


# ============================================================
# Instrumented Models for Gradient Analysis
# ============================================================

class InstrumentedEulerTransform(nn.Module):
    """
    Euler transform with full instrumentation for gradient analysis.
    
    SEPARATED theta computation preserving complex structure:
    theta_real = h_real / wavelength + b + t*phi
    theta_imag = h_imag / wavelength + b + t*phi
    
    Then complex multiplication of e^(i*theta_real) * e^(i*theta_imag)
    """
    
    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model
        self._lut = None
        
        self.last_theta = None
        self.last_wavelength = None
        self.last_sin = None
        self.last_cos = None
        self.grad_stats = {}
    
    def _get_lut(self, device):
        if self._lut is None or self._lut.sin_table.device != device:
            self._lut = get_global_lut(4096, device)
        return self._lut
    
    def forward(
        self, 
        h_real: torch.Tensor, 
        h_imag: torch.Tensor, 
        w: torch.Tensor, 
        b: torch.Tensor, 
        t: torch.Tensor,
        record_grads: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Separated euler transform preserving complex state.
        
        Args:
            h_real, h_imag: Previous complex state (batch, d_model)
            w, b: Token embeddings for this timestep (batch, d_model)
            t: Timestep scalar or (batch,)
        """
        lut = self._get_lut(h_real.device)
        
        # Wavelength from w embedding
        wavelength = 1.0 + w.abs()
        
        # Time phase
        t_phi = t * PHI if t.dim() == 0 else t.unsqueeze(-1) * PHI
        t_phi = wrap_time_periodic(t_phi)
        
        # SEPARATED theta computation (preserving complex structure)
        theta_real = h_real / wavelength + b + t_phi
        theta_imag = h_imag / wavelength + b + t_phi
        
        # Euler transform on each component
        sin_real, cos_real = lut.lookup_sin_cos(theta_real)
        sin_imag, cos_imag = lut.lookup_sin_cos(theta_imag)
        
        # Complex multiplication: e^(i*theta_real) * e^(i*theta_imag)
        h_real_new = cos_real * cos_imag - sin_real * sin_imag
        h_imag_new = cos_real * sin_imag + sin_real * cos_imag
        
        if record_grads:
            self.last_theta = (theta_real.detach(), theta_imag.detach())
            self.last_wavelength = wavelength.detach()
            self.last_sin = (sin_real.detach(), sin_imag.detach())
            self.last_cos = (cos_real.detach(), cos_imag.detach())
        
        return h_real_new, h_imag_new
    
    def analyze_gradient_flow(
        self,
        h_real: torch.Tensor,
        h_imag: torch.Tensor,
        w: torch.Tensor,
        b: torch.Tensor,
        t: torch.Tensor,
    ) -> Dict:
        """Compute and analyze gradients through the euler transform."""
        h_real = h_real.clone().requires_grad_(True)
        h_imag = h_imag.clone().requires_grad_(True)
        w = w.clone().requires_grad_(True)
        b = b.clone().requires_grad_(True)
        
        h_real_new, h_imag_new = self.forward(h_real, h_imag, w, b, t, record_grads=True)
        
        loss = (h_real_new.sum() + h_imag_new.sum())
        loss.backward()
        
        stats = {
            'h_real_grad': compute_activation_stats(h_real.grad) if h_real.grad is not None else None,
            'h_imag_grad': compute_activation_stats(h_imag.grad) if h_imag.grad is not None else None,
            'w_grad': compute_activation_stats(w.grad) if w.grad is not None else None,
            'b_grad': compute_activation_stats(b.grad) if b.grad is not None else None,
            'theta_real_stats': compute_activation_stats(self.last_theta[0]),
            'theta_imag_stats': compute_activation_stats(self.last_theta[1]),
            'wavelength_stats': compute_activation_stats(self.last_wavelength),
        }
        
        if h_real.grad is not None and w.grad is not None:
            stats['h_to_w_grad_ratio'] = h_real.grad.norm().item() / (w.grad.norm().item() + 1e-8)
            stats['h_to_b_grad_ratio'] = h_real.grad.norm().item() / (b.grad.norm().item() + 1e-8)
        
        return stats


class InstrumentedResonantLayer(nn.Module):
    """ResonantLayer with gradient instrumentation."""
    
    def __init__(self, d_model: int, num_neurons: int, use_swish: bool = True):
        super().__init__()
        self.d_model = d_model
        self.num_neurons = num_neurons
        self.use_swish = use_swish
        
        self.input_proj = ComplexLinear(d_model, d_model, bias=True)
        
        self.W = nn.Parameter(torch.randn(num_neurons, d_model) * 0.02)
        self.bias = nn.Parameter(torch.zeros(num_neurons))
        
        self.proj_real_to_real = nn.Linear(num_neurons, d_model, bias=False)
        self.proj_real_to_imag = nn.Linear(num_neurons, d_model, bias=False)
        self.proj_imag_to_real = nn.Linear(num_neurons, d_model, bias=False)
        self.proj_imag_to_imag = nn.Linear(num_neurons, d_model, bias=False)
        
        self._lut = None
        self.last_activations = {}
    
    def _get_lut(self, device):
        if self._lut is None or self._lut.sin_table.device != device:
            self._lut = get_global_lut(4096, device)
        return self._lut
    
    def forward(
        self, 
        x_real: torch.Tensor, 
        x_imag: torch.Tensor, 
        t: torch.Tensor,
        record: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        lut = self._get_lut(x_real.device)
        
        proj_real, proj_imag = self.input_proj(x_real, x_imag)
        x_collapsed = proj_real + proj_imag
        
        if record:
            self.last_activations['input_real'] = x_real.detach()
            self.last_activations['input_imag'] = x_imag.detach()
            self.last_activations['proj_real'] = proj_real.detach()
            self.last_activations['proj_imag'] = proj_imag.detach()
            self.last_activations['collapsed'] = x_collapsed.detach()
        
        wavelength = 1.0 + self.W.abs()
        if t.dim() == 1:
            t = t.unsqueeze(-1)
        t = wrap_time_periodic(t)
        
        theta = x_collapsed @ (1.0 / wavelength).T + self.bias + t
        
        if record:
            self.last_activations['theta'] = theta.detach()
        
        sin_theta, cos_theta = lut.lookup_sin_cos(theta)
        
        if record:
            self.last_activations['sin'] = sin_theta.detach()
            self.last_activations['cos'] = cos_theta.detach()
        
        out_real = self.proj_real_to_real(cos_theta) + self.proj_imag_to_real(sin_theta)
        out_imag = self.proj_real_to_imag(cos_theta) + self.proj_imag_to_imag(sin_theta)
        
        if self.use_swish:
            out_real = F.silu(out_real)
            out_imag = F.silu(out_imag)
        
        if record:
            self.last_activations['out_real'] = out_real.detach()
            self.last_activations['out_imag'] = out_imag.detach()
        
        return out_real, out_imag


class InstrumentedEchoChamber(nn.Module):
    """Echo Chamber with full instrumentation for Q/K/V analysis."""
    
    def __init__(
        self,
        d_model: int,
        n_heads: int = 4,
        alpha: float = 1.0,
        output_mode: str = 'complex_linear',
    ):
        super().__init__()
        assert d_model % n_heads == 0
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.output_mode = output_mode
        self.scale = math.sqrt(self.d_head)
        
        self.input_proj = ComplexLinear(d_model, d_model, bias=True)
        self.prenorm = nn.LayerNorm(d_model)
        
        self.W_qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        
        alpha_clamped = max(0.001, min(0.999, alpha))
        self.log_alpha = nn.Parameter(
            torch.zeros(n_heads) + math.log(alpha_clamped / (1 - alpha_clamped))
        )
        
        self.output_proj = ComplexLinear(d_model, d_model, bias=False)
        
        self._lut = None
        
        self.last_q = None
        self.last_k = None
        self.last_v = None
        self.last_attention_scores = None
        self.last_retrieved = None
        self.attention_entropy_history = []
        self.value_state_norm_history = []
        self.qkv_grad_history = defaultdict(list)
    
    def _get_lut(self, device):
        if self._lut is None or self._lut.sin_table.device != device:
            self._lut = get_global_lut(4096, device)
        return self._lut
    
    @property
    def alpha(self):
        return torch.sigmoid(self.log_alpha)
    
    def init_state(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.zeros(batch_size, self.n_heads, self.d_head, device=device)
    
    def forward_step(
        self,
        x_real: torch.Tensor,
        x_imag: torch.Tensor,
        value_state: torch.Tensor,
        t: torch.Tensor,
        record: bool = False,
    ) -> Tuple[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        batch_size = x_real.shape[0]
        device = x_real.device
        
        proj_real, proj_imag = self.input_proj(x_real, x_imag)
        x_collapsed = proj_real + proj_imag
        x_collapsed = self.prenorm(x_collapsed)
        
        qkv = self.W_qkv(x_collapsed)
        qkv = qkv.view(batch_size, 3, self.n_heads, self.d_head)
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]
        
        if record:
            self.last_q = q.detach()
            self.last_k = k.detach()
            self.last_v = v.detach()
        
        alpha = self.alpha.view(1, self.n_heads, 1)
        new_value_state = alpha * v + (1 - alpha) * value_state
        
        if record:
            self.value_state_norm_history.append(new_value_state.norm(dim=-1).mean().item())
        
        scores = torch.sum(q * k, dim=-1, keepdim=True) / self.scale
        gate = torch.sigmoid(scores)
        
        if record:
            self.last_attention_scores = scores.detach()
            p = gate.detach().squeeze(-1)
            entropy = -p * torch.log(p + 1e-8) - (1 - p) * torch.log(1 - p + 1e-8)
            self.attention_entropy_history.append(entropy.mean().item())
        
        retrieved = gate * new_value_state
        retrieved = retrieved.view(batch_size, self.d_model)
        
        if record:
            self.last_retrieved = retrieved.detach()
        
        out_real, out_imag = self.output_proj(retrieved, torch.zeros_like(retrieved))
        
        return (out_real, out_imag), new_value_state
    
    def get_analysis(self) -> Dict:
        """Get analysis of Q/K/V behavior."""
        analysis = {}
        
        if self.last_q is not None:
            analysis['q_stats'] = compute_activation_stats(self.last_q)
            analysis['k_stats'] = compute_activation_stats(self.last_k)
            analysis['v_stats'] = compute_activation_stats(self.last_v)
            
            qk_sim = F.cosine_similarity(
                self.last_q.reshape(-1, self.d_head),
                self.last_k.reshape(-1, self.d_head),
                dim=-1
            )
            analysis['qk_cosine_sim'] = {
                'mean': qk_sim.mean().item(),
                'std': qk_sim.std().item(),
            }
        
        if self.attention_entropy_history:
            analysis['attention_entropy'] = {
                'mean': sum(self.attention_entropy_history) / len(self.attention_entropy_history),
                'min': min(self.attention_entropy_history),
                'max': max(self.attention_entropy_history),
                'last': self.attention_entropy_history[-1],
            }
        
        if self.value_state_norm_history:
            analysis['value_state_norm'] = {
                'mean': sum(self.value_state_norm_history) / len(self.value_state_norm_history),
                'min': min(self.value_state_norm_history),
                'max': max(self.value_state_norm_history),
                'last': self.value_state_norm_history[-1],
            }
        
        return analysis


# ============================================================
# Full Instrumented Model
# ============================================================

class InstrumentedRIN(nn.Module):
    """
    Fully instrumented RIN for gradient flow analysis.
    
    Maintains COMPLEX STATE throughout (no collapse except final output).
    NO RESIDUALS - pure rotary phase flow.
    """
    
    def __init__(
        self,
        vocab_size: int,
        num_signals: int,
        d_model: int = 64,
        num_layers: int = 2,
        num_neurons: int = 128,
        n_heads: int = 4,
        alpha: float = 1.0,
        use_echo: bool = True,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.num_signals = num_signals
        self.d_model = d_model
        self.use_echo = use_echo
        
        self.token_embedding = nn.Embedding(vocab_size, 2 * d_model)
        
        self.euler = InstrumentedEulerTransform(d_model)
        
        self.layers = nn.ModuleList([
            InstrumentedResonantLayer(d_model, num_neurons, use_swish=True)
            for _ in range(num_layers)
        ])
        
        if use_echo:
            self.echo = InstrumentedEchoChamber(
                d_model=d_model,
                n_heads=n_heads,
                alpha=alpha,
                output_mode='complex_linear',
            )
        else:
            self.echo = None
        
        self.output_proj = ComplexLinear(d_model, num_signals, bias=False)
        
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        with torch.no_grad():
            self.token_embedding.weight.mul_(0.5)
        
        self.grad_tracker = GradientTracker()
        self.embedding_grad_history = []
        self.record_mode = False
    
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        
        h_real = torch.zeros(batch_size, self.d_model, device=device)
        h_imag = torch.zeros(batch_size, self.d_model, device=device)
        
        if self.echo is not None:
            echo_state = self.echo.init_state(batch_size, device)
        
        embeddings = self.token_embedding(input_ids)
        w_emb = embeddings[:, :, :self.d_model]
        b_emb = embeddings[:, :, self.d_model:]
        
        t_indices = torch.arange(seq_len, device=device, dtype=torch.float32)
        
        for t in range(seq_len):
            t_val = t_indices[t]
            
            # Euler transform (preserving complex state)
            h_real, h_imag = self.euler(
                h_real, h_imag,
                w_emb[:, t], b_emb[:, t],
                t_val,
                record_grads=self.record_mode
            )
            
            # Resonant layers (NO RESIDUALS - pure transformation)
            x_real, x_imag = h_real, h_imag
            for layer in self.layers:
                x_real, x_imag = layer(x_real, x_imag, t_val * PHI, record=self.record_mode)
            
            # Echo chamber (if enabled)
            if self.echo is not None:
                (echo_real, echo_imag), echo_state = self.echo.forward_step(
                    x_real, x_imag, echo_state, t_val * PHI, record=self.record_mode
                )
                # Additive interference
                x_real = x_real + echo_real
                x_imag = x_imag + echo_imag
        
        # Final output: complex -> logits (only place we collapse)
        logits_real, logits_imag = self.output_proj(x_real, x_imag)
        return logits_real + logits_imag
    
    def analyze_embedding_gradients(self) -> Dict:
        """Analyze gradients flowing to embeddings."""
        if self.token_embedding.weight.grad is None:
            return {'error': 'No gradients computed yet'}
        
        grad = self.token_embedding.weight.grad
        
        w_grad = grad[:, :self.d_model]
        b_grad = grad[:, self.d_model:]
        
        analysis = {
            'total_grad_norm': grad.norm().item(),
            'w_grad': {
                'norm': w_grad.norm().item(),
                'mean': w_grad.mean().item(),
                'std': w_grad.std().item(),
                'max': w_grad.abs().max().item(),
                'dead_frac': (w_grad.abs() < 1e-8).float().mean().item(),
            },
            'b_grad': {
                'norm': b_grad.norm().item(),
                'mean': b_grad.mean().item(),
                'std': b_grad.std().item(),
                'max': b_grad.abs().max().item(),
                'dead_frac': (b_grad.abs() < 1e-8).float().mean().item(),
            },
            'w_to_b_ratio': w_grad.norm().item() / (b_grad.norm().item() + 1e-8),
        }
        
        per_token_norm = grad.norm(dim=-1)
        analysis['per_token'] = {
            'norm_mean': per_token_norm.mean().item(),
            'norm_std': per_token_norm.std().item(),
            'norm_max': per_token_norm.max().item(),
            'norm_min': per_token_norm.min().item(),
        }
        
        return analysis


# ============================================================
# Dataset
# ============================================================

class NeedleDataset(Dataset):
    """Simple needle-in-haystack dataset."""
    
    def __init__(self, num_samples, num_signals, min_dist, max_dist, num_noise, seed=42):
        self.trigger_id = 0
        self.signal_start = 1
        self.noise_start = num_signals + 1
        self.vocab_size = num_signals + num_noise + 1
        
        random.seed(seed)
        self.data = []
        
        for _ in range(num_samples):
            signal = random.randint(0, num_signals - 1)
            dist = random.randint(min_dist, max_dist)
            
            seq = [self.trigger_id, self.signal_start + signal]
            for _ in range(dist):
                seq.append(random.randint(self.noise_start, self.vocab_size - 1))
            seq.append(self.trigger_id)
            
            self.data.append((torch.tensor(seq, dtype=torch.long), signal, dist))
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]


def collate_fn(batch):
    seqs, signals, dists = zip(*batch)
    max_len = max(len(s) for s in seqs)
    padded = torch.zeros(len(seqs), max_len, dtype=torch.long)
    for i, s in enumerate(seqs):
        padded[i, :len(s)] = s
    return padded, torch.tensor(signals), torch.tensor(dists)


# ============================================================
# Main Analysis
# ============================================================

def run_analysis(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    dataset = NeedleDataset(
        num_samples=1000,
        num_signals=10,
        min_dist=1,
        max_dist=args.max_distance,
        num_noise=50,
    )
    loader = DataLoader(dataset, batch_size=64, shuffle=True, collate_fn=collate_fn)
    
    print(f"\nVocab: {dataset.vocab_size}, Signals: 10, Distance: 1-{args.max_distance}")
    
    model = InstrumentedRIN(
        vocab_size=dataset.vocab_size,
        num_signals=10,
        d_model=64,
        num_layers=2,
        num_neurons=128,
        n_heads=4,
        alpha=args.alpha,
        use_echo=args.use_echo,
    ).to(device)
    
    model_name = "RIN + Echo" if args.use_echo else "Vanilla RIN"
    print(f"\nModel: {model_name}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    
    all_embedding_grads = []
    all_layer_grads = []
    all_echo_analysis = []
    losses = []
    accuracies = []
    
    print("\n" + "="*70)
    print("GRADIENT FLOW ANALYSIS")
    print("="*70)
    
    model.record_mode = True
    
    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, (seqs, signals, dists) in enumerate(loader):
            seqs = seqs.to(device)
            signals = signals.to(device)
            
            optimizer.zero_grad()
            logits = model(seqs)
            loss = F.cross_entropy(logits, signals)
            loss.backward()
            
            if batch_idx == 0:
                emb_analysis = model.analyze_embedding_gradients()
                all_embedding_grads.append(emb_analysis)
                
                layer_grads = {}
                for i, layer in enumerate(model.layers):
                    if layer.W.grad is not None:
                        layer_grads[f'layer{i}_W'] = {
                            'norm': layer.W.grad.norm().item(),
                            'mean': layer.W.grad.mean().item(),
                        }
                    # Check input_proj weight grad
                    for pname, param in layer.input_proj.named_parameters():
                        if param.grad is not None:
                            layer_grads[f'layer{i}_input_proj_{pname}'] = {
                                'norm': param.grad.norm().item(),
                            }
                            break  # Just get one for brevity
                all_layer_grads.append(layer_grads)
                
                if model.echo is not None:
                    echo_analysis = model.echo.get_analysis()
                    all_echo_analysis.append(echo_analysis)
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            epoch_loss += loss.item()
            correct += (logits.argmax(-1) == signals).sum().item()
            total += signals.size(0)
        
        losses.append(epoch_loss / len(loader))
        accuracies.append(correct / total)
        
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"\nEpoch {epoch+1}/{args.epochs}")
            print(f"  Loss: {losses[-1]:.4f}, Acc: {accuracies[-1]*100:.1f}%")
            
            emb = all_embedding_grads[-1]
            if 'error' not in emb:
                print(f"\n  EMBEDDING GRADIENTS:")
                print(f"    Total norm: {emb['total_grad_norm']:.6f}")
                print(f"    W (wavelength) grad norm: {emb['w_grad']['norm']:.6f}")
                print(f"    B (bias) grad norm: {emb['b_grad']['norm']:.6f}")
                print(f"    W/B ratio: {emb['w_to_b_ratio']:.4f}")
                print(f"    W dead fraction: {emb['w_grad']['dead_frac']*100:.1f}%")
                print(f"    B dead fraction: {emb['b_grad']['dead_frac']*100:.1f}%")
            
            if all_layer_grads:
                print(f"\n  LAYER GRADIENTS:")
                for key, val in all_layer_grads[-1].items():
                    print(f"    {key}: norm={val['norm']:.6f}")
            
            if all_echo_analysis and model.echo is not None:
                ea = all_echo_analysis[-1]
                print(f"\n  ECHO CHAMBER ANALYSIS:")
                if 'q_stats' in ea:
                    print(f"    Q mean: {ea['q_stats'].mean:.4f}, std: {ea['q_stats'].std:.4f}")
                    print(f"    K mean: {ea['k_stats'].mean:.4f}, std: {ea['k_stats'].std:.4f}")
                    print(f"    V mean: {ea['v_stats'].mean:.4f}, std: {ea['v_stats'].std:.4f}")
                if 'qk_cosine_sim' in ea:
                    print(f"    Q-K cosine sim: {ea['qk_cosine_sim']['mean']:.4f} ± {ea['qk_cosine_sim']['std']:.4f}")
                if 'attention_entropy' in ea:
                    print(f"    Attention entropy: {ea['attention_entropy']['mean']:.4f}")
                if 'value_state_norm' in ea:
                    print(f"    Value state norm: {ea['value_state_norm']['last']:.4f}")
    
    print("\n" + "="*70)
    print("FINAL DETAILED ANALYSIS")
    print("="*70)
    
    # Analyze euler transform specifically
    print("\n--- EULER TRANSFORM GRADIENT FLOW ---")
    model.eval()
    batch = next(iter(loader))
    seqs, signals, dists = [x.to(device) for x in batch]
    
    embeddings = model.token_embedding(seqs)
    w_emb = embeddings[:, :, :model.d_model]
    b_emb = embeddings[:, :, model.d_model:]
    
    h_real = torch.zeros(seqs.shape[0], model.d_model, device=device)
    h_imag = torch.zeros(seqs.shape[0], model.d_model, device=device)
    
    euler_analysis = model.euler.analyze_gradient_flow(
        h_real, h_imag, w_emb[:, 0], b_emb[:, 0], torch.tensor(0.0, device=device)
    )
    
    print(f"\nEuler Transform (t=0, h_real=0, h_imag=0):")
    print(f"  Theta (real) - mean: {euler_analysis['theta_real_stats'].mean:.4f}, std: {euler_analysis['theta_real_stats'].std:.4f}")
    print(f"  Theta (imag) - mean: {euler_analysis['theta_imag_stats'].mean:.4f}, std: {euler_analysis['theta_imag_stats'].std:.4f}")
    print(f"  Wavelength - mean: {euler_analysis['wavelength_stats'].mean:.4f}, range: [{euler_analysis['wavelength_stats'].min_val:.4f}, {euler_analysis['wavelength_stats'].max_val:.4f}]")
    
    if euler_analysis['h_real_grad'] is not None:
        print(f"\n  Gradients to inputs:")
        print(f"    h_real grad norm: {euler_analysis['h_real_grad'].std:.6f}")
        print(f"    h_imag grad norm: {euler_analysis['h_imag_grad'].std:.6f}")
        if euler_analysis['w_grad'] is not None:
            print(f"    w (wavelength) grad norm: {euler_analysis['w_grad'].std:.6f}")
            print(f"    b (bias) grad norm: {euler_analysis['b_grad'].std:.6f}")
            print(f"    h/w ratio: {euler_analysis.get('h_to_w_grad_ratio', 'N/A')}")
            print(f"    h/b ratio: {euler_analysis.get('h_to_b_grad_ratio', 'N/A')}")
        else:
            print(f"    ⚠️  w and b are NON-LEAF TENSORS - grads not captured!")
            print(f"    This means embeddings slice views don't retain gradients directly.")
    
    print("\n--- TOKEN-LEVEL GRADIENT ANALYSIS ---")
    
    model.train()
    model.zero_grad()
    logits = model(seqs)
    loss = F.cross_entropy(logits, signals)
    loss.backward()
    
    emb_grad = model.token_embedding.weight.grad
    if emb_grad is not None:
        trigger_grad = emb_grad[0]
        print(f"Trigger token (id=0) grad norm: {trigger_grad.norm().item():.6f}")
        
        signal_grads = emb_grad[1:11]
        print(f"Signal tokens (1-10) grad norm: {signal_grads.norm().item():.6f}")
        print(f"  Per-token: {[f'{x:.4f}' for x in signal_grads.norm(dim=-1).tolist()]}")
        
        noise_grads = emb_grad[11:]
        print(f"Noise tokens (11+) grad norm: {noise_grads.norm().item():.6f}")
        
        print(f"\nW vs B breakdown:")
        print(f"  Trigger - W: {trigger_grad[:model.d_model].norm().item():.6f}, B: {trigger_grad[model.d_model:].norm().item():.6f}")
        print(f"  Signal avg - W: {signal_grads[:, :model.d_model].norm().item()/10:.6f}, B: {signal_grads[:, model.d_model:].norm().item()/10:.6f}")
    
    print("\n--- TEMPORAL GRADIENT FLOW ---")
    print("(Checking if early timesteps receive gradients from late timesteps)")
    
    # Critical test: gradient flow through euler chain
    # Fresh forward pass with retain_grad
    model.eval()
    model.zero_grad()
    
    # Get fresh embeddings
    batch = next(iter(loader))
    seqs, signals, dists = [x.to(device) for x in batch]
    embeddings = model.token_embedding(seqs)
    w_emb_fresh = embeddings[:, :, :model.d_model]
    b_emb_fresh = embeddings[:, :, model.d_model:]
    
    # Test with non-zero initial state to ensure gradients flow
    h_real = torch.randn(1, model.d_model, device=device, requires_grad=True) * 0.1
    h_imag = torch.randn(1, model.d_model, device=device, requires_grad=True) * 0.1
    
    # Store initial for checking
    init_h_real = h_real.clone()
    init_h_imag = h_imag.clone()
    init_h_real.retain_grad()
    init_h_imag.retain_grad()
    
    # Forward through euler chain
    h_r, h_i = h_real, h_imag
    for t in range(5):
        h_r, h_i = model.euler(
            h_r, h_i,
            w_emb_fresh[0:1, t].detach(),  # Detach to isolate euler gradient flow
            b_emb_fresh[0:1, t].detach(),
            torch.tensor(float(t), device=device)
        )
    
    # Backward from final state
    final_norm = h_r.norm() + h_i.norm()
    final_norm.backward()
    
    print(f"\nAfter 5 euler steps (with non-zero init):")
    print(f"  Final state norm: {final_norm.item():.4f}")
    if h_real.grad is not None:
        print(f"  Initial h_real grad: {h_real.grad.norm().item():.6f}")
        print(f"  Initial h_imag grad: {h_imag.grad.norm().item():.6f}")
    else:
        print(f"  Initial h_real grad: None")
        print(f"  Initial h_imag grad: None")
    
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)
    
    print("\n--- SUMMARY ---")
    print(f"Final accuracy: {accuracies[-1]*100:.1f}%")
    print(f"Final loss: {losses[-1]:.4f}")
    
    if all_embedding_grads:
        final_emb = all_embedding_grads[-1]
        if 'error' not in final_emb:
            print(f"\nEmbedding health:")
            print(f"  W gradient norm: {final_emb['w_grad']['norm']:.6f}")
            print(f"  B gradient norm: {final_emb['b_grad']['norm']:.6f}")
            if final_emb['w_grad']['dead_frac'] > 0.5:
                print(f"  ⚠️  WARNING: >50% of W gradients are near zero!")
            if final_emb['b_grad']['dead_frac'] > 0.5:
                print(f"  ⚠️  WARNING: >50% of B gradients are near zero!")
    
    print("\n--- POTENTIAL ISSUES ---")
    issues = []
    
    if accuracies[-1] < 0.15:
        issues.append("Model not learning (stuck at random chance)")
    
    if all_embedding_grads and 'error' not in all_embedding_grads[-1]:
        if all_embedding_grads[-1]['w_grad']['norm'] < 1e-6:
            issues.append("W (wavelength) embeddings receiving near-zero gradients")
        if all_embedding_grads[-1]['b_grad']['norm'] < 1e-6:
            issues.append("B (bias) embeddings receiving near-zero gradients")
    
    if all_echo_analysis and model.echo is not None:
        if all_echo_analysis[-1].get('attention_entropy', {}).get('mean', 0) > 0.6:
            issues.append("Echo attention has high entropy (not specializing)")
    
    if issues:
        for issue in issues:
            print(f"  ⚠️  {issue}")
    else:
        print("  No obvious issues detected")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--max_distance', type=int, default=10)
    parser.add_argument('--alpha', type=float, default=1.0)
    parser.add_argument('--use_echo', action='store_true', default=True)
    parser.add_argument('--no_echo', dest='use_echo', action='store_false')
    args = parser.parse_args()
    
    run_analysis(args)

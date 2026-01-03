#!/usr/bin/env python3
"""
Visualize RIN Embeddings as Quantum Wavefunctions

Each token embedding (w, b) defines a wave through Euler's formula:
    θ = h / (1 + |w|) + b + t·φ
    Ψ = cos(θ) + i·sin(θ)

This creates a complex wavefunction evolving over time - literally a
Schrödinger wave on the unit circle.

Visualization methods:
1. Parametric Helix - (t, Re(Ψ), Im(Ψ)) for each dimension
2. Probability Density Cloud - |Ψ|² volumetric visualization  
3. Phase-Magnitude Mapping - color=phase, size=magnitude
4. Wavelength Spectrum - distribution of learned frequencies
5. Token Comparison Grid - before/after training comparison

Usage:
    python visualize_embeddings.py
    python visualize_embeddings.py --epochs 200 --save_dir results/viz
"""

import argparse
import math
import random
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize, hsv_to_rgb
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.gridspec as gridspec

from rin import get_global_lut, PHI

# Set style
plt.style.use('dark_background')
plt.rcParams['figure.facecolor'] = '#0a0a0a'
plt.rcParams['axes.facecolor'] = '#0a0a0a'
plt.rcParams['axes.edgecolor'] = '#333333'
plt.rcParams['grid.color'] = '#222222'


# ============================================================================
# MODEL (same as train_modular.py)
# ============================================================================

class ModularRIN(nn.Module):
    """RIN for modular arithmetic with accessible embeddings."""
    
    def __init__(self, vocab_size, d_model=48, num_layers=2, num_neurons=96, use_swish=True):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        
        self.token_embedding = nn.Embedding(vocab_size, 2 * d_model)
        self.layers = nn.ModuleList([
            ModularResonantLayer(d_model, num_neurons, use_swish=use_swish)
            for _ in range(num_layers)
        ])
        self.output_proj = nn.Linear(d_model, vocab_size, bias=False)
        
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.output_proj.weight, std=0.02)
        with torch.no_grad():
            self.token_embedding.weight.mul_(0.5)
        
        self._lut = None
    
    def _get_lut(self, device):
        if self._lut is None or self._lut.sin_table.device != device:
            self._lut = get_global_lut(4096, device)
        return self._lut
    
    def get_wb_embeddings(self, token_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get (w, b) pairs for tokens."""
        emb = self.token_embedding(token_ids)
        w = emb[:, :self.d_model]
        b = emb[:, self.d_model:]
        return w, b
    
    def forward(self, input_ids):
        lut = self._get_lut(input_ids.device)
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        
        h_real = torch.zeros(batch_size, self.d_model, device=device)
        h_imag = torch.zeros(batch_size, self.d_model, device=device)
        
        embeddings = self.token_embedding(input_ids)
        w_emb = embeddings[:, :, :self.d_model]
        b_emb = embeddings[:, :, self.d_model:]
        
        # Pre-compute timestep tensors
        t_indices = torch.arange(seq_len, device=device, dtype=torch.float32) * PHI
        
        for t in range(seq_len):
            t_val = t_indices[t]
            wavelength = 1.0 + w_emb[:, t, :].abs()
            
            h_combined = h_real + h_imag
            theta = h_combined / wavelength + b_emb[:, t, :] + t_val
            
            h_imag, h_real = lut.lookup_sin_cos(theta)
            
            h = h_real + h_imag
            for layer in self.layers:
                h = h + layer(h, t_val)
        
        return self.output_proj(h)


class ModularResonantLayer(nn.Module):
    def __init__(self, d_model, num_neurons, use_swish=True):
        super().__init__()
        self.use_swish = use_swish
        self.W = nn.Parameter(torch.randn(num_neurons, d_model) * 0.02)
        self.bias = nn.Parameter(torch.zeros(num_neurons))
        self.proj_real = nn.Linear(num_neurons, d_model, bias=False)
        self.proj_imag = nn.Linear(num_neurons, d_model, bias=False)
        self._lut = None
    
    def _get_lut(self, device):
        if self._lut is None or self._lut.sin_table.device != device:
            self._lut = get_global_lut(4096, device)
        return self._lut
    
    def forward(self, x, t):
        lut = self._get_lut(x.device)
        theta = x @ self.W.T + self.bias + t
        sin_theta, cos_theta = lut.lookup_sin_cos(theta)
        out = self.proj_real(cos_theta) + self.proj_imag(sin_theta)
        return F.silu(out) if self.use_swish else out


# ============================================================================
# DATASET
# ============================================================================

class ModularAdditionDataset(Dataset):
    def __init__(self, p: int, split: str = "train", train_frac: float = 0.5, seed: int = 42):
        self.p = p
        self.data = []
        all_pairs = [(a, b) for a in range(p) for b in range(p)]
        random.seed(seed)
        random.shuffle(all_pairs)
        split_idx = int(len(all_pairs) * train_frac)
        pairs = all_pairs[:split_idx] if split == "train" else all_pairs[split_idx:]
        for a, b in pairs:
            self.data.append((torch.tensor([a, b, p]), (a + b) % p))
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]


# ============================================================================
# WAVEFUNCTION COMPUTATION
# ============================================================================

def compute_wavefunction(w: np.ndarray, b: np.ndarray, t_range: np.ndarray, 
                         h_init: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute the wavefunction Ψ(t) for a single (w, b) pair.
    
    θ(t) = h / (1 + |w|) + b + t·φ
    Ψ(t) = cos(θ) + i·sin(θ)
    
    Returns: (real_part, imag_part) arrays over t
    """
    wavelength = 1.0 + np.abs(w)
    theta = h_init / wavelength + b + t_range * PHI
    return np.cos(theta), np.sin(theta)


def compute_all_wavefunctions(model: nn.Module, token_ids: List[int], 
                               t_range: np.ndarray) -> Dict[int, Dict]:
    """Compute wavefunctions for all specified tokens across all dimensions."""
    model.eval()
    device = next(model.parameters()).device
    
    results = {}
    with torch.no_grad():
        for token_id in token_ids:
            token_tensor = torch.tensor([token_id], device=device)
            w, b = model.get_wb_embeddings(token_tensor)
            w = w[0].cpu().numpy()
            b = b[0].cpu().numpy()
            
            # Compute wavefunction for each dimension
            wavefunctions = []
            for dim in range(len(w)):
                real, imag = compute_wavefunction(w[dim], b[dim], t_range)
                wavefunctions.append({
                    'w': w[dim],
                    'b': b[dim],
                    'real': real,
                    'imag': imag,
                    'magnitude': np.sqrt(real**2 + imag**2),
                    'phase': np.arctan2(imag, real),
                })
            
            results[token_id] = {
                'w_full': w,
                'b_full': b,
                'wavelengths': 1.0 + np.abs(w),
                'wavefunctions': wavefunctions,
            }
    
    return results


# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def plot_parametric_helix(ax, real: np.ndarray, imag: np.ndarray, t: np.ndarray,
                          color='cyan', alpha=0.8, linewidth=1.5, label=None):
    """Plot a single parametric helix (t, Re(Ψ), Im(Ψ))."""
    # Color by time progression
    points = np.array([t, real, imag]).T.reshape(-1, 1, 3)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    
    from mpl_toolkits.mplot3d.art3d import Line3DCollection
    norm = Normalize(t.min(), t.max())
    colors = cm.plasma(norm(t[:-1]))
    
    lc = Line3DCollection(segments, colors=colors, linewidth=linewidth, alpha=alpha)
    ax.add_collection(lc)
    
    ax.set_xlabel('Time (t)', fontsize=10, color='white')
    ax.set_ylabel('Re(Ψ)', fontsize=10, color='white')
    ax.set_zlabel('Im(Ψ)', fontsize=10, color='white')


def plot_helix_grid(wavefunctions: Dict[int, Dict], t_range: np.ndarray, 
                    token_labels: Dict[int, str], dims_to_show: List[int],
                    title: str = "Parametric Helix Visualization") -> plt.Figure:
    """Plot helix for multiple tokens and dimensions."""
    n_tokens = len(wavefunctions)
    n_dims = len(dims_to_show)
    
    fig = plt.figure(figsize=(4 * n_dims, 4 * n_tokens))
    fig.suptitle(title, fontsize=16, color='white', y=1.02)
    
    for i, (token_id, data) in enumerate(wavefunctions.items()):
        for j, dim in enumerate(dims_to_show):
            ax = fig.add_subplot(n_tokens, n_dims, i * n_dims + j + 1, projection='3d')
            
            wf = data['wavefunctions'][dim]
            plot_parametric_helix(ax, wf['real'], wf['imag'], t_range)
            
            ax.set_title(f"{token_labels.get(token_id, str(token_id))} | dim {dim}\n"
                        f"w={wf['w']:.3f}, b={wf['b']:.3f}", 
                        fontsize=9, color='white')
            ax.set_xlim(t_range.min(), t_range.max())
            ax.set_ylim(-1.2, 1.2)
            ax.set_zlim(-1.2, 1.2)
            
            # Style
            ax.xaxis.pane.fill = False
            ax.yaxis.pane.fill = False
            ax.zaxis.pane.fill = False
            ax.xaxis.pane.set_edgecolor('#333333')
            ax.yaxis.pane.set_edgecolor('#333333')
            ax.zaxis.pane.set_edgecolor('#333333')
    
    plt.tight_layout()
    return fig


def plot_phase_magnitude_cloud(wavefunctions: Dict[int, Dict], t_range: np.ndarray,
                               token_labels: Dict[int, str],
                               title: str = "Phase-Magnitude Cloud") -> plt.Figure:
    """Volumetric visualization with phase as color, magnitude as size."""
    n_tokens = len(wavefunctions)
    fig = plt.figure(figsize=(6 * min(n_tokens, 4), 5 * ((n_tokens - 1) // 4 + 1)))
    fig.suptitle(title, fontsize=16, color='white', y=1.02)
    
    for idx, (token_id, data) in enumerate(wavefunctions.items()):
        ax = fig.add_subplot(((n_tokens - 1) // 4 + 1), min(n_tokens, 4), idx + 1, projection='3d')
        
        # Aggregate across dimensions
        all_phases = []
        all_mags = []
        all_t = []
        all_dims = []
        
        n_dims = len(data['wavefunctions'])
        sample_dims = np.linspace(0, n_dims - 1, min(n_dims, 16), dtype=int)
        
        for dim in sample_dims:
            wf = data['wavefunctions'][dim]
            # Subsample time for performance
            t_sample = t_range[::4]
            phase_sample = wf['phase'][::4]
            mag_sample = wf['magnitude'][::4]
            
            all_t.extend(t_sample)
            all_dims.extend([dim] * len(t_sample))
            all_phases.extend(phase_sample)
            all_mags.extend(mag_sample)
        
        all_t = np.array(all_t)
        all_dims = np.array(all_dims)
        all_phases = np.array(all_phases)
        all_mags = np.array(all_mags)
        
        # Map phase to HSV color (cyclic)
        hue = (all_phases + np.pi) / (2 * np.pi)  # [0, 1]
        saturation = np.ones_like(hue) * 0.9
        value = np.ones_like(hue) * 0.9
        hsv = np.stack([hue, saturation, value], axis=-1)
        colors = hsv_to_rgb(hsv)
        
        # Size based on magnitude
        sizes = 20 * all_mags ** 2
        
        scatter = ax.scatter(all_t, all_dims, all_mags, 
                            c=colors, s=sizes, alpha=0.6, edgecolors='none')
        
        ax.set_xlabel('Time', fontsize=9, color='white')
        ax.set_ylabel('Dimension', fontsize=9, color='white')
        ax.set_zlabel('|Ψ|', fontsize=9, color='white')
        ax.set_title(token_labels.get(token_id, str(token_id)), fontsize=11, color='white')
        
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
    
    plt.tight_layout()
    return fig


def plot_wavelength_spectrum(wavefunctions: Dict[int, Dict], 
                             token_labels: Dict[int, str],
                             title: str = "Wavelength Spectrum") -> plt.Figure:
    """Distribution of learned wavelengths across tokens."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(title, fontsize=16, color='white', y=1.02)
    
    # 1. Wavelength distribution per token (violin plot simulation)
    ax1 = axes[0, 0]
    token_ids = list(wavefunctions.keys())
    wavelength_data = [wavefunctions[tid]['wavelengths'] for tid in token_ids]
    labels = [token_labels.get(tid, str(tid)) for tid in token_ids]
    
    positions = np.arange(len(token_ids))
    parts = ax1.violinplot(wavelength_data, positions=positions, showmeans=True, showmedians=True)
    
    for pc in parts['bodies']:
        pc.set_facecolor('cyan')
        pc.set_alpha(0.6)
    parts['cmeans'].set_color('yellow')
    parts['cmedians'].set_color('magenta')
    
    ax1.set_xticks(positions)
    ax1.set_xticklabels(labels, rotation=45, ha='right', fontsize=9)
    ax1.set_ylabel('Wavelength (1 + |w|)', color='white')
    ax1.set_title('Wavelength Distribution per Token', color='white')
    ax1.grid(True, alpha=0.3)
    
    # 2. Phase offset distribution
    ax2 = axes[0, 1]
    phase_data = [wavefunctions[tid]['b_full'] for tid in token_ids]
    
    parts2 = ax2.violinplot(phase_data, positions=positions, showmeans=True, showmedians=True)
    for pc in parts2['bodies']:
        pc.set_facecolor('orange')
        pc.set_alpha(0.6)
    parts2['cmeans'].set_color('yellow')
    parts2['cmedians'].set_color('magenta')
    
    ax2.set_xticks(positions)
    ax2.set_xticklabels(labels, rotation=45, ha='right', fontsize=9)
    ax2.set_ylabel('Phase Offset (b)', color='white')
    ax2.set_title('Phase Offset Distribution per Token', color='white')
    ax2.grid(True, alpha=0.3)
    
    # 3. Wavelength heatmap (tokens x dimensions)
    ax3 = axes[1, 0]
    wavelength_matrix = np.array([wavefunctions[tid]['wavelengths'] for tid in token_ids])
    im = ax3.imshow(wavelength_matrix, aspect='auto', cmap='plasma')
    ax3.set_yticks(np.arange(len(token_ids)))
    ax3.set_yticklabels(labels, fontsize=9)
    ax3.set_xlabel('Dimension', color='white')
    ax3.set_title('Wavelength Heatmap', color='white')
    plt.colorbar(im, ax=ax3, label='Wavelength')
    
    # 4. Phase offset heatmap
    ax4 = axes[1, 1]
    phase_matrix = np.array([wavefunctions[tid]['b_full'] for tid in token_ids])
    im2 = ax4.imshow(phase_matrix, aspect='auto', cmap='twilight')
    ax4.set_yticks(np.arange(len(token_ids)))
    ax4.set_yticklabels(labels, fontsize=9)
    ax4.set_xlabel('Dimension', color='white')
    ax4.set_title('Phase Offset Heatmap', color='white')
    plt.colorbar(im2, ax=ax4, label='Phase (b)')
    
    plt.tight_layout()
    return fig


def plot_probability_density_evolution(wavefunctions: Dict[int, Dict], 
                                        t_range: np.ndarray,
                                        token_labels: Dict[int, str],
                                        title: str = "Probability Density |Ψ|²") -> plt.Figure:
    """Plot |Ψ|² evolution over time and dimensions."""
    n_tokens = len(wavefunctions)
    fig, axes = plt.subplots(1, n_tokens, figsize=(5 * n_tokens, 5))
    if n_tokens == 1:
        axes = [axes]
    
    fig.suptitle(title, fontsize=16, color='white', y=1.02)
    
    for ax, (token_id, data) in zip(axes, wavefunctions.items()):
        n_dims = len(data['wavefunctions'])
        
        # Build |Ψ|² matrix: (dims, time)
        prob_density = np.zeros((n_dims, len(t_range)))
        for dim in range(n_dims):
            wf = data['wavefunctions'][dim]
            prob_density[dim] = wf['real']**2 + wf['imag']**2
        
        im = ax.imshow(prob_density, aspect='auto', cmap='inferno',
                      extent=[t_range[0], t_range[-1], n_dims, 0])
        ax.set_xlabel('Time (t)', color='white')
        ax.set_ylabel('Dimension', color='white')
        ax.set_title(token_labels.get(token_id, str(token_id)), color='white')
        plt.colorbar(im, ax=ax, label='|Ψ|²')
    
    plt.tight_layout()
    return fig


def plot_comparison_summary(before: Dict[int, Dict], after: Dict[int, Dict],
                            token_labels: Dict[int, str],
                            title: str = "Before vs After Training") -> plt.Figure:
    """Side-by-side comparison of key metrics."""
    fig = plt.figure(figsize=(16, 12))
    gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.3, wspace=0.25)
    fig.suptitle(title, fontsize=18, color='white', y=0.98)
    
    token_ids = list(before.keys())
    labels = [token_labels.get(tid, str(tid)) for tid in token_ids]
    positions = np.arange(len(token_ids))
    
    # Row 1: Mean wavelength comparison
    ax1 = fig.add_subplot(gs[0, 0])
    mean_wl_before = [np.mean(before[tid]['wavelengths']) for tid in token_ids]
    mean_wl_after = [np.mean(after[tid]['wavelengths']) for tid in token_ids]
    
    width = 0.35
    ax1.bar(positions - width/2, mean_wl_before, width, label='Before', color='#4a90d9', alpha=0.8)
    ax1.bar(positions + width/2, mean_wl_after, width, label='After', color='#d94a4a', alpha=0.8)
    ax1.set_xticks(positions)
    ax1.set_xticklabels(labels, rotation=45, ha='right', fontsize=9)
    ax1.set_ylabel('Mean Wavelength', color='white')
    ax1.set_title('Mean Wavelength per Token', color='white')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Row 1: Wavelength std comparison
    ax2 = fig.add_subplot(gs[0, 1])
    std_wl_before = [np.std(before[tid]['wavelengths']) for tid in token_ids]
    std_wl_after = [np.std(after[tid]['wavelengths']) for tid in token_ids]
    
    ax2.bar(positions - width/2, std_wl_before, width, label='Before', color='#4a90d9', alpha=0.8)
    ax2.bar(positions + width/2, std_wl_after, width, label='After', color='#d94a4a', alpha=0.8)
    ax2.set_xticks(positions)
    ax2.set_xticklabels(labels, rotation=45, ha='right', fontsize=9)
    ax2.set_ylabel('Wavelength Std', color='white')
    ax2.set_title('Wavelength Diversity per Token', color='white')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Row 2: Phase distribution change
    ax3 = fig.add_subplot(gs[1, 0])
    for i, tid in enumerate(token_ids[:6]):  # Limit for readability
        ax3.hist(before[tid]['b_full'], bins=20, alpha=0.5, 
                label=f'{token_labels.get(tid, tid)} before', histtype='step', linewidth=2)
    ax3.set_xlabel('Phase Offset (b)', color='white')
    ax3.set_ylabel('Count', color='white')
    ax3.set_title('Phase Distribution (Before)', color='white')
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3)
    
    ax4 = fig.add_subplot(gs[1, 1])
    for i, tid in enumerate(token_ids[:6]):
        ax4.hist(after[tid]['b_full'], bins=20, alpha=0.5,
                label=f'{token_labels.get(tid, tid)} after', histtype='step', linewidth=2)
    ax4.set_xlabel('Phase Offset (b)', color='white')
    ax4.set_ylabel('Count', color='white')
    ax4.set_title('Phase Distribution (After)', color='white')
    ax4.legend(fontsize=8)
    ax4.grid(True, alpha=0.3)
    
    # Row 3: Embedding distance from initialization
    ax5 = fig.add_subplot(gs[2, :])
    
    # Compute L2 distance of embeddings from initial state
    distances = []
    for tid in token_ids:
        w_diff = after[tid]['w_full'] - before[tid]['w_full']
        b_diff = after[tid]['b_full'] - before[tid]['b_full']
        dist = np.sqrt(np.sum(w_diff**2) + np.sum(b_diff**2))
        distances.append(dist)
    
    colors = cm.viridis(np.linspace(0, 1, len(token_ids)))
    bars = ax5.bar(positions, distances, color=colors, alpha=0.8)
    ax5.set_xticks(positions)
    ax5.set_xticklabels(labels, rotation=45, ha='right', fontsize=9)
    ax5.set_ylabel('Embedding Distance from Init', color='white')
    ax5.set_title('How Much Each Token Changed During Training', color='white')
    ax5.grid(True, alpha=0.3)
    
    # Highlight most changed
    max_idx = np.argmax(distances)
    bars[max_idx].set_color('red')
    bars[max_idx].set_alpha(1.0)
    
    plt.tight_layout()
    return fig


def plot_unit_circle_trajectory(wavefunctions: Dict[int, Dict], t_range: np.ndarray,
                                 token_labels: Dict[int, str], dim: int = 0,
                                 title: str = "Unit Circle Trajectories") -> plt.Figure:
    """Plot trajectories on the unit circle for a specific dimension."""
    n_tokens = len(wavefunctions)
    n_cols = min(4, n_tokens)
    n_rows = (n_tokens - 1) // n_cols + 1
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
    fig.suptitle(f"{title} (dim={dim})", fontsize=16, color='white', y=1.02)
    
    if n_tokens == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    
    for idx, (token_id, data) in enumerate(wavefunctions.items()):
        row, col = idx // n_cols, idx % n_cols
        ax = axes[row, col]
        
        wf = data['wavefunctions'][dim]
        real, imag = wf['real'], wf['imag']
        
        # Draw unit circle
        theta_circle = np.linspace(0, 2 * np.pi, 100)
        ax.plot(np.cos(theta_circle), np.sin(theta_circle), 'gray', alpha=0.3, linewidth=1)
        
        # Plot trajectory with time coloring
        points = np.array([real, imag]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        
        from matplotlib.collections import LineCollection
        norm = Normalize(t_range.min(), t_range.max())
        lc = LineCollection(segments, cmap='plasma', norm=norm, linewidth=2, alpha=0.8)
        lc.set_array(t_range[:-1])
        ax.add_collection(lc)
        
        # Start and end markers
        ax.scatter(real[0], imag[0], c='lime', s=100, marker='o', zorder=5, label='Start')
        ax.scatter(real[-1], imag[-1], c='red', s=100, marker='x', zorder=5, label='End')
        
        ax.set_xlim(-1.3, 1.3)
        ax.set_ylim(-1.3, 1.3)
        ax.set_aspect('equal')
        ax.set_xlabel('Re(Ψ)', color='white')
        ax.set_ylabel('Im(Ψ)', color='white')
        ax.set_title(f"{token_labels.get(token_id, str(token_id))}\nw={wf['w']:.3f}", 
                    color='white', fontsize=10)
        ax.grid(True, alpha=0.2)
    
    # Hide empty subplots
    for idx in range(n_tokens, n_rows * n_cols):
        row, col = idx // n_cols, idx % n_cols
        axes[row, col].axis('off')
    
    plt.tight_layout()
    return fig


# ============================================================================
# TRAINING
# ============================================================================

def train_model(model, train_loader, test_loader, epochs, device, lr=1e-3, weight_decay=1.0):
    """Train the model and return best test accuracy."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
    
    best_acc = 0
    for epoch in range(epochs):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(x), y)
            loss.backward()
            optimizer.step()
        scheduler.step()
        
        # Test
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                correct += (model(x).argmax(-1) == y).sum().item()
                total += y.size(0)
        acc = correct / total
        best_acc = max(best_acc, acc)
        
        if (epoch + 1) % 50 == 0 or epoch == epochs - 1:
            print(f"  Epoch {epoch+1:4d} | Test Acc: {acc*100:.1f}%")
    
    return best_acc


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Visualize RIN embeddings as wavefunctions")
    parser.add_argument("--p", type=int, default=97, help="Modulus")
    parser.add_argument("--d_model", type=int, default=48, help="Model dimension")
    parser.add_argument("--epochs", type=int, default=150, help="Training epochs")
    parser.add_argument("--save_dir", type=str, default="results/embeddings", help="Save directory")
    parser.add_argument("--t_max", type=float, default=20.0, help="Max time for visualization")
    parser.add_argument("--t_steps", type=int, default=500, help="Time steps")
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Golden ratio φ = {PHI:.6f}")
    
    # Create save directory
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Time range for visualization
    t_range = np.linspace(0, args.t_max, args.t_steps)
    
    # Tokens to visualize (interesting ones for modular arithmetic)
    # 0, 1 (basics), 48 (half of p), 96 (p-1), = sign
    key_tokens = [0, 1, 2, 48, 49, 96, args.p]  # p is the '=' token
    token_labels = {
        0: "0",
        1: "1", 
        2: "2",
        48: "48",
        49: "49 (carry)",
        96: "96",
        args.p: "= (sep)",
    }
    
    # Data
    train_ds = ModularAdditionDataset(args.p, "train")
    test_ds = ModularAdditionDataset(args.p, "test")
    train_loader = DataLoader(train_ds, batch_size=256, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=256)
    
    # Create model
    vocab_size = args.p + 1
    model = ModularRIN(vocab_size, d_model=args.d_model).to(device)
    
    print("\n" + "="*60)
    print("BEFORE TRAINING - Computing wavefunctions...")
    print("="*60)
    
    # Compute wavefunctions BEFORE training
    wf_before = compute_all_wavefunctions(model, key_tokens, t_range)
    
    # Generate visualizations - BEFORE
    print("Generating visualizations (before training)...")
    
    # 1. Helix visualization
    fig = plot_helix_grid(wf_before, t_range, token_labels, 
                          dims_to_show=[0, args.d_model//4, args.d_model//2],
                          title="Parametric Helix (Before Training)")
    fig.savefig(save_dir / "01_helix_before.png", dpi=150, bbox_inches='tight', 
                facecolor='#0a0a0a')
    plt.close(fig)
    
    # 2. Phase-magnitude cloud
    fig = plot_phase_magnitude_cloud(wf_before, t_range, token_labels,
                                     title="Phase-Magnitude Cloud (Before)")
    fig.savefig(save_dir / "02_phase_cloud_before.png", dpi=150, bbox_inches='tight',
                facecolor='#0a0a0a')
    plt.close(fig)
    
    # 3. Wavelength spectrum
    fig = plot_wavelength_spectrum(wf_before, token_labels,
                                   title="Wavelength Spectrum (Before)")
    fig.savefig(save_dir / "03_wavelength_before.png", dpi=150, bbox_inches='tight',
                facecolor='#0a0a0a')
    plt.close(fig)
    
    # 4. Probability density
    fig = plot_probability_density_evolution(wf_before, t_range, token_labels,
                                             title="Probability Density |Ψ|² (Before)")
    fig.savefig(save_dir / "04_prob_density_before.png", dpi=150, bbox_inches='tight',
                facecolor='#0a0a0a')
    plt.close(fig)
    
    # 5. Unit circle trajectories
    fig = plot_unit_circle_trajectory(wf_before, t_range, token_labels, dim=0,
                                       title="Unit Circle Trajectories (Before)")
    fig.savefig(save_dir / "05_unit_circle_before.png", dpi=150, bbox_inches='tight',
                facecolor='#0a0a0a')
    plt.close(fig)
    
    # Train model
    print("\n" + "="*60)
    print("TRAINING...")
    print("="*60)
    best_acc = train_model(model, train_loader, test_loader, args.epochs, device)
    print(f"\nBest test accuracy: {best_acc*100:.2f}%")
    
    print("\n" + "="*60)
    print("AFTER TRAINING - Computing wavefunctions...")
    print("="*60)
    
    # Compute wavefunctions AFTER training
    wf_after = compute_all_wavefunctions(model, key_tokens, t_range)
    
    # Generate visualizations - AFTER
    print("Generating visualizations (after training)...")
    
    # 1. Helix visualization
    fig = plot_helix_grid(wf_after, t_range, token_labels,
                          dims_to_show=[0, args.d_model//4, args.d_model//2],
                          title="Parametric Helix (After Training)")
    fig.savefig(save_dir / "06_helix_after.png", dpi=150, bbox_inches='tight',
                facecolor='#0a0a0a')
    plt.close(fig)
    
    # 2. Phase-magnitude cloud
    fig = plot_phase_magnitude_cloud(wf_after, t_range, token_labels,
                                     title="Phase-Magnitude Cloud (After)")
    fig.savefig(save_dir / "07_phase_cloud_after.png", dpi=150, bbox_inches='tight',
                facecolor='#0a0a0a')
    plt.close(fig)
    
    # 3. Wavelength spectrum
    fig = plot_wavelength_spectrum(wf_after, token_labels,
                                   title="Wavelength Spectrum (After)")
    fig.savefig(save_dir / "08_wavelength_after.png", dpi=150, bbox_inches='tight',
                facecolor='#0a0a0a')
    plt.close(fig)
    
    # 4. Probability density
    fig = plot_probability_density_evolution(wf_after, t_range, token_labels,
                                             title="Probability Density |Ψ|² (After)")
    fig.savefig(save_dir / "09_prob_density_after.png", dpi=150, bbox_inches='tight',
                facecolor='#0a0a0a')
    plt.close(fig)
    
    # 5. Unit circle trajectories
    fig = plot_unit_circle_trajectory(wf_after, t_range, token_labels, dim=0,
                                       title="Unit Circle Trajectories (After)")
    fig.savefig(save_dir / "10_unit_circle_after.png", dpi=150, bbox_inches='tight',
                facecolor='#0a0a0a')
    plt.close(fig)
    
    # 6. Comparison summary
    print("Generating comparison visualizations...")
    fig = plot_comparison_summary(wf_before, wf_after, token_labels,
                                  title="Before vs After Training Comparison")
    fig.savefig(save_dir / "11_comparison.png", dpi=150, bbox_inches='tight',
                facecolor='#0a0a0a')
    plt.close(fig)
    
    print(f"\n{'='*60}")
    print(f"All visualizations saved to: {save_dir}")
    print(f"{'='*60}")
    print("\nGenerated files:")
    for f in sorted(save_dir.glob("*.png")):
        print(f"  - {f.name}")


if __name__ == "__main__":
    main()

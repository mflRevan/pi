#!/usr/bin/env python3
"""
Interactive 3D Visualization of RIN Embeddings as Quantum Wavefunctions

Uses Plotly for interactive exploration of the wavefunction space.
Creates an HTML file you can open in a browser to explore the embeddings.

Visualizations:
1. Interactive Helix Gallery - rotate and zoom into each token's wavefunction
2. 3D Phase Space - all tokens in a unified phase-magnitude space
3. Embedding Evolution Animation - watch embeddings transform during training
4. Interference Patterns - when multiple tokens combine

Usage:
    python visualize_interactive.py
    python visualize_interactive.py --epochs 200 --animate
"""

import argparse
import math
import random
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import json

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import plotly.express as px
    PLOTLY_AVAILABLE = True
except ImportError:
    print("Plotly not installed. Run: pip install plotly")
    PLOTLY_AVAILABLE = False

from rin import get_global_lut, PHI


# ============================================================================
# MODEL (same as train_modular.py)
# ============================================================================

class ModularRIN(nn.Module):
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

def compute_wavefunction(w: np.ndarray, b: np.ndarray, t_range: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    wavelength = 1.0 + np.abs(w)
    theta = b + t_range * PHI  # Simplified: starting from h=0
    return np.cos(theta), np.sin(theta)


def compute_all_wavefunctions(model: nn.Module, token_ids: List[int], t_range: np.ndarray) -> Dict:
    model.eval()
    device = next(model.parameters()).device
    results = {}
    
    with torch.no_grad():
        for token_id in token_ids:
            token_tensor = torch.tensor([token_id], device=device)
            w, b = model.get_wb_embeddings(token_tensor)
            w = w[0].cpu().numpy()
            b = b[0].cpu().numpy()
            
            wavefunctions = []
            for dim in range(len(w)):
                wavelength = 1.0 + abs(w[dim])
                theta = b[dim] + t_range * PHI / wavelength
                real = np.cos(theta)
                imag = np.sin(theta)
                wavefunctions.append({
                    'w': w[dim], 'b': b[dim],
                    'real': real, 'imag': imag,
                    'magnitude': np.sqrt(real**2 + imag**2),
                    'phase': np.arctan2(imag, real),
                    'wavelength': wavelength,
                })
            
            results[token_id] = {
                'w_full': w, 'b_full': b,
                'wavelengths': 1.0 + np.abs(w),
                'wavefunctions': wavefunctions,
            }
    
    return results


# ============================================================================
# INTERACTIVE VISUALIZATIONS
# ============================================================================

def create_helix_figure(wavefunctions: Dict, t_range: np.ndarray, 
                        token_labels: Dict, dims: List[int],
                        title: str = "Quantum Wavefunction Helices") -> go.Figure:
    """Create interactive 3D helix visualization."""
    
    n_tokens = len(wavefunctions)
    n_dims = len(dims)
    
    fig = make_subplots(
        rows=n_tokens, cols=n_dims,
        specs=[[{'type': 'scene'}] * n_dims for _ in range(n_tokens)],
        subplot_titles=[f"{token_labels.get(tid, tid)} | dim {d}" 
                       for tid in wavefunctions.keys() for d in dims],
        vertical_spacing=0.05,
        horizontal_spacing=0.05,
    )
    
    colors = px.colors.sequential.Plasma
    
    for i, (token_id, data) in enumerate(wavefunctions.items()):
        for j, dim in enumerate(dims):
            wf = data['wavefunctions'][dim]
            
            # Create color array based on time
            color_idx = np.linspace(0, len(colors) - 1, len(t_range)).astype(int)
            
            fig.add_trace(
                go.Scatter3d(
                    x=t_range, y=wf['real'], z=wf['imag'],
                    mode='lines',
                    line=dict(
                        color=t_range,
                        colorscale='Plasma',
                        width=4,
                    ),
                    name=f"Token {token_labels.get(token_id, token_id)}, dim {dim}",
                    hovertemplate=(
                        f"Token: {token_labels.get(token_id, token_id)}<br>"
                        f"Dim: {dim}<br>"
                        f"w={wf['w']:.4f}, b={wf['b']:.4f}<br>"
                        "t=%{x:.2f}<br>"
                        "Re(Ψ)=%{y:.4f}<br>"
                        "Im(Ψ)=%{z:.4f}"
                    ),
                ),
                row=i+1, col=j+1
            )
    
    # Update layout for each subplot
    for i in range(n_tokens):
        for j in range(n_dims):
            scene_name = f'scene{i * n_dims + j + 1}' if i > 0 or j > 0 else 'scene'
            fig.update_layout(**{
                scene_name: dict(
                    xaxis_title='Time (t)',
                    yaxis_title='Re(Ψ)',
                    zaxis_title='Im(Ψ)',
                    xaxis=dict(backgroundcolor='rgba(10,10,10,0.9)', gridcolor='#333'),
                    yaxis=dict(backgroundcolor='rgba(10,10,10,0.9)', gridcolor='#333', range=[-1.2, 1.2]),
                    zaxis=dict(backgroundcolor='rgba(10,10,10,0.9)', gridcolor='#333', range=[-1.2, 1.2]),
                    bgcolor='rgba(10,10,10,0.9)',
                )
            })
    
    fig.update_layout(
        title=dict(text=title, font=dict(size=20, color='white')),
        paper_bgcolor='#0a0a0a',
        plot_bgcolor='#0a0a0a',
        font=dict(color='white'),
        height=300 * n_tokens,
        showlegend=False,
    )
    
    return fig


def create_phase_space_figure(wavefunctions: Dict, t_range: np.ndarray,
                               token_labels: Dict,
                               title: str = "Phase Space Exploration") -> go.Figure:
    """Create unified phase-magnitude space visualization."""
    
    fig = go.Figure()
    
    # Color palette
    colors = px.colors.qualitative.Set3
    
    for idx, (token_id, data) in enumerate(wavefunctions.items()):
        color = colors[idx % len(colors)]
        
        # Aggregate across dimensions (sample a few)
        sample_dims = np.linspace(0, len(data['wavefunctions'])-1, 8, dtype=int)
        
        all_t = []
        all_phase = []
        all_mag = []
        
        for dim in sample_dims:
            wf = data['wavefunctions'][dim]
            # Subsample time
            step = max(1, len(t_range) // 50)
            all_t.extend(t_range[::step])
            all_phase.extend(wf['phase'][::step])
            all_mag.extend(wf['magnitude'][::step])
        
        fig.add_trace(go.Scatter3d(
            x=all_t,
            y=all_phase,
            z=all_mag,
            mode='markers',
            marker=dict(
                size=3,
                color=color,
                opacity=0.7,
            ),
            name=token_labels.get(token_id, str(token_id)),
            hovertemplate=(
                f"Token: {token_labels.get(token_id, str(token_id))}<br>"
                "t=%{x:.2f}<br>"
                "Phase=%{y:.4f}<br>"
                "|Ψ|=%{z:.4f}"
            ),
        ))
    
    fig.update_layout(
        title=dict(text=title, font=dict(size=20, color='white')),
        scene=dict(
            xaxis_title='Time (t)',
            yaxis_title='Phase (arg Ψ)',
            zaxis_title='Magnitude |Ψ|',
            xaxis=dict(backgroundcolor='rgba(10,10,10,0.9)', gridcolor='#333'),
            yaxis=dict(backgroundcolor='rgba(10,10,10,0.9)', gridcolor='#333'),
            zaxis=dict(backgroundcolor='rgba(10,10,10,0.9)', gridcolor='#333'),
            bgcolor='rgba(10,10,10,0.9)',
        ),
        paper_bgcolor='#0a0a0a',
        plot_bgcolor='#0a0a0a',
        font=dict(color='white'),
        height=700,
        legend=dict(
            bgcolor='rgba(30,30,30,0.8)',
            font=dict(color='white'),
        ),
    )
    
    return fig


def create_unit_circle_animation(wavefunctions: Dict, t_range: np.ndarray,
                                  token_labels: Dict, dim: int = 0,
                                  title: str = "Unit Circle Evolution") -> go.Figure:
    """Create animated unit circle visualization."""
    
    # Create frames
    n_frames = min(100, len(t_range))
    frame_indices = np.linspace(0, len(t_range)-1, n_frames, dtype=int)
    
    colors = px.colors.qualitative.Set3
    
    # Initial frame
    fig = go.Figure()
    
    # Add unit circle
    theta_circle = np.linspace(0, 2*np.pi, 100)
    fig.add_trace(go.Scatter(
        x=np.cos(theta_circle),
        y=np.sin(theta_circle),
        mode='lines',
        line=dict(color='gray', width=1, dash='dot'),
        name='Unit Circle',
        hoverinfo='skip',
    ))
    
    # Add initial positions for each token
    for idx, (token_id, data) in enumerate(wavefunctions.items()):
        wf = data['wavefunctions'][dim]
        color = colors[idx % len(colors)]
        
        # Trail
        fig.add_trace(go.Scatter(
            x=wf['real'][:frame_indices[0]+1],
            y=wf['imag'][:frame_indices[0]+1],
            mode='lines',
            line=dict(color=color, width=2),
            opacity=0.5,
            name=f"{token_labels.get(token_id, token_id)} trail",
            showlegend=False,
        ))
        
        # Current position
        fig.add_trace(go.Scatter(
            x=[wf['real'][0]],
            y=[wf['imag'][0]],
            mode='markers',
            marker=dict(size=15, color=color, symbol='circle'),
            name=token_labels.get(token_id, str(token_id)),
        ))
    
    # Create animation frames
    frames = []
    for fi, tidx in enumerate(frame_indices):
        frame_data = [
            # Unit circle (static)
            go.Scatter(
                x=np.cos(theta_circle),
                y=np.sin(theta_circle),
                mode='lines',
                line=dict(color='gray', width=1, dash='dot'),
            )
        ]
        
        for idx, (token_id, data) in enumerate(wavefunctions.items()):
            wf = data['wavefunctions'][dim]
            color = colors[idx % len(colors)]
            
            # Trail
            start_idx = max(0, tidx - 30)
            frame_data.append(go.Scatter(
                x=wf['real'][start_idx:tidx+1],
                y=wf['imag'][start_idx:tidx+1],
                mode='lines',
                line=dict(color=color, width=2),
                opacity=0.5,
            ))
            
            # Current position
            frame_data.append(go.Scatter(
                x=[wf['real'][tidx]],
                y=[wf['imag'][tidx]],
                mode='markers',
                marker=dict(size=15, color=color, symbol='circle'),
            ))
        
        frames.append(go.Frame(data=frame_data, name=str(fi)))
    
    fig.frames = frames
    
    # Animation controls
    fig.update_layout(
        title=dict(text=f"{title} (dim={dim})", font=dict(size=20, color='white')),
        xaxis=dict(
            range=[-1.5, 1.5], 
            title='Re(Ψ)', 
            scaleanchor='y',
            scaleratio=1,
            gridcolor='#333',
        ),
        yaxis=dict(
            range=[-1.5, 1.5], 
            title='Im(Ψ)',
            gridcolor='#333',
        ),
        paper_bgcolor='#0a0a0a',
        plot_bgcolor='#0a0a0a',
        font=dict(color='white'),
        height=600,
        width=600,
        updatemenus=[dict(
            type='buttons',
            showactive=False,
            y=1.15,
            x=0.5,
            xanchor='center',
            buttons=[
                dict(label='▶ Play',
                     method='animate',
                     args=[None, dict(frame=dict(duration=50, redraw=True),
                                     fromcurrent=True,
                                     transition=dict(duration=0))]),
                dict(label='⏸ Pause',
                     method='animate',
                     args=[[None], dict(frame=dict(duration=0, redraw=False),
                                       mode='immediate',
                                       transition=dict(duration=0))]),
            ],
        )],
        sliders=[dict(
            active=0,
            steps=[dict(method='animate',
                       args=[[str(k)], dict(mode='immediate',
                                           frame=dict(duration=50, redraw=True),
                                           transition=dict(duration=0))],
                       label=f't={t_range[frame_indices[k]]:.1f}')
                   for k in range(len(frame_indices))],
            x=0.1, len=0.8,
            xanchor='left',
            y=-0.05,
            currentvalue=dict(
                prefix='Time: ',
                visible=True,
                font=dict(color='white'),
            ),
            font=dict(color='white'),
            tickcolor='white',
        )],
    )
    
    return fig


def create_wavelength_comparison(before: Dict, after: Dict, 
                                  token_labels: Dict,
                                  title: str = "Wavelength Evolution") -> go.Figure:
    """Compare wavelength distributions before and after training."""
    
    token_ids = list(before.keys())
    labels = [token_labels.get(tid, str(tid)) for tid in token_ids]
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[
            'Mean Wavelength per Token',
            'Wavelength Std (Diversity)',
            'Before Training Heatmap',
            'After Training Heatmap',
        ],
        specs=[
            [{'type': 'bar'}, {'type': 'bar'}],
            [{'type': 'heatmap'}, {'type': 'heatmap'}],
        ],
        vertical_spacing=0.15,
        horizontal_spacing=0.1,
    )
    
    # Mean wavelength comparison
    mean_before = [np.mean(before[tid]['wavelengths']) for tid in token_ids]
    mean_after = [np.mean(after[tid]['wavelengths']) for tid in token_ids]
    
    fig.add_trace(go.Bar(x=labels, y=mean_before, name='Before', marker_color='#4a90d9'), row=1, col=1)
    fig.add_trace(go.Bar(x=labels, y=mean_after, name='After', marker_color='#d94a4a'), row=1, col=1)
    
    # Std comparison
    std_before = [np.std(before[tid]['wavelengths']) for tid in token_ids]
    std_after = [np.std(after[tid]['wavelengths']) for tid in token_ids]
    
    fig.add_trace(go.Bar(x=labels, y=std_before, name='Before', marker_color='#4a90d9', showlegend=False), row=1, col=2)
    fig.add_trace(go.Bar(x=labels, y=std_after, name='After', marker_color='#d94a4a', showlegend=False), row=1, col=2)
    
    # Heatmaps
    wl_before = np.array([before[tid]['wavelengths'] for tid in token_ids])
    wl_after = np.array([after[tid]['wavelengths'] for tid in token_ids])
    
    fig.add_trace(go.Heatmap(z=wl_before, y=labels, colorscale='Plasma', name='Before', showscale=False), row=2, col=1)
    fig.add_trace(go.Heatmap(z=wl_after, y=labels, colorscale='Plasma', name='After'), row=2, col=2)
    
    fig.update_layout(
        title=dict(text=title, font=dict(size=20, color='white')),
        paper_bgcolor='#0a0a0a',
        plot_bgcolor='#0a0a0a',
        font=dict(color='white'),
        height=800,
        barmode='group',
        legend=dict(bgcolor='rgba(30,30,30,0.8)'),
    )
    
    # Update axes
    for i in range(1, 3):
        for j in range(1, 3):
            fig.update_xaxes(gridcolor='#333', row=i, col=j)
            fig.update_yaxes(gridcolor='#333', row=i, col=j)
    
    return fig


def create_interference_pattern(wavefunctions: Dict, t_range: np.ndarray,
                                token_a: int, token_b: int,
                                token_labels: Dict,
                                title: str = "Interference Pattern") -> go.Figure:
    """Visualize interference when two tokens combine."""
    
    if token_a not in wavefunctions or token_b not in wavefunctions:
        return go.Figure()
    
    data_a = wavefunctions[token_a]
    data_b = wavefunctions[token_b]
    
    # Sum wavefunctions across a few dimensions
    sample_dims = [0, len(data_a['wavefunctions'])//4, len(data_a['wavefunctions'])//2]
    
    fig = make_subplots(
        rows=1, cols=len(sample_dims),
        specs=[[{'type': 'scene'}] * len(sample_dims)],
        subplot_titles=[f'Dimension {d}' for d in sample_dims],
    )
    
    for col, dim in enumerate(sample_dims):
        wf_a = data_a['wavefunctions'][dim]
        wf_b = data_b['wavefunctions'][dim]
        
        # Combined wavefunction
        combined_real = wf_a['real'] + wf_b['real']
        combined_imag = wf_a['imag'] + wf_b['imag']
        
        # Individual wavefunctions
        fig.add_trace(go.Scatter3d(
            x=t_range, y=wf_a['real'], z=wf_a['imag'],
            mode='lines', line=dict(color='cyan', width=3),
            name=f"{token_labels.get(token_a, token_a)}",
            showlegend=(col == 0),
        ), row=1, col=col+1)
        
        fig.add_trace(go.Scatter3d(
            x=t_range, y=wf_b['real'], z=wf_b['imag'],
            mode='lines', line=dict(color='magenta', width=3),
            name=f"{token_labels.get(token_b, token_b)}",
            showlegend=(col == 0),
        ), row=1, col=col+1)
        
        # Combined
        fig.add_trace(go.Scatter3d(
            x=t_range, y=combined_real, z=combined_imag,
            mode='lines', line=dict(color='yellow', width=4),
            name='Interference',
            showlegend=(col == 0),
        ), row=1, col=col+1)
    
    # Update scenes
    for i in range(len(sample_dims)):
        scene_name = f'scene{i+1}' if i > 0 else 'scene'
        fig.update_layout(**{
            scene_name: dict(
                xaxis_title='Time',
                yaxis_title='Re(Ψ)',
                zaxis_title='Im(Ψ)',
                bgcolor='rgba(10,10,10,0.9)',
                xaxis=dict(backgroundcolor='rgba(10,10,10,0.9)', gridcolor='#333'),
                yaxis=dict(backgroundcolor='rgba(10,10,10,0.9)', gridcolor='#333'),
                zaxis=dict(backgroundcolor='rgba(10,10,10,0.9)', gridcolor='#333'),
            )
        })
    
    label_a = token_labels.get(token_a, str(token_a))
    label_b = token_labels.get(token_b, str(token_b))
    
    fig.update_layout(
        title=dict(text=f"{title}: {label_a} + {label_b}", font=dict(size=20, color='white')),
        paper_bgcolor='#0a0a0a',
        font=dict(color='white'),
        height=500,
        legend=dict(bgcolor='rgba(30,30,30,0.8)'),
    )
    
    return fig


# ============================================================================
# TRAINING
# ============================================================================

def train_with_snapshots(model, train_loader, test_loader, epochs, device, 
                          snapshot_epochs: List[int], token_ids: List[int],
                          t_range: np.ndarray, lr=1e-3, weight_decay=1.0):
    """Train and capture embedding snapshots at specific epochs."""
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
    
    snapshots = {0: compute_all_wavefunctions(model, token_ids, t_range)}
    accuracies = {0: 0.0}
    
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
        
        if (epoch + 1) in snapshot_epochs:
            print(f"  Epoch {epoch+1:4d} | Acc: {acc*100:.1f}% [snapshot]")
            snapshots[epoch + 1] = compute_all_wavefunctions(model, token_ids, t_range)
            accuracies[epoch + 1] = acc
        elif (epoch + 1) % 50 == 0:
            print(f"  Epoch {epoch+1:4d} | Acc: {acc*100:.1f}%")
    
    return snapshots, accuracies


# ============================================================================
# MAIN
# ============================================================================

def main():
    if not PLOTLY_AVAILABLE:
        print("ERROR: Plotly is required. Install with: pip install plotly")
        return
    
    parser = argparse.ArgumentParser(description="Interactive RIN embedding visualization")
    parser.add_argument("--p", type=int, default=97, help="Modulus")
    parser.add_argument("--d_model", type=int, default=48, help="Model dimension")
    parser.add_argument("--epochs", type=int, default=200, help="Training epochs")
    parser.add_argument("--save_dir", type=str, default="results/interactive", help="Save directory")
    parser.add_argument("--t_max", type=float, default=15.0, help="Max time")
    parser.add_argument("--t_steps", type=int, default=300, help="Time steps")
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Golden ratio φ = {PHI:.6f}")
    
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    t_range = np.linspace(0, args.t_max, args.t_steps)
    
    # Key tokens for modular arithmetic
    key_tokens = [0, 1, 2, 48, 49, 96, args.p]
    token_labels = {
        0: "0", 1: "1", 2: "2", 48: "48",
        49: "49 (carry)", 96: "96", args.p: "= (sep)",
    }
    
    # Data
    train_ds = ModularAdditionDataset(args.p, "train")
    test_ds = ModularAdditionDataset(args.p, "test")
    train_loader = DataLoader(train_ds, batch_size=256, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=256)
    
    # Model
    vocab_size = args.p + 1
    model = ModularRIN(vocab_size, d_model=args.d_model).to(device)
    
    print("\n" + "="*60)
    print("Training with snapshots...")
    print("="*60)
    
    snapshot_epochs = [1, 10, 25, 50, 100, args.epochs]
    snapshots, accuracies = train_with_snapshots(
        model, train_loader, test_loader, args.epochs, device,
        snapshot_epochs, key_tokens, t_range
    )
    
    print("\n" + "="*60)
    print("Creating interactive visualizations...")
    print("="*60)
    
    before = snapshots[0]
    after = snapshots[args.epochs]
    
    # 1. Helix Gallery
    print("  Creating helix gallery...")
    fig = create_helix_figure(after, t_range, token_labels, 
                              dims=[0, args.d_model//4, args.d_model//2],
                              title="Quantum Wavefunction Helices (After Training)")
    fig.write_html(save_dir / "01_helix_gallery.html")
    
    # 2. Phase Space
    print("  Creating phase space...")
    fig = create_phase_space_figure(after, t_range, token_labels,
                                    title="Phase-Magnitude Space Exploration")
    fig.write_html(save_dir / "02_phase_space.html")
    
    # 3. Unit Circle Animation
    print("  Creating unit circle animation...")
    fig = create_unit_circle_animation(after, t_range, token_labels, dim=0,
                                        title="Unit Circle Evolution")
    fig.write_html(save_dir / "03_unit_circle_anim.html")
    
    # 4. Wavelength Comparison
    print("  Creating wavelength comparison...")
    fig = create_wavelength_comparison(before, after, token_labels,
                                       title="Embedding Evolution: Before vs After")
    fig.write_html(save_dir / "04_wavelength_compare.html")
    
    # 5. Interference Pattern
    print("  Creating interference patterns...")
    fig = create_interference_pattern(after, t_range, 1, 48, token_labels,
                                      title="Wavefunction Interference")
    fig.write_html(save_dir / "05_interference_1_48.html")
    
    fig = create_interference_pattern(after, t_range, 49, 48, token_labels,
                                      title="Carry Bit Interference")
    fig.write_html(save_dir / "06_interference_carry.html")
    
    print(f"\n{'='*60}")
    print(f"All interactive visualizations saved to: {save_dir}")
    print(f"{'='*60}")
    print("\nGenerated HTML files (open in browser):")
    for f in sorted(save_dir.glob("*.html")):
        print(f"  - {f.name}")
    
    print(f"\nFinal accuracy: {accuracies[args.epochs]*100:.2f}%")


if __name__ == "__main__":
    main()

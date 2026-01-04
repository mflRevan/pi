#!/usr/bin/env python3
"""
Visualize gradient distributions and evolution during RIN training.
"""

import json
import sys
from pathlib import Path
import numpy as np

try:
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
except ImportError:
    print("matplotlib not installed. Install with: pip install matplotlib")
    sys.exit(1)


def load_gradient_data(filepath):
    """Load gradient history from JSON file."""
    with open(filepath, 'r') as f:
        data = json.load(f)
    return data


def plot_gradient_evolution(grad_history, output_dir):
    """Plot how gradients evolve over epochs."""
    epochs = grad_history['epochs']
    
    # Group components by type
    groups = {
        'Embeddings & Output': {
            'embeddings': 'Embeddings',
            'output_layer': 'Output Layer',
        },
        'Time-Dependent Measurement': {
            'layer_0_amplitude': 'Amplitude',
            'layer_0_angle': 'Angle',
        },
        'Resonant Parameters': {
            'layer_0_wavelength': 'Wavelength (W)',
            'layer_0_phase_offset': 'Phase Offset (B)',
        },
        'ComplexLinear Mixer': {
            'layer_0_mixer_real': 'Mixer Real',
            'layer_0_mixer_imag': 'Mixer Imaginary',
        },
    }
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Gradient Evolution During Training', fontsize=16, fontweight='bold')
    
    for ax, (group_name, components) in zip(axes.flat, groups.items()):
        for key, label in components.items():
            if key in grad_history and len(grad_history[key]) > 0:
                values = grad_history[key]
                ax.plot(epochs[:len(values)], values, marker='o', label=label, linewidth=2, markersize=4)
        
        ax.set_xlabel('Epoch', fontsize=11)
        ax.set_ylabel('Gradient Norm', fontsize=11)
        ax.set_title(group_name, fontsize=12, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = output_dir / 'gradient_evolution.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    plt.close()


def plot_gradient_distributions(grad_history, output_dir):
    """Plot distribution of gradient magnitudes."""
    # Collect all data
    components = {
        'Embeddings': grad_history.get('embeddings', []),
        'Amplitude': grad_history.get('layer_0_amplitude', []),
        'Angle': grad_history.get('layer_0_angle', []),
        'Wavelength': grad_history.get('layer_0_wavelength', []),
        'Phase Offset': grad_history.get('layer_0_phase_offset', []),
        'Mixer Real': grad_history.get('layer_0_mixer_real', []),
        'Mixer Imag': grad_history.get('layer_0_mixer_imag', []),
        'Output': grad_history.get('output_layer', []),
    }
    
    # Filter out empty components
    components = {k: v for k, v in components.items() if len(v) > 0}
    
    # Create violin plot
    fig, ax = plt.subplots(figsize=(12, 6))
    
    data = [v for v in components.values()]
    labels = list(components.keys())
    
    parts = ax.violinplot(data, positions=range(len(labels)), showmeans=True, showmedians=True)
    
    # Color the violins
    colors = plt.cm.Set3(np.linspace(0, 1, len(labels)))
    for i, pc in enumerate(parts['bodies']):
        pc.set_facecolor(colors[i])
        pc.set_alpha(0.7)
    
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_ylabel('Gradient Norm', fontsize=12)
    ax.set_title('Gradient Distribution per Component (Violin Plot)', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    output_path = output_dir / 'gradient_distributions.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    plt.close()


def plot_gradient_histograms(grad_history, output_dir):
    """Plot histograms of gradient values."""
    components = {
        'Embeddings': grad_history.get('embeddings', []),
        'Amplitude': grad_history.get('layer_0_amplitude', []),
        'Angle': grad_history.get('layer_0_angle', []),
        'Wavelength': grad_history.get('layer_0_wavelength', []),
        'Phase Offset': grad_history.get('layer_0_phase_offset', []),
        'Mixer Real': grad_history.get('layer_0_mixer_real', []),
        'Mixer Imag': grad_history.get('layer_0_mixer_imag', []),
        'Output': grad_history.get('output_layer', []),
    }
    
    components = {k: v for k, v in components.items() if len(v) > 0}
    
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    fig.suptitle('Gradient Histograms', fontsize=16, fontweight='bold')
    
    for ax, (name, values) in zip(axes.flat, components.items()):
        ax.hist(values, bins=15, alpha=0.7, color='steelblue', edgecolor='black')
        ax.axvline(np.mean(values), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(values):.4f}')
        ax.axvline(np.median(values), color='green', linestyle=':', linewidth=2, label=f'Median: {np.median(values):.4f}')
        ax.set_xlabel('Gradient Norm', fontsize=10)
        ax.set_ylabel('Count', fontsize=10)
        ax.set_title(name, fontsize=11, fontweight='bold')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, axis='y')
    
    # Hide extra subplots if any
    for i in range(len(components), len(axes.flat)):
        axes.flat[i].set_visible(False)
    
    plt.tight_layout()
    output_path = output_dir / 'gradient_histograms.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    plt.close()


def plot_gradient_ratios(grad_history, output_dir):
    """Plot ratios between different gradient components."""
    epochs = grad_history['epochs']
    
    # Calculate interesting ratios
    mixer_real = grad_history.get('layer_0_mixer_real', [])
    mixer_imag = grad_history.get('layer_0_mixer_imag', [])
    embeddings = grad_history.get('embeddings', [])
    output = grad_history.get('output_layer', [])
    amplitude = grad_history.get('layer_0_amplitude', [])
    angle = grad_history.get('layer_0_angle', [])
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Gradient Ratios and Relationships', fontsize=16, fontweight='bold')
    
    # 1. Real/Imaginary balance in mixer
    if mixer_real and mixer_imag:
        min_len = min(len(mixer_real), len(mixer_imag))
        ratio = [mixer_real[i] / mixer_imag[i] if mixer_imag[i] > 0 else 0 
                 for i in range(min_len)]
        axes[0, 0].plot(epochs[:min_len], ratio, marker='o', color='purple', linewidth=2)
        axes[0, 0].axhline(1.0, color='red', linestyle='--', label='Perfect Balance')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Ratio')
        axes[0, 0].set_title('ComplexLinear Mixer: Real/Imaginary Ratio')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Mixer to Embeddings ratio
    if mixer_real and embeddings:
        min_len = min(len(mixer_real), len(embeddings))
        ratio = [mixer_real[i] / embeddings[i] if embeddings[i] > 0 else 0 
                 for i in range(min_len)]
        axes[0, 1].plot(epochs[:min_len], ratio, marker='s', color='orange', linewidth=2)
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Ratio')
        axes[0, 1].set_title('Mixer/Embeddings Gradient Ratio')
        axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Embeddings to Output ratio (gradient flow quality)
    if embeddings and output:
        min_len = min(len(embeddings), len(output))
        ratio = [embeddings[i] / output[i] if output[i] > 0 else 0 
                 for i in range(min_len)]
        axes[1, 0].plot(epochs[:min_len], ratio, marker='^', color='red', linewidth=2)
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Ratio')
        axes[1, 0].set_title('Embeddings/Output Gradient Ratio (Flow Quality)')
        axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Amplitude/Angle balance (time-dependent measurement)
    if amplitude and angle:
        min_len = min(len(amplitude), len(angle))
        ratio = [amplitude[i] / angle[i] if angle[i] > 0 else 0 
                 for i in range(min_len)]
        axes[1, 1].plot(epochs[:min_len], ratio, marker='d', color='green', linewidth=2)
        axes[1, 1].axhline(1.0, color='red', linestyle='--', label='Equal Learning')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Ratio')
        axes[1, 1].set_title('Time-Dependent Measurement: Amplitude/Angle Ratio')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = output_dir / 'gradient_ratios.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    plt.close()


def plot_log_scale_comparison(grad_history, output_dir):
    """Plot all gradients on log scale to see relative magnitudes."""
    epochs = grad_history['epochs']
    
    components = {
        'Embeddings': grad_history.get('embeddings', []),
        'Amplitude': grad_history.get('layer_0_amplitude', []),
        'Angle': grad_history.get('layer_0_angle', []),
        'Wavelength': grad_history.get('layer_0_wavelength', []),
        'Phase Offset': grad_history.get('layer_0_phase_offset', []),
        'Mixer Real': grad_history.get('layer_0_mixer_real', []),
        'Mixer Imag': grad_history.get('layer_0_mixer_imag', []),
        'Output': grad_history.get('output_layer', []),
    }
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(components)))
    
    for (name, values), color in zip(components.items(), colors):
        if len(values) > 0:
            # Add small epsilon to avoid log(0)
            safe_values = [max(v, 1e-10) for v in values]
            ax.plot(epochs[:len(values)], safe_values, marker='o', label=name, 
                   linewidth=2, markersize=5, color=color, alpha=0.8)
    
    ax.set_yscale('log')
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Gradient Norm (log scale)', fontsize=12)
    ax.set_title('All Gradient Components (Log Scale)', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10, loc='best')
    ax.grid(True, alpha=0.3, which='both')
    
    plt.tight_layout()
    output_path = output_dir / 'gradient_log_scale.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    plt.close()


def plot_cumulative_gradient_flow(grad_history, output_dir):
    """Plot cumulative gradient norm to see which parameters receive most signal."""
    components = {
        'Embeddings': grad_history.get('embeddings', []),
        'Amplitude': grad_history.get('layer_0_amplitude', []),
        'Angle': grad_history.get('layer_0_angle', []),
        'Wavelength': grad_history.get('layer_0_wavelength', []),
        'Phase Offset': grad_history.get('layer_0_phase_offset', []),
        'Mixer Real': grad_history.get('layer_0_mixer_real', []),
        'Mixer Imag': grad_history.get('layer_0_mixer_imag', []),
        'Output': grad_history.get('output_layer', []),
    }
    
    epochs = grad_history['epochs']
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(components)))
    
    for (name, values), color in zip(components.items(), colors):
        if len(values) > 0:
            cumulative = np.cumsum(values)
            ax.plot(epochs[:len(values)], cumulative, label=name, 
                   linewidth=2.5, color=color, alpha=0.8)
    
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Cumulative Gradient Norm', fontsize=12)
    ax.set_title('Cumulative Gradient Flow (Total Signal Received)', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10, loc='best')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = output_dir / 'cumulative_gradient_flow.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    plt.close()


def main():
    if len(sys.argv) > 1:
        filepath = Path(sys.argv[1])
    else:
        # Use most recent gradient file
        grad_dir = Path("results/gradients")
        if not grad_dir.exists():
            print("No gradient results found!")
            sys.exit(1)
        
        files = sorted(grad_dir.glob("gradient_analysis_*.json"), 
                      key=lambda p: p.stat().st_mtime)
        if not files:
            print("No gradient analysis files found!")
            sys.exit(1)
        
        filepath = files[-1]
    
    print(f"Loading gradient data from: {filepath}")
    data = load_gradient_data(filepath)
    grad_history = data['gradient_history']
    
    # Create output directory
    output_dir = Path("results/gradient_plots")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\nGenerating visualizations...")
    print("-" * 60)
    
    plot_gradient_evolution(grad_history, output_dir)
    plot_gradient_distributions(grad_history, output_dir)
    plot_gradient_histograms(grad_history, output_dir)
    plot_gradient_ratios(grad_history, output_dir)
    plot_log_scale_comparison(grad_history, output_dir)
    plot_cumulative_gradient_flow(grad_history, output_dir)
    
    print("-" * 60)
    print(f"\n✓ All plots saved to: {output_dir}/")
    print("\nGenerated plots:")
    print("  1. gradient_evolution.png - Evolution over time by component type")
    print("  2. gradient_distributions.png - Violin plots showing distribution")
    print("  3. gradient_histograms.png - Histograms with mean/median")
    print("  4. gradient_ratios.png - Ratios between components")
    print("  5. gradient_log_scale.png - All components on log scale")
    print("  6. cumulative_gradient_flow.png - Total signal received")


if __name__ == "__main__":
    main()

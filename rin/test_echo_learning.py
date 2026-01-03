"""Test Echo Chamber learning dynamics - track interference scores over epochs."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

import sys
sys.path.insert(0, '/home/aiman/pi')

from rin.echo_chamber import EchoChamberModel

def test_scoring_functions():
    """Compare different scoring approaches for constructive interference."""
    print("="*80)
    print("SCORING FUNCTION COMPARISON")
    print("="*80)
    
    # Simulate various interference patterns
    patterns = [
        ("Perfect constructive", 1.0, 0.0),
        ("Good constructive", 0.7, 0.1),
        ("Moderate match", 0.3, 0.2),
        ("Orthogonal", 0.0, 0.0),
        ("Anti-phase (destructive)", -0.7, 0.1),
        ("Perfect destructive", -1.0, 0.0),
    ]
    
    print(f"\n{'Pattern':<25} | {'Real':>6} {'Imag':>6} | {'Mag':>6} | {'RealOnly':>9} | {'ExpDist':>8} | {'ExpReal':>8}")
    print("-"*100)
    
    for name, real, imag in patterns:
        # Current approach: magnitude (WRONG - measures both constructive and destructive)
        mag = math.sqrt(real**2 + imag**2)
        score_mag = torch.sigmoid(torch.tensor(mag)).item()
        
        # Real-only approach (ignores imaginary rotation)
        score_real = torch.sigmoid(torch.tensor(real)).item()
        
        # Exponential decay from perfect match (1, 0)
        # Distance from ideal: sqrt((1-real)^2 + imag^2)
        dist = math.sqrt((1 - real)**2 + imag**2)
        k = 2.0  # decay rate
        score_exp_dist = math.exp(-k * dist)
        
        # Exponential with just real part: exp(-k * (1 - real))
        # When real=+1 → exp(0)=1, when real=-1 → exp(-2k)≈0
        score_exp_real = math.exp(-k * (1 - real))
        
        print(f"{name:<25} | {real:6.2f} {imag:6.2f} | {score_mag:6.3f} | {score_real:9.3f} | {score_exp_dist:8.3f} | {score_exp_real:8.3f}")
    
    print("\n" + "="*80)
    print("ANALYSIS:")
    print("  - Magnitude: HIGH for both constructive (+1,0) AND destructive (-1,0) → WRONG")
    print("  - Real-only: Correct sign, but ignores imaginary (rotation)")
    print("  - Exp(distance): Peaks at (1,0), decays with distance → CORRECT for complex match")
    print("  - Exp(1-real): Simpler, focuses on real alignment → Good approximation")
    print("="*80)


def test_learning_dynamics():
    """Train for a few epochs and track how interference scores evolve."""
    print("\n" + "="*80)
    print("LEARNING DYNAMICS - INTERFERENCE EVOLUTION")
    print("="*80)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    vocab_size = 64
    marker = vocab_size - 1
    seq_len = 16
    batch_size = 32
    
    model = EchoChamberModel(
        vocab_size=vocab_size,
        d_model=64,
        num_layers=1,
        num_neurons=64,
        n_echo_heads=4,
        fusion_mode="additive",
    ).to(device)
    
    print(f"Parameters: {model.get_num_params():,}")
    print(f"Task: Marker retrieval (store value after marker, recall at end)")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.1)
    
    # Track epochs: 0, 10, 25, 50, 100, 200
    track_epochs = [0, 10, 25, 50, 100, 200]
    
    for epoch in range(max(track_epochs) + 1):
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
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        # Track diagnostics at specific epochs
        if epoch in track_epochs:
            model.eval()
            with torch.no_grad():
                # Single sequence for detailed analysis
                test_seq = seq[:1]
                test_target = targets[:1]
                
                logits, diag = model(test_seq, return_diagnostics=True)
                pred = logits[:, -1, :].argmax(dim=-1)
                correct = (pred == test_target).item()
                
                # Collect interference scores
                int_reals = []
                int_imags = []
                norm_reals = []
                norm_imags = []
                distances = []
                alphas = []
                
                for step in diag:
                    echo = step['layer0_echo']
                    hd = echo['head_details']
                    
                    int_real = sum(h['interference_real'].mean().item() for h in hd) / len(hd)
                    int_imag = sum(h['interference_imag'].mean().item() for h in hd) / len(hd)
                    norm_r = sum(h['norm_int_real'].mean().item() for h in hd) / len(hd)
                    norm_i = sum(h['norm_int_imag'].mean().item() for h in hd) / len(hd)
                    dist = sum(h['distance_from_ideal'].mean().item() for h in hd) / len(hd)
                    alpha = echo['alpha'].mean().item()
                    
                    int_reals.append(int_real)
                    int_imags.append(int_imag)
                    norm_reals.append(norm_r)
                    norm_imags.append(norm_i)
                    distances.append(dist)
                    alphas.append(alpha)
                
                # Stats
                alpha_mean = sum(alphas) / len(alphas)
                alpha_std = (sum((a - alpha_mean)**2 for a in alphas) / len(alphas)) ** 0.5
                norm_r_mean = sum(norm_reals) / len(norm_reals)
                norm_r_range = max(norm_reals) - min(norm_reals)
                dist_mean = sum(distances) / len(distances)
                
                print(f"\nEpoch {epoch:3d}: loss={loss.item():.4f}, correct={correct}")
                print(f"  α: mean={alpha_mean:.3f}, std={alpha_std:.3f}, range=[{min(alphas):.3f}, {max(alphas):.3f}]")
                print(f"  norm_real: mean={norm_r_mean:.3f}, range=[{min(norm_reals):.3f}, {max(norm_reals):.3f}] (target: near +1)")
                print(f"  norm_imag: range=[{min(norm_imags):.3f}, {max(norm_imags):.3f}] (target: near 0)")
                print(f"  distance: mean={dist_mean:.3f}, range=[{min(distances):.3f}, {max(distances):.3f}] (target: near 0)")
                
                # Check trigger magnitudes (are they changing?)
                trigger_mags = []
                for head in model.echo_chambers[0].heads:
                    t_mag = (head.trigger_real**2 + head.trigger_imag**2).sum().sqrt().item()
                    trigger_mags.append(t_mag)
                print(f"  trigger_mags: [{', '.join([f'{m:.3f}' for m in trigger_mags])}]")
                
                # Sample timesteps
                print(f"  Sample timesteps:")
                for t in [0, 7, 15]:
                    print(f"    t={t:2d}: α={alphas[t]:.3f}, norm_r={norm_reals[t]:6.3f}, norm_i={norm_imags[t]:6.3f}, dist={distances[t]:.3f}")
            
            model.train()
    
    print("\n" + "="*80)
    print("OBSERVATIONS:")
    print("  - Check if interference scores develop dynamic range")
    print("  - Check if alpha becomes sparse (high variance)")
    print("  - Check if distance from ideal decreases with learning")
    print("="*80)


def main():
    test_scoring_functions()
    test_learning_dynamics()


if __name__ == "__main__":
    main()

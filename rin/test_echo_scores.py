"""Quick test to analyze Echo Chamber interference scores."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

import sys
sys.path.insert(0, '/home/aiman/pi')

from rin.echo_chamber import EchoChamberModel

def main():
    print("="*80)
    print("ECHO CHAMBER INTERFERENCE SCORE ANALYSIS")
    print("="*80)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    
    model = EchoChamberModel(
        vocab_size=64,
        d_model=64,
        num_layers=1,
        num_neurons=64,
        n_echo_heads=4,
    ).to(device)
    
    print(f"Parameters: {model.get_num_params():,}")
    print(f"d_head = {64 // 4} = 16")
    
    # Single forward pass with diagnostics
    x = torch.randint(0, 64, (1, 20), device=device)
    
    with torch.no_grad():
        _, diag = model(x, return_diagnostics=True)
    
    print("\n" + "-"*110)
    print(f"{'t':>3} | {'α':>6} | {'int_real':>9} | {'int_imag':>9} | {'norm_r':>7} | {'norm_i':>7} | {'distance':>8} | {'temp':>5} | Scores")
    print("-"*110)
    
    all_norm_real = []
    all_norm_imag = []
    all_distances = []
    all_alphas = []
    
    for step in diag:
        t = step['t']
        echo = step['layer0_echo']
        alpha = echo['alpha'].mean().item()
        head_scores = echo['head_scores'].squeeze().tolist()
        
        # Get head details
        hd = echo['head_details']
        int_real = sum(h['interference_real'].mean().item() for h in hd) / len(hd)
        int_imag = sum(h['interference_imag'].mean().item() for h in hd) / len(hd)
        norm_r = sum(h['norm_int_real'].mean().item() for h in hd) / len(hd)
        norm_i = sum(h['norm_int_imag'].mean().item() for h in hd) / len(hd)
        dist = sum(h['distance_from_ideal'].mean().item() for h in hd) / len(hd)
        temp = hd[0]['temperature'].item()
        
        all_norm_real.append(norm_r)
        all_norm_imag.append(norm_i)
        all_distances.append(dist)
        all_alphas.append(alpha)
        
        head_str = ' '.join([f'{s:.2f}' for s in head_scores])
        bar = '█' * int(alpha * 20)
        print(f"{t:3d} | {alpha:.4f} | {int_real:9.4f} | {int_imag:9.4f} | {norm_r:7.3f} | {norm_i:7.3f} | {dist:8.4f} | {temp:5.2f} | [{head_str}] {bar}")
    
    print("-"*110)
    
    # Stats
    print(f"\nnorm_int_real: min={min(all_norm_real):.4f}, max={max(all_norm_real):.4f} (ideal=+1.0)")
    print(f"norm_int_imag: min={min(all_norm_imag):.4f}, max={max(all_norm_imag):.4f} (ideal=0.0)")
    print(f"distance:      min={min(all_distances):.4f}, max={max(all_distances):.4f} (ideal=0.0)")
    print(f"alpha:         min={min(all_alphas):.4f}, max={max(all_alphas):.4f}, range={max(all_alphas)-min(all_alphas):.4f}")
    
    # Trigger magnitudes
    print("\nTrigger magnitudes:")
    for i, head in enumerate(model.echo_chambers[0].heads):
        t_mag = (head.trigger_real**2 + head.trigger_imag**2).sum().sqrt().item()
        print(f"  Head {i}: {t_mag:.4f}")
    
    # Temperature
    temp = model.echo_chambers[0].heads[0].temperature.item()
    print(f"\nTemperature (decay rate k): {abs(temp) + 0.1:.2f}")
    
    # What we expect
    print(f"\nExponential decay analysis:")
    print(f"  score = exp(-k * distance)")
    print(f"  Perfect match (dist=0):     exp(0) = {math.exp(0):.4f}")
    print(f"  Good match (dist=0.5):      exp(-{(abs(temp)+0.1)*0.5:.2f}) = {math.exp(-(abs(temp)+0.1)*0.5):.4f}")
    print(f"  Moderate (dist=1.0):        exp(-{(abs(temp)+0.1)*1.0:.2f}) = {math.exp(-(abs(temp)+0.1)*1.0):.4f}")
    print(f"  Destructive (dist=2.0):     exp(-{(abs(temp)+0.1)*2.0:.2f}) = {math.exp(-(abs(temp)+0.1)*2.0):.4f}")
    
    print(f"\nWith current distance range [{min(all_distances):.3f}, {max(all_distances):.3f}]:")
    print(f"  → alpha range [{math.exp(-(abs(temp)+0.1)*max(all_distances)):.4f}, {math.exp(-(abs(temp)+0.1)*min(all_distances)):.4f}]")
    print(f"  Observed alpha range: [{min(all_alphas):.4f}, {max(all_alphas):.4f}]")

if __name__ == "__main__":
    main()

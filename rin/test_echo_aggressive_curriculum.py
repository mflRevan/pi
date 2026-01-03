#!/usr/bin/env python3
"""
Echo Chamber V2 - Aggressive Curriculum Learning

Very aggressive curriculum:
1. Train on distances 2-8 for 500 epochs (build strong foundations)
2. Suddenly jump to distance 60 and continue training

Configuration:
- lr = 0.1 (high)
- weight_decay = 0.0001 (very low, allow strong memory)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
sys.path.insert(0, '/home/aiman/pi')

from rin.echo_chamber_v2 import EchoChamberV2


def evaluate_distance(model, out_proj, delay: int, batch_size: int = 32, device='cuda'):
    """Evaluate model on a specific delay."""
    model.eval()
    
    seq_len = delay + 5
    value_pos = 2
    target_pos = value_pos + delay
    
    with torch.no_grad():
        model.reset_memory(batch_size, device)
        test_real = torch.randn(batch_size, model.d_model, device=device)
        test_imag = torch.randn(batch_size, model.d_model, device=device)
        
        for t in range(seq_len):
            if t == value_pos:
                x_real = test_real
                x_imag = test_imag
            else:
                x_real = torch.randn(batch_size, model.d_model, device=device) * 0.1
                x_imag = torch.randn(batch_size, model.d_model, device=device) * 0.1
            out_real, _, _ = model(x_real, x_imag, torch.tensor([t], device=device))
        
        pred = out_proj(out_real)
        corr = F.cosine_similarity(pred, test_real, dim=-1).mean().item()
    
    model.train()
    return corr


def train_epoch(model, out_proj, optimizer, delays, batch_size: int = 32, device='cuda'):
    """Train for one epoch, sampling from a list of delays."""
    model.train()
    
    # Sample a random delay from the list
    delay = delays[torch.randint(len(delays), (1,)).item()]
    
    seq_len = delay + 5
    value_pos = 2
    target_pos = value_pos + delay
    
    model.reset_memory(batch_size, device)
    optimizer.zero_grad()
    
    value_real = torch.randn(batch_size, model.d_model, device=device)
    value_imag = torch.randn(batch_size, model.d_model, device=device)
    
    outputs = []
    for t in range(seq_len):
        if t == value_pos:
            x_real = value_real
            x_imag = value_imag
        else:
            x_real = torch.randn(batch_size, model.d_model, device=device) * 0.1
            x_imag = torch.randn(batch_size, model.d_model, device=device) * 0.1
        
        out_real, out_imag, _ = model(x_real, x_imag, torch.tensor([t], device=device))
        outputs.append(out_real)
    
    pred = out_proj(outputs[target_pos])
    loss = F.mse_loss(pred, value_real)
    loss.backward()
    
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    torch.nn.utils.clip_grad_norm_(out_proj.parameters(), 1.0)
    
    optimizer.step()
    
    return loss.item(), delay


def aggressive_curriculum():
    """Aggressive curriculum: 2-8 steps for 500 epochs, then jump to 60."""
    
    print("\n" + "="*70)
    print("AGGRESSIVE CURRICULUM LEARNING")
    print("="*70)
    print("\nConfiguration:")
    print("  - Learning rate: 0.1")
    print("  - Weight decay: 0.0001 (very low)")
    print("  - Phase 1: Distance 2-8 for 500 epochs")
    print("  - Phase 2: Distance 60 for 500 epochs")
    print("="*70)
    
    torch.manual_seed(42)
    
    d_model = 64
    batch_size = 32
    
    # Create model
    model = EchoChamberV2(d_model=d_model, n_heads=4, detach_memory=False)
    out_proj = nn.Linear(d_model, d_model)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    out_proj = out_proj.to(device)
    
    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(out_proj.parameters()),
        lr=0.1,
        weight_decay=0.0001
    )
    
    print(f"\nDevice: {device}")
    
    # Initial state
    decay_init = model.compute_decay().mean().item()
    print(f"Initial decay: {decay_init:.4f}")
    
    # PHASE 1: Short distances (2-8)
    print("\n" + "="*70)
    print("PHASE 1: Training on distances 2-8")
    print("="*70)
    
    short_distances = [2, 3, 4, 5, 6, 7, 8]
    phase1_epochs = 500
    
    phase1_history = []
    
    for epoch in range(phase1_epochs):
        loss, trained_delay = train_epoch(model, out_proj, optimizer, short_distances, batch_size, device)
        
        # Evaluate every 25 epochs
        if epoch % 25 == 0 or epoch == phase1_epochs - 1:
            # Evaluate on all short distances
            short_corrs = {}
            for d in short_distances:
                corr = evaluate_distance(model, out_proj, d, batch_size, device)
                short_corrs[d] = corr
            
            # Also evaluate on distance 60 to see baseline
            corr_60 = evaluate_distance(model, out_proj, 60, batch_size, device)
            
            decay_current = model.compute_decay().mean().item()
            persistence_short = (model.compute_decay() ** 8).mean().item()
            persistence_long = (model.compute_decay() ** 60).mean().item()
            
            avg_short_corr = sum(short_corrs.values()) / len(short_corrs)
            
            phase1_history.append({
                'epoch': epoch,
                'loss': loss,
                'avg_short_corr': avg_short_corr,
                'corr_60': corr_60,
                'decay': decay_current,
                'persistence_8': persistence_short,
                'persistence_60': persistence_long,
                'short_corrs': short_corrs,
            })
            
            print(f"Epoch {epoch:3d}: loss={loss:.4f}, avg_short={avg_short_corr:.4f}, "
                  f"d60={corr_60:.4f}, decay={decay_current:.4f}, "
                  f"pers8={persistence_short:.4f}, pers60={persistence_long:.4f}")
    
    # Summary of Phase 1
    print("\n" + "-"*70)
    print("PHASE 1 SUMMARY")
    print("-"*70)
    
    final_p1 = phase1_history[-1]
    print(f"Average short-distance correlation: {final_p1['avg_short_corr']:.4f}")
    print(f"Individual distances:")
    for d in short_distances:
        print(f"  Distance {d}: {final_p1['short_corrs'][d]:.4f}")
    print(f"\nDistance 60 (baseline before training): {final_p1['corr_60']:.4f}")
    print(f"Decay: {decay_init:.4f} → {final_p1['decay']:.4f} (Δ={final_p1['decay'] - decay_init:+.4f})")
    print(f"Persistence (8 steps): {final_p1['persistence_8']:.4f} ({final_p1['persistence_8']:.2%})")
    print(f"Persistence (60 steps): {final_p1['persistence_60']:.4f} ({final_p1['persistence_60']:.2%})")
    
    # PHASE 2: Long distance (60)
    print("\n" + "="*70)
    print("PHASE 2: SWITCHING TO DISTANCE 60")
    print("="*70)
    print("Testing if model can adapt from short-term to long-term memory...\n")
    
    long_distances = [60]
    phase2_epochs = 500
    
    phase2_history = []
    
    for epoch in range(phase2_epochs):
        loss, trained_delay = train_epoch(model, out_proj, optimizer, long_distances, batch_size, device)
        
        # Evaluate every 25 epochs
        if epoch % 25 == 0 or epoch == phase2_epochs - 1:
            # Evaluate on distance 60
            corr_60 = evaluate_distance(model, out_proj, 60, batch_size, device)
            
            # Check retention on short distances
            short_corrs = {}
            for d in [2, 4, 6, 8]:
                corr = evaluate_distance(model, out_proj, d, batch_size, device)
                short_corrs[d] = corr
            
            decay_current = model.compute_decay().mean().item()
            persistence_short = (model.compute_decay() ** 8).mean().item()
            persistence_long = (model.compute_decay() ** 60).mean().item()
            
            avg_short_corr = sum(short_corrs.values()) / len(short_corrs)
            
            phase2_history.append({
                'epoch': epoch,
                'loss': loss,
                'corr_60': corr_60,
                'avg_short_corr': avg_short_corr,
                'decay': decay_current,
                'persistence_8': persistence_short,
                'persistence_60': persistence_long,
                'short_corrs': short_corrs,
            })
            
            print(f"Epoch {epoch:3d}: loss={loss:.4f}, d60={corr_60:.4f}, "
                  f"avg_short={avg_short_corr:.4f}, decay={decay_current:.4f}, "
                  f"pers60={persistence_long:.4f}")
    
    # Final Summary
    print("\n" + "="*70)
    print("FINAL SUMMARY")
    print("="*70)
    
    final_p2 = phase2_history[-1]
    
    print(f"\nPhase 1 → Phase 2 Comparison:")
    print(f"{'Metric':<25} {'Phase 1':<15} {'Phase 2':<15} {'Change':<15}")
    print("-"*70)
    print(f"{'Decay':<25} {final_p1['decay']:<15.4f} {final_p2['decay']:<15.4f} {final_p2['decay'] - final_p1['decay']:+.4f}")
    print(f"{'Persistence (8 steps)':<25} {final_p1['persistence_8']:<15.4f} {final_p2['persistence_8']:<15.4f} {final_p2['persistence_8'] - final_p1['persistence_8']:+.4f}")
    print(f"{'Persistence (60 steps)':<25} {final_p1['persistence_60']:<15.4f} {final_p2['persistence_60']:<15.4f} {final_p2['persistence_60'] - final_p1['persistence_60']:+.4f}")
    print(f"{'Distance 60 corr':<25} {final_p1['corr_60']:<15.4f} {final_p2['corr_60']:<15.4f} {final_p2['corr_60'] - final_p1['corr_60']:+.4f}")
    print(f"{'Avg short distances':<25} {final_p1['avg_short_corr']:<15.4f} {final_p2['avg_short_corr']:<15.4f} {final_p2['avg_short_corr'] - final_p1['avg_short_corr']:+.4f}")
    
    print(f"\n--- Adaptation Analysis ---")
    if final_p2['decay'] > final_p1['decay']:
        print(f"✓ Decay INCREASED by {final_p2['decay'] - final_p1['decay']:.4f} (slower forgetting)")
    else:
        print(f"✗ Decay DECREASED by {final_p1['decay'] - final_p2['decay']:.4f} (faster forgetting)")
    
    if final_p2['persistence_60'] > final_p1['persistence_60']:
        improvement = (final_p2['persistence_60'] - final_p1['persistence_60']) / final_p1['persistence_60'] * 100
        print(f"✓ 60-step persistence IMPROVED by {improvement:.1f}%")
    else:
        print(f"✗ 60-step persistence did not improve")
    
    if final_p2['corr_60'] > 0.3:
        print(f"✓ Distance 60 achieved GOOD correlation ({final_p2['corr_60']:.4f})")
    elif final_p2['corr_60'] > 0.15:
        print(f"~ Distance 60 achieved MODERATE correlation ({final_p2['corr_60']:.4f})")
    else:
        print(f"✗ Distance 60 has WEAK correlation ({final_p2['corr_60']:.4f})")
    
    if final_p2['avg_short_corr'] > 0.5:
        print(f"✓ Short distances RETAINED well ({final_p2['avg_short_corr']:.4f})")
    elif final_p2['avg_short_corr'] > 0.3:
        print(f"~ Short distances PARTIALLY retained ({final_p2['avg_short_corr']:.4f})")
    else:
        print(f"✗ Short distances FORGOTTEN ({final_p2['avg_short_corr']:.4f})")
    
    # Learning curves
    print(f"\n--- Phase 2 Learning Curve (Distance 60) ---")
    print(f"Epoch   0: {phase2_history[0]['corr_60']:.4f}")
    print(f"Epoch 125: {phase2_history[5]['corr_60']:.4f}")
    print(f"Epoch 250: {phase2_history[10]['corr_60']:.4f}")
    print(f"Epoch 375: {phase2_history[15]['corr_60']:.4f}")
    print(f"Epoch 499: {phase2_history[-1]['corr_60']:.4f}")
    
    return phase1_history, phase2_history


if __name__ == "__main__":
    phase1_hist, phase2_hist = aggressive_curriculum()

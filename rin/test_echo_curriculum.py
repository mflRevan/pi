#!/usr/bin/env python3
"""
Echo Chamber V2 - Curriculum Learning

Start at distance ~10, gradually increase to 80 as model learns.
Monitor learning behavior and adaptation.
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


def train_at_distance(model, out_proj, optimizer, delay: int, epochs: int, 
                     batch_size: int = 32, device='cuda', verbose=True):
    """Train model at a specific delay for a number of epochs."""
    
    seq_len = delay + 5
    value_pos = 2
    target_pos = value_pos + delay
    
    best_corr = -1
    
    for epoch in range(epochs):
        model.train()
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
        
        # Eval every 10 epochs
        if epoch % 10 == 0 or epoch == epochs - 1:
            corr = evaluate_distance(model, out_proj, delay, batch_size, device)
            best_corr = max(best_corr, corr)
            
            if verbose and (epoch % 50 == 0 or epoch == epochs - 1):
                decay = model.compute_decay().mean().item()
                persistence = (model.compute_decay() ** delay).mean().item()
                print(f"  Epoch {epoch:3d}: loss={loss.item():.4f}, corr={corr:.4f}, "
                      f"decay={decay:.4f}, pers={persistence:.4f}")
    
    return best_corr


def curriculum_learning():
    """Curriculum learning: gradually increase distance from 10 to 80."""
    
    print("\n" + "="*70)
    print("CURRICULUM LEARNING - Distance 10 → 80")
    print("="*70)
    print("\nConfiguration:")
    print("  - Starting distance: 10")
    print("  - Target distance: 80")
    print("  - Mastery threshold: corr > 0.5")
    print("  - Learning rate: 0.1")
    print("  - Weight decay: 0.01")
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
        weight_decay=0.01
    )
    
    # Curriculum stages
    curriculum = [
        (10, 100, 0.5),   # (distance, max_epochs, threshold)
        (15, 100, 0.5),
        (20, 100, 0.45),
        (25, 150, 0.4),
        (30, 150, 0.35),
        (40, 200, 0.3),
        (50, 200, 0.25),
        (60, 200, 0.2),
        (70, 250, 0.15),
        (80, 300, 0.1),
    ]
    
    history = []
    
    print(f"\nInitial decay: {model.compute_decay().mean():.4f}\n")
    
    for stage_idx, (delay, max_epochs, threshold) in enumerate(curriculum):
        print(f"\n{'='*70}")
        print(f"STAGE {stage_idx + 1}: Distance = {delay} (threshold: {threshold:.2f})")
        print('='*70)
        
        # Get initial performance
        init_corr = evaluate_distance(model, out_proj, delay, batch_size, device)
        decay_before = model.compute_decay().mean().item()
        persistence_before = (model.compute_decay() ** delay).mean().item()
        
        print(f"Initial performance: corr={init_corr:.4f}")
        print(f"Decay: {decay_before:.4f}, Persistence ({delay} steps): {persistence_before:.4f}")
        print(f"\nTraining for up to {max_epochs} epochs...")
        
        # Train
        best_corr = train_at_distance(model, out_proj, optimizer, delay, max_epochs, 
                                      batch_size, device, verbose=True)
        
        # Post-training evaluation
        decay_after = model.compute_decay().mean().item()
        persistence_after = (model.compute_decay() ** delay).mean().item()
        
        # Evaluate on all previous distances to check retention
        retention = {}
        for prev_delay in [10, 15, 20, 25, 30, 40, 50, 60, 70]:
            if prev_delay < delay:
                ret_corr = evaluate_distance(model, out_proj, prev_delay, batch_size, device)
                retention[prev_delay] = ret_corr
        
        stage_result = {
            'stage': stage_idx + 1,
            'delay': delay,
            'threshold': threshold,
            'init_corr': init_corr,
            'best_corr': best_corr,
            'decay_before': decay_before,
            'decay_after': decay_after,
            'persistence_before': persistence_before,
            'persistence_after': persistence_after,
            'retention': retention,
        }
        history.append(stage_result)
        
        print(f"\n--- Stage {stage_idx + 1} Summary ---")
        print(f"Initial → Best: {init_corr:.4f} → {best_corr:.4f} (Δ={best_corr - init_corr:+.4f})")
        print(f"Decay: {decay_before:.4f} → {decay_after:.4f} (Δ={decay_after - decay_before:+.4f})")
        print(f"Persistence: {persistence_before:.4f} → {persistence_after:.4f}")
        
        if retention:
            print(f"\nRetention on previous distances:")
            for prev_d, prev_c in sorted(retention.items()):
                print(f"  Distance {prev_d}: {prev_c:.4f}")
        
        # Check if we met threshold
        if best_corr >= threshold:
            print(f"\n✓ Mastery achieved! (corr={best_corr:.4f} >= {threshold:.2f})")
        else:
            print(f"\n⚠ Below threshold (corr={best_corr:.4f} < {threshold:.2f})")
            if best_corr < 0.1:
                print("⚠ Very weak learning - may not generalize to longer distances")
    
    # Final summary
    print("\n" + "="*70)
    print("CURRICULUM LEARNING SUMMARY")
    print("="*70)
    print(f"{'Stage':<8} {'Delay':<8} {'Init→Best':<20} {'Decay Δ':<12} {'Status':<10}")
    print("-"*70)
    
    for h in history:
        corr_str = f"{h['init_corr']:.3f}→{h['best_corr']:.3f}"
        decay_delta = h['decay_after'] - h['decay_before']
        
        if h['best_corr'] >= h['threshold']:
            status = "✓ PASS"
        else:
            status = "✗ FAIL"
        
        print(f"{h['stage']:<8} {h['delay']:<8} {corr_str:<20} {decay_delta:+.4f}      {status:<10}")
    
    # Final decay analysis
    print(f"\n--- Decay Evolution ---")
    print(f"Initial decay: {history[0]['decay_before']:.4f}")
    print(f"Final decay: {history[-1]['decay_after']:.4f}")
    print(f"Total change: {history[-1]['decay_after'] - history[0]['decay_before']:+.4f}")
    
    # Check if model learned to increase memory with distance
    print(f"\n--- Memory Adaptation ---")
    for h in history[1:]:
        prev_decay = history[h['stage'] - 2]['decay_after']
        curr_decay = h['decay_after']
        change = curr_decay - prev_decay
        
        if change > 0.01:
            direction = "↑ SLOWER decay (longer memory)"
        elif change < -0.01:
            direction = "↓ FASTER decay (shorter memory)"
        else:
            direction = "→ Stable"
        
        print(f"Stage {h['stage']} (delay={h['delay']}): {direction} (Δ={change:+.4f})")
    
    return history


if __name__ == "__main__":
    history = curriculum_learning()

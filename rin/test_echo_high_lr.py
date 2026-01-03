#!/usr/bin/env python3
"""
Echo Chamber V2 - High Learning Rate Test

Test with lr=0.1 (10x normal) and compare weight decay 0.1 vs 0.01
at distance 50.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
sys.path.insert(0, '/home/aiman/pi')

from rin.echo_chamber_v2 import EchoChamberV2


def test_high_lr(weight_decay: float, epochs: int = 300):
    """Test needle retrieval at distance 50 with high lr."""
    
    print(f"\n{'='*70}")
    print(f"Testing: lr=0.1, weight_decay={weight_decay}, delay=50")
    print('='*70)
    
    torch.manual_seed(42)
    
    d_model = 64
    batch_size = 32
    delay = 50
    seq_len = delay + 5
    value_pos = 2
    target_pos = value_pos + delay
    
    # Create model with full BPTT
    model = EchoChamberV2(d_model=d_model, n_heads=4, detach_memory=False)
    out_proj = nn.Linear(d_model, d_model)
    
    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(out_proj.parameters()),
        lr=0.1,  # 10x higher!
        weight_decay=weight_decay
    )
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    out_proj = out_proj.to(device)
    
    # Initial state
    decay_init = model.compute_decay()
    persistence_init = (decay_init ** delay).mean().item()
    
    print(f"Initial decay: {decay_init.mean():.4f}")
    print(f"Initial {delay}-step persistence: {persistence_init:.4f} ({persistence_init:.2%})")
    print(f"\nStarting training...")
    print("-"*70)
    
    best_corr = -1
    best_epoch = -1
    history = []
    
    for epoch in range(epochs):
        model.train()
        model.reset_memory(batch_size, device)
        optimizer.zero_grad()
        
        value_real = torch.randn(batch_size, d_model, device=device)
        value_imag = torch.randn(batch_size, d_model, device=device)
        
        outputs = []
        for t in range(seq_len):
            if t == value_pos:
                x_real = value_real
                x_imag = value_imag
            else:
                x_real = torch.randn(batch_size, d_model, device=device) * 0.1
                x_imag = torch.randn(batch_size, d_model, device=device) * 0.1
            
            out_real, out_imag, _ = model(x_real, x_imag, torch.tensor([t], device=device))
            outputs.append(out_real)
        
        pred = out_proj(outputs[target_pos])
        loss = F.mse_loss(pred, value_real)
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        torch.nn.utils.clip_grad_norm_(out_proj.parameters(), 1.0)
        
        optimizer.step()
        
        # Eval every 10 epochs
        if epoch % 10 == 0 or epoch == epochs - 1:
            model.eval()
            with torch.no_grad():
                model.reset_memory(batch_size, device)
                test_real = torch.randn(batch_size, d_model, device=device)
                test_imag = torch.randn(batch_size, d_model, device=device)
                
                for t in range(seq_len):
                    if t == value_pos:
                        x_real = test_real
                        x_imag = test_imag
                    else:
                        x_real = torch.randn(batch_size, d_model, device=device) * 0.1
                        x_imag = torch.randn(batch_size, d_model, device=device) * 0.1
                    out_real, _, _ = model(x_real, x_imag, torch.tensor([t], device=device))
                
                pred = out_proj(out_real)
                corr = F.cosine_similarity(pred, test_real, dim=-1).mean().item()
                
            if corr > best_corr:
                best_corr = corr
                best_epoch = epoch
            
            decay_current = model.compute_decay()
            persistence_current = (decay_current ** delay).mean().item()
            
            history.append({
                'epoch': epoch,
                'loss': loss.item(),
                'corr': corr,
                'decay': decay_current.mean().item(),
                'persistence': persistence_current
            })
            
            print(f"Epoch {epoch:3d}: loss={loss.item():.4f}, corr={corr:.4f}, "
                  f"decay={decay_current.mean():.4f}, pers={persistence_current:.4f}")
            
            model.train()
    
    # Final report
    decay_final = model.compute_decay()
    persistence_final = (decay_final ** delay).mean().item()
    
    print(f"\n{'='*70}")
    print("RESULTS")
    print('='*70)
    print(f"Best correlation: {best_corr:.4f} at epoch {best_epoch}")
    print(f"Final correlation: {history[-1]['corr']:.4f}")
    print(f"\nDecay: {decay_init.mean():.4f} → {decay_final.mean():.4f} ({decay_final.mean().item() - decay_init.mean().item():+.4f})")
    print(f"Persistence: {persistence_init:.4f} → {persistence_final:.4f} ({persistence_final - persistence_init:+.4f})")
    
    if best_corr > 0.5:
        status = "✓ PASS"
    elif best_corr > 0.3:
        status = "~ PARTIAL"
    else:
        status = "✗ FAIL"
    print(f"\nStatus: {status}")
    
    return best_corr, history


def main():
    print("\n" + "#"*70)
    print("# HIGH LEARNING RATE TEST - Distance 50")
    print("#"*70)
    print("\nConfiguration: lr=0.1 (10x normal)")
    print("Testing weight decay: 0.1 and 0.01")
    
    # Test 1: weight_decay=0.1
    corr_01, hist_01 = test_high_lr(weight_decay=0.1, epochs=300)
    
    # Test 2: weight_decay=0.01
    corr_001, hist_001 = test_high_lr(weight_decay=0.01, epochs=300)
    
    # Comparison
    print("\n" + "="*70)
    print("COMPARISON")
    print("="*70)
    print(f"{'Weight Decay':<15} {'Best Corr':<15} {'Status':<15}")
    print("-"*45)
    print(f"{'0.1':<15} {corr_01:<15.4f} {'✓ PASS' if corr_01 > 0.5 else ('~ PARTIAL' if corr_01 > 0.3 else '✗ FAIL'):<15}")
    print(f"{'0.01':<15} {corr_001:<15.4f} {'✓ PASS' if corr_001 > 0.5 else ('~ PARTIAL' if corr_001 > 0.3 else '✗ FAIL'):<15}")
    
    if corr_001 > corr_01:
        print(f"\n→ Lower weight decay (0.01) performed better by {corr_001 - corr_01:.4f}")
    elif corr_01 > corr_001:
        print(f"\n→ Higher weight decay (0.1) performed better by {corr_01 - corr_001:.4f}")
    else:
        print("\n→ Both performed equally")


if __name__ == "__main__":
    main()

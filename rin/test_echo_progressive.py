#!/usr/bin/env python3
"""
Echo Chamber V2 - Progressive Distance Testing

Test multiple distances with weight decay 0.1 to find the breaking point.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
sys.path.insert(0, '/home/aiman/pi')

from rin.echo_chamber_v2 import EchoChamberV2


def test_distance(delay: int, epochs: int = 200, weight_decay: float = 0.1):
    """Test a single distance."""
    
    torch.manual_seed(42)
    
    d_model = 64
    batch_size = 32
    seq_len = delay + 5
    value_pos = 2
    target_pos = value_pos + delay
    
    # Create model with full BPTT
    model = EchoChamberV2(d_model=d_model, n_heads=4, detach_memory=False)
    out_proj = nn.Linear(d_model, d_model)
    
    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(out_proj.parameters()),
        lr=1e-2,
        weight_decay=weight_decay
    )
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    out_proj = out_proj.to(device)
    
    # Initial state
    decay_init = model.compute_decay().mean().item()
    persistence_init = (model.compute_decay() ** delay).mean().item()
    
    best_corr = -1
    
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
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        torch.nn.utils.clip_grad_norm_(out_proj.parameters(), 1.0)
        
        optimizer.step()
        
        # Eval every 50 epochs
        if epoch % 50 == 0 or epoch == epochs - 1:
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
                
            best_corr = max(best_corr, corr)
            model.train()
    
    # Final stats
    decay_final = model.compute_decay().mean().item()
    persistence_final = (model.compute_decay() ** delay).mean().item()
    
    return {
        'delay': delay,
        'best_corr': best_corr,
        'decay_init': decay_init,
        'decay_final': decay_final,
        'persistence_init': persistence_init,
        'persistence_final': persistence_final,
    }


def test_progressive_distances():
    """Test multiple distances to find breaking point."""
    
    print("\n" + "="*70)
    print("PROGRESSIVE DISTANCE TESTING - Weight Decay 0.1")
    print("="*70)
    
    distances = [10, 15, 20, 25, 30, 35, 40, 50]
    results = []
    
    for delay in distances:
        print(f"\n--- Testing Distance {delay} ---")
        result = test_distance(delay, epochs=200, weight_decay=0.1)
        results.append(result)
        
        print(f"  Best correlation: {result['best_corr']:.4f}")
        print(f"  Decay: {result['decay_init']:.4f} → {result['decay_final']:.4f}")
        print(f"  Persistence: {result['persistence_init']:.4f} → {result['persistence_final']:.4f}")
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"{'Delay':<8} {'Corr':<10} {'Decay→':<12} {'Persist→':<12} {'Status':<15}")
    print("-"*70)
    
    for r in results:
        if r['best_corr'] > 0.5:
            status = "✓ PASS"
        elif r['best_corr'] > 0.3:
            status = "~ PARTIAL"
        else:
            status = "✗ FAIL"
        
        decay_str = f"{r['decay_init']:.3f}→{r['decay_final']:.3f}"
        persist_str = f"{r['persistence_init']:.3f}→{r['persistence_final']:.3f}"
        
        print(f"{r['delay']:<8} {r['best_corr']:<10.4f} {decay_str:<12} {persist_str:<12} {status:<15}")
    
    # Find breaking point
    passing = [r for r in results if r['best_corr'] > 0.3]
    if passing:
        max_passing = max(passing, key=lambda x: x['delay'])
        print(f"\nMaximum working distance: {max_passing['delay']} (corr={max_passing['best_corr']:.4f})")
    else:
        print(f"\nNo distances achieved corr > 0.3")


if __name__ == "__main__":
    test_progressive_distances()

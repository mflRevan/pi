#!/usr/bin/env python3
"""
Full BPTT Test for Echo Chamber V2

Tests the Echo Chamber with memory gradient flow enabled.
This should enable learning long-term dependencies.

Key change: detach_memory=False allows gradients to flow through memory history.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
sys.path.insert(0, '/home/aiman/pi')

from rin.echo_chamber_v2 import EchoChamberV2


def test_echo_chamber_bptt(delay: int = 10, epochs: int = 200, verbose: bool = True):
    """Test Echo Chamber with full BPTT for a copy task with given delay."""
    
    torch.manual_seed(42)
    
    d_model = 64
    batch_size = 32
    seq_len = delay + 5
    value_pos = 2
    target_pos = value_pos + delay
    
    # Create model with detach_memory=False for full BPTT
    model = EchoChamberV2(d_model=d_model, n_heads=4, detach_memory=False)
    
    # Output projection to produce prediction
    out_proj = nn.Linear(d_model, d_model)
    
    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(out_proj.parameters()),
        lr=1e-2,
        weight_decay=0.01  # Crucial - force model to justify memory
    )
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    out_proj = out_proj.to(device)
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"Echo Chamber BPTT Test - Delay={delay}")
        print(f"{'='*60}")
        print(f"Device: {device}")
        print(f"Model: d_model={d_model}, n_heads=4, detach_memory=False")
        print(f"Sequence: value at pos {value_pos}, retrieve at pos {target_pos}")
        
        # Check initial decay
        decay = model.compute_decay()
        print(f"\nInitial decay: mean={decay.mean():.4f}, min={decay.min():.4f}, max={decay.max():.4f}")
        persistence = decay ** delay
        print(f"Expected {delay}-step persistence: mean={persistence.mean():.4f}")
    
    best_corr = -1
    best_epoch = -1
    
    for epoch in range(epochs):
        model.train()
        model.reset_memory(batch_size, device)
        optimizer.zero_grad()
        
        # Generate sequence: noise, VALUE, noise...
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
            
            # Echo Chamber uses complex inputs (real, imag)
            out_real, out_imag, _ = model(x_real, x_imag, torch.tensor([t], device=device))
            outputs.append(out_real)
        
        # Project final output
        pred = out_proj(outputs[target_pos])
        
        # Loss: retrieve the stored value
        loss = F.mse_loss(pred, value_real)
        
        loss.backward()
        
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        torch.nn.utils.clip_grad_norm_(out_proj.parameters(), 1.0)
        
        optimizer.step()
        
        # Evaluation
        if epoch % 20 == 0 or epoch == epochs - 1:
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
                
            if verbose:
                decay = model.compute_decay()
                print(f"Epoch {epoch:3d}: loss={loss.item():.4f}, corr={corr:.4f}, "
                      f"decay_mean={decay.mean():.4f}")
    
    if verbose:
        print(f"\nBest correlation: {best_corr:.4f} at epoch {best_epoch}")
        
        # Final diagnostics
        decay = model.compute_decay()
        print(f"\nFinal decay stats:")
        print(f"  Mean: {decay.mean():.4f}")
        print(f"  {delay}-step persistence: {(decay ** delay).mean():.4f}")
        
        # Check gradient flow
        print("\nParameter gradients (should be non-zero):")
        for name, param in model.named_parameters():
            if param.grad is not None:
                print(f"  {name}: {param.grad.abs().mean():.6f}")
    
    return best_corr


def test_multiple_delays():
    """Test Echo Chamber across multiple delays."""
    
    print("\n" + "#"*70)
    print("# ECHO CHAMBER V2 - FULL BPTT TEST")
    print("#"*70)
    
    delays = [2, 5, 10, 15, 20]
    results = {}
    
    for delay in delays:
        corr = test_echo_chamber_bptt(delay=delay, epochs=200, verbose=True)
        results[delay] = corr
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY - Echo Chamber with Full BPTT")
    print("="*70)
    print(f"{'Delay':<10} {'Correlation':<15} {'Status':<15}")
    print("-"*40)
    
    for delay, corr in results.items():
        if corr > 0.5:
            status = "✓ PASS"
        elif corr > 0.2:
            status = "~ PARTIAL"
        else:
            status = "✗ FAIL"
        print(f"{delay:<10} {corr:<15.4f} {status:<15}")
    
    return results


def compare_detach_modes():
    """Compare learning with and without memory detachment."""
    
    print("\n" + "#"*70)
    print("# COMPARISON: Detached vs Connected Memory")
    print("#"*70)
    
    delays = [5, 10]
    
    for delay in delays:
        print(f"\n{'='*60}")
        print(f"Delay = {delay}")
        print('='*60)
        
        # Test with detachment (original)
        torch.manual_seed(42)
        model_detached = EchoChamberV2(d_model=64, n_heads=4, detach_memory=True)
        print(f"\nWith detach_memory=True:")
        corr_detached = train_model(model_detached, delay=delay, epochs=100)
        
        # Test without detachment (full BPTT)
        torch.manual_seed(42)
        model_connected = EchoChamberV2(d_model=64, n_heads=4, detach_memory=False)
        print(f"\nWith detach_memory=False:")
        corr_connected = train_model(model_connected, delay=delay, epochs=100)
        
        print(f"\nResult: Detached={corr_detached:.4f}, Connected={corr_connected:.4f}, "
              f"Improvement={corr_connected - corr_detached:+.4f}")


def train_model(model, delay: int, epochs: int = 100):
    """Train a model and return best correlation."""
    
    d_model = 64
    batch_size = 32
    seq_len = delay + 5
    value_pos = 2
    target_pos = value_pos + delay
    
    out_proj = nn.Linear(d_model, d_model)
    
    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(out_proj.parameters()),
        lr=1e-2,
        weight_decay=0.01
    )
    
    device = 'cpu'
    model = model.to(device)
    out_proj = out_proj.to(device)
    
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
        loss = F.mse_loss(pred, value_real)  # Predict the real part
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        torch.nn.utils.clip_grad_norm_(out_proj.parameters(), 1.0)
        
        optimizer.step()
        
        # Quick eval
        if epoch % 25 == 0 or epoch == epochs - 1:
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
            print(f"  Epoch {epoch:3d}: loss={loss.item():.4f}, corr={corr:.4f}")
    
    return best_corr


if __name__ == "__main__":
    # Quick comparison first
    compare_detach_modes()
    
    # Full test across delays
    test_multiple_delays()

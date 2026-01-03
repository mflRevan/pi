#!/usr/bin/env python3
"""
Echo Chamber V2 - Needle Retrieval at Distance 50

Test with:
- AdamW weight decay = 0.1 (force strong memory justification)
- Learning rate = 1e-2
- Distance = 50 steps
- 300 epochs with detailed tracking
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
sys.path.insert(0, '/home/aiman/pi')

from rin.echo_chamber_v2 import EchoChamberV2


def test_needle_retrieval_50():
    """Test needle retrieval at distance 50 with high weight decay."""
    
    print("\n" + "="*70)
    print("ECHO CHAMBER V2 - NEEDLE RETRIEVAL AT DISTANCE 50")
    print("="*70)
    print("Configuration:")
    print("  - Weight decay: 0.1 (10x normal)")
    print("  - Learning rate: 1e-2")
    print("  - Delay: 50 steps")
    print("  - Epochs: 300")
    print("  - detach_memory: False (full BPTT)")
    print("="*70)
    
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
        lr=1e-2,
        weight_decay=0.1  # 10x normal - force strong justification
    )
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    out_proj = out_proj.to(device)
    
    print(f"\nDevice: {device}")
    
    # Check initial state
    decay = model.compute_decay()
    print(f"\nInitial decay: mean={decay.mean():.4f}, min={decay.min():.4f}, max={decay.max():.4f}")
    persistence = decay ** delay
    print(f"Expected {delay}-step persistence: mean={persistence.mean():.4f}")
    print(f"  (This means: {persistence.mean().item():.2%} of signal survives {delay} steps)")
    
    print(f"\nStarting training...")
    print("-"*70)
    
    # Track metrics
    history = {
        'epoch': [],
        'loss': [],
        'corr': [],
        'decay_mean': [],
        'persistence': [],
        'beta_mean': [],
    }
    
    best_corr = -1
    best_epoch = -1
    
    epochs = 300
    
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
        
        # Evaluation every 10 epochs
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
            
            # Get current stats
            decay_current = model.compute_decay()
            decay_mean = decay_current.mean().item()
            persistence_current = (decay_current ** delay).mean().item()
            beta_mean = model.get_beta().mean().item()
            
            # Track
            history['epoch'].append(epoch)
            history['loss'].append(loss.item())
            history['corr'].append(corr)
            history['decay_mean'].append(decay_mean)
            history['persistence'].append(persistence_current)
            history['beta_mean'].append(beta_mean)
            
            print(f"Epoch {epoch:3d}: loss={loss.item():.4f}, corr={corr:.4f}, "
                  f"decay={decay_mean:.4f}, pers={persistence_current:.4f} ({persistence_current:.2%})")
            
            model.train()
    
    # Final report
    print("\n" + "="*70)
    print("FINAL RESULTS")
    print("="*70)
    
    decay_final = model.compute_decay()
    beta_final = model.get_beta()
    persistence_final = (decay_final ** delay).mean().item()
    
    print(f"\nBest correlation: {best_corr:.4f} at epoch {best_epoch}")
    print(f"Final correlation: {history['corr'][-1]:.4f}")
    
    print(f"\nFinal decay statistics:")
    print(f"  Mean: {decay_final.mean():.4f}")
    print(f"  Min:  {decay_final.min():.4f}")
    print(f"  Max:  {decay_final.max():.4f}")
    print(f"  Std:  {decay_final.std():.4f}")
    
    print(f"\nFinal beta_eff statistics:")
    print(f"  Mean: {beta_final.mean():.4f}")
    print(f"  Min:  {beta_final.min():.4f}")
    print(f"  Max:  {beta_final.max():.4f}")
    
    print(f"\n{delay}-step persistence:")
    print(f"  Initial: {history['persistence'][0]:.4f} ({history['persistence'][0]:.2%})")
    print(f"  Final:   {persistence_final:.4f} ({persistence_final:.2%})")
    print(f"  Change:  {persistence_final - history['persistence'][0]:+.4f}")
    
    # Accuracy interpretation
    if best_corr > 0.7:
        status = "✓ EXCELLENT - Strong memory retention"
    elif best_corr > 0.5:
        status = "✓ GOOD - Clear memory signal"
    elif best_corr > 0.3:
        status = "~ MODERATE - Partial memory"
    elif best_corr > 0.1:
        status = "⚠ WEAK - Minimal memory"
    else:
        status = "✗ FAILED - No memory"
    
    print(f"\nStatus: {status}")
    
    # Check if model learned to slow decay
    decay_change = decay_final.mean().item() - history['decay_mean'][0]
    print(f"\nDecay adaptation: {decay_change:+.4f}")
    if decay_change > 0.01:
        print("  → Model learned to SLOW decay (increase memory)")
    elif decay_change < -0.01:
        print("  → Model learned to SPEED decay (reduce memory)")
    else:
        print("  → Decay remained stable")
    
    # Learning curve summary
    print(f"\nLearning curve:")
    print(f"  Epoch   0: corr={history['corr'][0]:.4f}")
    print(f"  Epoch  50: corr={history['corr'][5]:.4f}")
    print(f"  Epoch 100: corr={history['corr'][10]:.4f}")
    print(f"  Epoch 150: corr={history['corr'][15]:.4f}")
    print(f"  Epoch 200: corr={history['corr'][20]:.4f}")
    print(f"  Epoch 250: corr={history['corr'][25]:.4f}")
    print(f"  Epoch 299: corr={history['corr'][-1]:.4f}")
    
    # Gradient check
    print(f"\nFinal parameter gradients:")
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_mag = param.grad.abs().mean().item()
            if grad_mag > 1e-6:
                print(f"  {name:30s}: {grad_mag:.6f}")
    
    return history, best_corr


if __name__ == "__main__":
    history, best_corr = test_needle_retrieval_50()

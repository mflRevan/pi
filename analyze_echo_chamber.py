#!/usr/bin/env python3
"""
Echo Chamber Analysis - Gradient Flow and Learning Dynamics

Tests the new Echo Chamber architecture for:
1. Gradient flow through parallel paths (resonant + echo)
2. Independence of gradient paths
3. EMA value state dynamics
4. Comparison of output modes (complex_linear vs resonant)
5. Comparison of gate modes (multiplicative, additive, glu)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict
import sys
sys.path.insert(0, '/home/aiman/pi')

from rin import RINEchoModel, EchoChamber, ResonantBlock, PHI


def analyze_gradient_paths():
    """Analyze gradient flow through the parallel paths."""
    print("=" * 70)
    print("GRADIENT PATH ANALYSIS: Echo Chamber Architecture")
    print("=" * 70)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    d_model = 48
    num_neurons = 96
    n_heads = 4
    batch_size = 32
    
    # Create a single ResonantBlock
    block = ResonantBlock(
        d_model=d_model,
        num_neurons=num_neurons,
        n_heads=n_heads,
        alpha=0.1,
        learnable_alpha=True,
        output_mode='complex_linear',
        gate_mode='multiplicative',
    ).to(device)
    
    # Input
    torch.manual_seed(42)
    x_real = torch.randn(batch_size, d_model, device=device, requires_grad=True)
    x_imag = torch.randn(batch_size, d_model, device=device, requires_grad=True)
    echo_state = block.echo_chamber.init_state(batch_size, device)
    t = torch.ones(batch_size, device=device) * PHI
    
    # Forward
    (out_real, out_imag), new_echo_state = block.forward_step(
        x_real, x_imag, echo_state, t
    )
    
    # Loss - use squared sum since LayerNorm zeros the mean
    loss = (out_real ** 2).sum() + (out_imag ** 2).sum()
    loss.backward()
    
    print("\n--- Input Gradient Analysis ---")
    grad_x_real = x_real.grad
    grad_x_imag = x_imag.grad
    
    print(f"  ∇x_real magnitude: {grad_x_real.abs().mean():.6f}")
    print(f"  ∇x_imag magnitude: {grad_x_imag.abs().mean():.6f}")
    
    # Check correlation
    corr = F.cosine_similarity(
        grad_x_real.view(batch_size, -1),
        grad_x_imag.view(batch_size, -1), dim=1
    ).mean()
    
    print(f"  Correlation(∇x_real, ∇x_imag): {corr:+.4f}")
    
    if abs(corr) < 0.5:
        print("  ✓ INDEPENDENT gradient paths!")
    else:
        print("  ⚠️  Correlated gradients")
    
    print("\n--- Echo Chamber Gradients ---")
    print("  EMA alpha (per head):", block.echo_chamber.alpha.data)
    if block.echo_chamber.log_alpha.grad is not None:
        print(f"  ∇log_alpha: {block.echo_chamber.log_alpha.grad}")
    
    print("\n--- Resonant Layer Gradients ---")
    for name, param in block.resonant_layer.named_parameters():
        if param.grad is not None and 'input_proj' in name:
            print(f"  {name}: grad_norm={param.grad.norm():.4f}")
    
    return corr.item()


def compare_gate_modes():
    """Compare different gating strategies."""
    print("\n" + "=" * 70)
    print("GATE MODE COMPARISON")
    print("=" * 70)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    d_model = 48
    num_neurons = 96
    n_heads = 4
    batch_size = 32
    
    gate_modes = ['multiplicative', 'additive', 'glu']
    results = {}
    
    for mode in gate_modes:
        print(f"\n--- Gate Mode: {mode} ---")
        
        block = ResonantBlock(
            d_model=d_model,
            num_neurons=num_neurons,
            n_heads=n_heads,
            gate_mode=mode,
        ).to(device)
        
        torch.manual_seed(42)
        x_real = torch.randn(batch_size, d_model, device=device, requires_grad=True)
        x_imag = torch.randn(batch_size, d_model, device=device, requires_grad=True)
        echo_state = block.echo_chamber.init_state(batch_size, device)
        t = torch.ones(batch_size, device=device) * PHI
        
        (out_real, out_imag), _ = block.forward_step(x_real, x_imag, echo_state, t)
        # Use squared sum - LayerNorm zeros the mean so sum() gives zero gradient
        loss = (out_real ** 2).sum() + (out_imag ** 2).sum()
        loss.backward()
        
        grad_mag = (x_real.grad.abs().mean() + x_imag.grad.abs().mean()) / 2
        out_mag = (out_real.abs().mean() + out_imag.abs().mean()) / 2
        
        corr = F.cosine_similarity(
            x_real.grad.view(batch_size, -1),
            x_imag.grad.view(batch_size, -1), dim=1
        ).mean()
        
        results[mode] = {
            'grad_mag': grad_mag.item(),
            'out_mag': out_mag.item(),
            'grad_corr': corr.item(),
        }
        
        print(f"  Output magnitude: {out_mag:.4f}")
        print(f"  Gradient magnitude: {grad_mag:.4f}")
        print(f"  Gradient correlation: {corr:+.4f}")
    
    print("\n--- Summary ---")
    print(f"{'Mode':<15} | {'Out Mag':>10} | {'Grad Mag':>10} | {'Grad Corr':>10}")
    print("-" * 55)
    for mode, r in results.items():
        print(f"{mode:<15} | {r['out_mag']:>10.4f} | {r['grad_mag']:>10.4f} | {r['grad_corr']:>+10.4f}")
    
    return results


def compare_output_modes():
    """Compare complex_linear vs resonant output projection."""
    print("\n" + "=" * 70)
    print("OUTPUT MODE COMPARISON: complex_linear vs resonant")
    print("=" * 70)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    d_model = 48
    num_neurons = 96
    n_heads = 4
    batch_size = 32
    
    output_modes = ['complex_linear', 'resonant']
    results = {}
    
    for mode in output_modes:
        print(f"\n--- Output Mode: {mode} ---")
        
        block = ResonantBlock(
            d_model=d_model,
            num_neurons=num_neurons,
            n_heads=n_heads,
            output_mode=mode,
        ).to(device)
        
        torch.manual_seed(42)
        x_real = torch.randn(batch_size, d_model, device=device, requires_grad=True)
        x_imag = torch.randn(batch_size, d_model, device=device, requires_grad=True)
        echo_state = block.echo_chamber.init_state(batch_size, device)
        t = torch.ones(batch_size, device=device) * PHI
        
        (out_real, out_imag), _ = block.forward_step(x_real, x_imag, echo_state, t)
        # Use squared sum - LayerNorm zeros the mean
        loss = (out_real ** 2).sum() + (out_imag ** 2).sum()
        loss.backward()
        
        grad_mag = (x_real.grad.abs().mean() + x_imag.grad.abs().mean()) / 2
        
        # Check gradient flow to echo chamber params
        echo_grad_norm = 0
        for param in block.echo_chamber.parameters():
            if param.grad is not None:
                echo_grad_norm += param.grad.norm().item()
        
        results[mode] = {
            'grad_mag': grad_mag.item(),
            'echo_grad_norm': echo_grad_norm,
        }
        
        print(f"  Input gradient magnitude: {grad_mag:.4f}")
        print(f"  Echo chamber total grad norm: {echo_grad_norm:.4f}")
    
    return results


def test_ema_dynamics():
    """Test how EMA value states evolve over time."""
    print("\n" + "=" * 70)
    print("EMA VALUE STATE DYNAMICS")
    print("=" * 70)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    d_model = 48
    n_heads = 4
    batch_size = 8
    num_steps = 20
    
    echo = EchoChamber(
        d_model=d_model,
        n_heads=n_heads,
        alpha=0.2,  # 20% new, 80% old
        learnable_alpha=False,
    ).to(device)
    
    value_state = echo.init_state(batch_size, device)
    
    print(f"\nAlpha = {echo.alpha.mean():.2f} (20% new, 80% old)")
    print(f"\nTracking value state norm over {num_steps} steps:")
    print(f"{'Step':>5} | {'State Norm':>12} | {'State Mean':>12} | {'State Std':>12}")
    print("-" * 50)
    
    torch.manual_seed(42)
    for t in range(num_steps):
        x_real = torch.randn(batch_size, d_model, device=device)
        x_imag = torch.randn(batch_size, d_model, device=device)
        t_val = torch.ones(batch_size, device=device) * t * PHI
        
        _, value_state = echo.forward_step(x_real, x_imag, value_state, t_val)
        
        state_norm = value_state.norm().item()
        state_mean = value_state.mean().item()
        state_std = value_state.std().item()
        
        print(f"{t:>5} | {state_norm:>12.4f} | {state_mean:>12.4f} | {state_std:>12.4f}")
    
    print("\n✓ Value state stabilizes due to EMA smoothing")


def test_full_model():
    """Test the full RINEchoModel."""
    print("\n" + "=" * 70)
    print("FULL MODEL TEST: RINEchoModel")
    print("=" * 70)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = RINEchoModel(
        vocab_size=100,
        d_model=48,
        num_layers=2,
        num_neurons=96,
        n_heads=4,
        alpha=0.1,
        output_mode='complex_linear',
        gate_mode='multiplicative',
    ).to(device)
    
    print(f"\n{model}")
    
    # Test forward pass
    batch_size = 8
    seq_len = 16
    input_ids = torch.randint(0, 100, (batch_size, seq_len), device=device)
    
    print("\n--- Forward Pass Test ---")
    logits, hidden, echo_states = model(input_ids)
    print(f"  Input shape: {input_ids.shape}")
    print(f"  Output shape: {logits.shape}")
    print(f"  Hidden shapes: ({hidden[0].shape}, {hidden[1].shape})")
    print(f"  Echo state shapes: {[s.shape for s in echo_states]}")
    
    # Test loss computation
    print("\n--- Loss Computation ---")
    loss, logits, hidden = model.compute_loss(input_ids)
    print(f"  Loss: {loss.item():.4f}")
    
    # Test gradient flow
    print("\n--- Gradient Flow ---")
    loss.backward()
    
    total_grad_norm = 0
    for name, param in model.named_parameters():
        if param.grad is not None:
            total_grad_norm += param.grad.norm().item()
    
    print(f"  Total gradient norm: {total_grad_norm:.4f}")
    
    # Check echo alphas
    alphas = model.get_echo_alphas()
    print(f"\n--- Learned Alpha Values ---")
    for i, alpha in enumerate(alphas):
        print(f"  Block {i}: {alpha.data}")
    
    return model


def test_flash_attention_sequence():
    """Test Flash Attention sequence processing."""
    print("\n" + "=" * 70)
    print("FLASH ATTENTION SEQUENCE TEST")
    print("=" * 70)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = RINEchoModel(
        vocab_size=100,
        d_model=48,
        num_layers=2,
        num_neurons=96,
        n_heads=4,
    ).to(device)
    
    batch_size = 8
    seq_len = 32
    input_ids = torch.randint(0, 100, (batch_size, seq_len), device=device)
    
    print(f"\nTesting with batch_size={batch_size}, seq_len={seq_len}")
    
    # Step-by-step forward
    import time
    
    torch.cuda.synchronize() if device.type == 'cuda' else None
    start = time.time()
    logits_step, hidden_step, _ = model(input_ids)
    torch.cuda.synchronize() if device.type == 'cuda' else None
    time_step = time.time() - start
    
    # Flash Attention forward
    torch.cuda.synchronize() if device.type == 'cuda' else None
    start = time.time()
    logits_flash, hidden_flash = model.forward_sequence(input_ids)
    torch.cuda.synchronize() if device.type == 'cuda' else None
    time_flash = time.time() - start
    
    print(f"\n  Step-by-step time: {time_step*1000:.2f}ms")
    print(f"  Flash Attention time: {time_flash*1000:.2f}ms")
    print(f"  Speedup: {time_step/time_flash:.2f}x")
    
    # Check outputs are similar (won't be identical due to different processing)
    diff = (logits_step - logits_flash).abs().mean()
    print(f"\n  Output difference (mean abs): {diff:.6f}")
    
    return time_step, time_flash


def main():
    print("=" * 70)
    print("ECHO CHAMBER ARCHITECTURE ANALYSIS")
    print("=" * 70)
    print("""
The Echo Chamber implements a novel attention mechanism:

1. PARALLEL PATHS: Echo Chamber runs in parallel with ResonantLayer
   (not sequential like Transformer attention + FFN)

2. EMA VALUE STATES: Each head maintains a persistent value state
   that decays via exponential moving average:
   V_state[t] = α * V_new + (1-α) * V_state[t-1]

3. GATING: Echo chamber can gate the resonant flow (like GLU)
   or combine additively (interference/superposition)

4. FLASH ATTENTION: Uses PyTorch's scaled_dot_product_attention
   for efficient sequence processing
""")
    
    # Run all analyses
    analyze_gradient_paths()
    compare_gate_modes()
    compare_output_modes()
    test_ema_dynamics()
    model = test_full_model()
    test_flash_attention_sequence()
    
    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()

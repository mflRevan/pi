#!/usr/bin/env python3
"""
Gradient Flow Analysis for Echo Chamber

Analyzes gradient flow through:
1. Parallel paths (ResonantLayer vs EchoChamber)
2. Different gate modes (additive vs glu)
3. Different output modes (complex_linear vs resonant)
4. EMA alpha effect on gradient flow
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
sys.path.insert(0, '/home/aiman/pi')

from rin import PHI, get_global_lut
from rin.echo_chamber import EchoChamber, ResonantBlock
from rin.model import ComplexLinear, ResonantLayer


def analyze_echo_chamber_gradients():
    """Analyze gradient flow through EchoChamber."""
    print("=" * 70)
    print("ECHO CHAMBER GRADIENT ANALYSIS")
    print("=" * 70)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    d_model = 64
    batch_size = 32
    n_heads = 4
    
    # Test different alpha values
    for alpha in [1.0, 0.5, 0.1]:
        print(f"\n--- Alpha = {alpha} ---")
        
        echo = EchoChamber(
            d_model=d_model,
            n_heads=n_heads,
            alpha=alpha,
            learnable_alpha=True,
            output_mode='complex_linear',
        ).to(device)
        
        # Create input
        x_real = torch.randn(batch_size, d_model, device=device, requires_grad=True)
        x_imag = torch.randn(batch_size, d_model, device=device, requires_grad=True)
        value_state = echo.init_state(batch_size, device)
        t = torch.ones(batch_size, device=device) * PHI
        
        # Forward
        (out_real, out_imag), new_state = echo.forward_step(x_real, x_imag, value_state, t)
        
        # Backward
        loss = out_real.sum() + out_imag.sum()
        loss.backward()
        
        print(f"  ∇x_real: mean={x_real.grad.abs().mean():.6f}, std={x_real.grad.std():.6f}")
        print(f"  ∇x_imag: mean={x_imag.grad.abs().mean():.6f}, std={x_imag.grad.std():.6f}")
        
        # Check gradient independence
        corr = F.cosine_similarity(
            x_real.grad.view(batch_size, -1),
            x_imag.grad.view(batch_size, -1), dim=1
        ).mean()
        print(f"  Correlation(∇x_real, ∇x_imag): {corr:+.4f}")
        
        # Check QKV gradients
        if echo.W_qkv.weight.grad is not None:
            print(f"  ∇W_qkv: norm={echo.W_qkv.weight.grad.norm():.4f}")
        
        # Check alpha gradient
        if hasattr(echo, 'log_alpha') and echo.log_alpha.grad is not None:
            print(f"  ∇log_alpha: {echo.log_alpha.grad.tolist()}")


def analyze_resonant_block_gradients():
    """Analyze gradient flow through ResonantBlock with different gate modes."""
    print("\n" + "=" * 70)
    print("RESONANT BLOCK GRADIENT ANALYSIS")
    print("=" * 70)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    d_model = 64
    num_neurons = 128
    batch_size = 32
    n_heads = 4
    
    for gate_mode in ['additive', 'glu']:
        print(f"\n--- Gate Mode: {gate_mode} ---")
        
        block = ResonantBlock(
            d_model=d_model,
            num_neurons=num_neurons,
            n_heads=n_heads,
            alpha=1.0,
            gate_mode=gate_mode,
            output_mode='complex_linear',
        ).to(device)
        
        # Create input
        x_real = torch.randn(batch_size, d_model, device=device, requires_grad=True)
        x_imag = torch.randn(batch_size, d_model, device=device, requires_grad=True)
        echo_state = block.echo_chamber.init_state(batch_size, device)
        t = torch.ones(batch_size, device=device) * PHI
        
        # Forward
        (out_real, out_imag), new_state = block.forward_step(x_real, x_imag, echo_state, t)
        
        # Backward
        loss = out_real.sum() + out_imag.sum()
        loss.backward()
        
        print(f"  ∇x_real: mean={x_real.grad.abs().mean():.6f}")
        print(f"  ∇x_imag: mean={x_imag.grad.abs().mean():.6f}")
        
        # Gradient independence
        corr = F.cosine_similarity(
            x_real.grad.view(batch_size, -1),
            x_imag.grad.view(batch_size, -1), dim=1
        ).mean()
        print(f"  Correlation(∇x_real, ∇x_imag): {corr:+.4f}")
        
        # Check component gradients
        res_grad_norm = sum(p.grad.norm().item() for p in block.resonant_layer.parameters() if p.grad is not None)
        echo_grad_norm = sum(p.grad.norm().item() for p in block.echo_chamber.parameters() if p.grad is not None)
        
        print(f"  ResonantLayer grad norm: {res_grad_norm:.4f}")
        print(f"  EchoChamber grad norm: {echo_grad_norm:.4f}")
        print(f"  Ratio (echo/res): {echo_grad_norm/res_grad_norm:.4f}" if res_grad_norm > 0 else "  Ratio: N/A")


def analyze_output_modes():
    """Compare complex_linear vs resonant output projection."""
    print("\n" + "=" * 70)
    print("OUTPUT MODE COMPARISON")
    print("=" * 70)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    d_model = 64
    batch_size = 32
    n_heads = 4
    
    for output_mode in ['complex_linear', 'resonant']:
        print(f"\n--- Output Mode: {output_mode} ---")
        
        echo = EchoChamber(
            d_model=d_model,
            n_heads=n_heads,
            alpha=1.0,
            output_mode=output_mode,
        ).to(device)
        
        x_real = torch.randn(batch_size, d_model, device=device, requires_grad=True)
        x_imag = torch.randn(batch_size, d_model, device=device, requires_grad=True)
        value_state = echo.init_state(batch_size, device)
        t = torch.ones(batch_size, device=device) * PHI
        
        (out_real, out_imag), _ = echo.forward_step(x_real, x_imag, value_state, t)
        
        loss = out_real.sum() + out_imag.sum()
        loss.backward()
        
        print(f"  Output stats:")
        print(f"    out_real: mean={out_real.mean():.4f}, std={out_real.std():.4f}")
        print(f"    out_imag: mean={out_imag.mean():.4f}, std={out_imag.std():.4f}")
        
        print(f"  Gradient stats:")
        print(f"    ∇x_real: mean={x_real.grad.abs().mean():.6f}")
        print(f"    ∇x_imag: mean={x_imag.grad.abs().mean():.6f}")
        
        # Check if output is on unit circle (resonant should be)
        if output_mode == 'resonant':
            magnitude = torch.sqrt(out_real ** 2 + out_imag ** 2)
            print(f"  Output magnitude (should be ~1): mean={magnitude.mean():.4f}, std={magnitude.std():.4f}")


def analyze_parallel_path_balance():
    """Analyze gradient balance between parallel paths."""
    print("\n" + "=" * 70)
    print("PARALLEL PATH GRADIENT BALANCE")
    print("=" * 70)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    d_model = 64
    num_neurons = 128
    batch_size = 32
    n_heads = 4
    
    block = ResonantBlock(
        d_model=d_model,
        num_neurons=num_neurons,
        n_heads=n_heads,
        alpha=1.0,
        gate_mode='additive',
    ).to(device)
    
    # Track activations
    activations = {}
    
    def save_activation(name):
        def hook(module, input, output):
            if isinstance(output, tuple):
                activations[name] = (output[0].detach(), output[1].detach() if len(output) > 1 else None)
            else:
                activations[name] = output.detach()
        return hook
    
    block.resonant_layer.register_forward_hook(save_activation('resonant'))
    
    x_real = torch.randn(batch_size, d_model, device=device, requires_grad=True)
    x_imag = torch.randn(batch_size, d_model, device=device, requires_grad=True)
    echo_state = block.echo_chamber.init_state(batch_size, device)
    t = torch.ones(batch_size, device=device) * PHI
    
    (out_real, out_imag), _ = block.forward_step(x_real, x_imag, echo_state, t)
    
    loss = out_real.sum() + out_imag.sum()
    loss.backward()
    
    print("\n  Activation magnitudes:")
    if 'resonant' in activations:
        res_real, res_imag = activations['resonant']
        print(f"    Resonant path: real={res_real.abs().mean():.4f}, imag={res_imag.abs().mean():.4f}")
    
    print(f"    Output: real={out_real.abs().mean():.4f}, imag={out_imag.abs().mean():.4f}")
    
    print("\n  Parameter gradient norms:")
    for name, param in block.named_parameters():
        if param.grad is not None:
            print(f"    {name}: {param.grad.norm():.4f}")


def main():
    analyze_echo_chamber_gradients()
    analyze_resonant_block_gradients()
    analyze_output_modes()
    analyze_parallel_path_balance()
    
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print("""
Key findings to check:
1. Gradient independence: ∇x_real and ∇x_imag should have low correlation
   (thanks to ComplexLinear before both paths)
   
2. Path balance: Both resonant and echo paths should receive gradients
   - If one dominates, the other path won't learn effectively
   
3. Alpha effect: Higher alpha (1.0) = instant updates, lower = memory decay
   - Should affect how gradients flow through value state updates
   
4. Output mode: Resonant output constrains to unit circle (magnitude ≈ 1)
   - Complex linear is more flexible but may have gradient scaling issues
""")


if __name__ == "__main__":
    main()

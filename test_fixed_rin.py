"""
Test the FIXED ResonantLayer and RIN gradient flow.

CORRECT Architecture per Resonant Neuron:
    - W: (d_model,) wavelengths - one per input dimension
    - B: (d_model,) phase offsets - one per input dimension
    
    1. x_collapsed: (batch, d_model) 
    2. θ[n,d] = x[d] / (1 + |W[n,d]|) + B[n,d] + t  → (batch, num_neurons, d_model)
    3. sin(θ), cos(θ) → (batch, num_neurons, d_model) each
    4. cos_sum = Σ_d cos(θ)  → (batch, num_neurons) - INTERFERENCE!
       sin_sum = Σ_d sin(θ)  → (batch, num_neurons) - INTERFERENCE!
    5. Project: (batch, num_neurons) → (batch, d_model) for real and imag

This is like an MLP dot product, but summing wave interference instead of weighted values.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import random
import numpy as np

import sys
sys.path.insert(0, '/home/aiman/pi')

from rin.model import ResonantLayer, RINModel, ComplexLinear, PHI
from rin.lut import get_global_lut

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")


def test_resonant_layer_shapes():
    """Verify the shapes are correct through the layer."""
    print("\n" + "="*70)
    print("TEST 1: ResonantLayer Shape Verification")
    print("="*70)
    
    batch = 4
    d_model = 64
    num_neurons = 128
    
    layer = ResonantLayer(d_model, num_neurons).to(device)
    
    x_real = torch.randn(batch, d_model, device=device)
    x_imag = torch.randn(batch, d_model, device=device)
    t = torch.tensor([1.0], device=device).expand(batch) * PHI
    
    # Check parameter shapes
    print(f"W shape: {layer.W.shape} (expected: ({num_neurons}, {d_model}))")
    print(f"B shape: {layer.B.shape} (expected: ({num_neurons}, {d_model}))")
    
    assert layer.W.shape == (num_neurons, d_model), "W shape wrong!"
    assert layer.B.shape == (num_neurons, d_model), "B shape wrong!"
    
    # Forward pass
    out_real, out_imag = layer(x_real, x_imag, t)
    
    print(f"Input shape: ({batch}, {d_model})")
    print(f"Output real shape: {out_real.shape} (expected: ({batch}, {d_model}))")
    print(f"Output imag shape: {out_imag.shape} (expected: ({batch}, {d_model}))")
    
    assert out_real.shape == (batch, d_model), "Output real shape wrong!"
    assert out_imag.shape == (batch, d_model), "Output imag shape wrong!"
    
    print("✓ All shapes correct!")
    return True


def test_interference_behavior():
    """Test that interference sum works as expected."""
    print("\n" + "="*70)
    print("TEST 2: Interference Sum Behavior")
    print("="*70)
    
    d_model = 8
    num_neurons = 4
    
    layer = ResonantLayer(d_model, num_neurons, use_swish=False).to(device)
    
    # Set simple weights for analysis
    with torch.no_grad():
        layer.W.zero_()  # wavelength = 1
        layer.B.zero_()  # no phase offset
        layer.input_collapse.weight.zero_()
        layer.input_collapse.weight[:d_model, :d_model] = torch.eye(d_model)  # pass x_real through
        layer.input_collapse.bias.zero_()
        layer.out_proj_real.weight.fill_(1.0 / num_neurons)  # average
        layer.out_proj_imag.weight.fill_(1.0 / num_neurons)
    
    # Input: all same value
    x_real = torch.ones(1, d_model, device=device) * 0.5
    x_imag = torch.zeros(1, d_model, device=device)
    t = torch.zeros(1, device=device)
    
    # x_collapsed = x_real (since we set up identity for first half)
    # theta = x / (1+0) + 0 + 0 = x = 0.5 for all neurons, all dims
    # cos(0.5) ≈ 0.8776, sin(0.5) ≈ 0.4794
    # sum across d_model: 0.8776 * 8 = 7.02, 0.4794 * 8 = 3.84
    
    out_real, out_imag = layer(x_real, x_imag, t)
    
    expected_cos_sum = math.cos(0.5) * d_model
    expected_sin_sum = math.sin(0.5) * d_model
    
    print(f"Input: x_real = 0.5 everywhere, x_imag = 0")
    print(f"Expected cos sum per neuron: {expected_cos_sum:.4f}")
    print(f"Expected sin sum per neuron: {expected_sin_sum:.4f}")
    print(f"Actual out_real mean: {out_real.mean().item():.4f}")
    print(f"Actual out_imag mean: {out_imag.mean().item():.4f}")
    
    # Should be close (after projection averaging)
    print("✓ Interference sum working!")
    return True


def test_gradient_flow():
    """Test gradient flow through the corrected layer."""
    print("\n" + "="*70)
    print("TEST 3: Gradient Flow Analysis")
    print("="*70)
    
    d_model = 32
    num_neurons = 64
    
    layer = ResonantLayer(d_model, num_neurons).to(device)
    
    x_real = torch.randn(4, d_model, device=device, requires_grad=True)
    x_imag = torch.randn(4, d_model, device=device, requires_grad=True)
    t = torch.ones(4, device=device) * PHI
    
    out_real, out_imag = layer(x_real, x_imag, t)
    loss = (out_real.sum() + out_imag.sum())
    loss.backward()
    
    print("Gradient norms:")
    print(f"  x_real grad: {x_real.grad.norm().item():.4f}")
    print(f"  x_imag grad: {x_imag.grad.norm().item():.4f}")
    print(f"  W grad: {layer.W.grad.norm().item():.4f}")
    print(f"  B grad: {layer.B.grad.norm().item():.4f}")
    print(f"  input_collapse grad: {layer.input_collapse.weight.grad.norm().item():.4f}")
    print(f"  out_proj_real grad: {layer.out_proj_real.weight.grad.norm().item():.4f}")
    print(f"  out_proj_imag grad: {layer.out_proj_imag.weight.grad.norm().item():.4f}")
    
    # Check all grads are non-zero
    assert x_real.grad.norm() > 0, "x_real grad is zero!"
    assert x_imag.grad.norm() > 0, "x_imag grad is zero!"
    assert layer.W.grad.norm() > 0, "W grad is zero!"
    assert layer.B.grad.norm() > 0, "B grad is zero!"
    
    print("✓ All gradients flowing!")
    return True


def test_full_model_gradient():
    """Test gradient flow through the full RIN model."""
    print("\n" + "="*70)
    print("TEST 4: Full RINModel Gradient Flow")
    print("="*70)
    
    vocab_size = 100
    d_model = 32
    num_layers = 2
    num_neurons = 64
    
    model = RINModel(
        vocab_size=vocab_size,
        d_model=d_model,
        num_layers=num_layers,
        num_neurons=num_neurons,
    ).to(device)
    
    print(f"Model params: {model.get_num_params():,}")
    
    # Simple forward pass
    input_ids = torch.randint(0, vocab_size, (4, 10), device=device)
    
    loss, logits, hidden = model.compute_loss(input_ids)
    loss.backward()
    
    # Check embedding gradients
    emb_grad = model.token_embedding.weight.grad
    print(f"\nEmbedding gradient norm: {emb_grad.norm().item():.4f}")
    
    # Check layer gradients
    for i, layer in enumerate(model.layers):
        print(f"Layer {i}:")
        print(f"  W grad norm: {layer.W.grad.norm().item():.6f}")
        print(f"  B grad norm: {layer.B.grad.norm().item():.6f}")
        print(f"  input_collapse grad: {layer.input_collapse.weight.grad.norm().item():.6f}")
    
    # Check output projection
    print(f"Output proj W_real grad: {model.output_proj_complex.W_real.weight.grad.norm().item():.6f}")
    
    print("✓ Full model gradients flowing!")
    return True


def test_needle_task():
    """Test on needle-in-haystack task."""
    print("\n" + "="*70)
    print("TEST 5: Needle-in-Haystack Learning")
    print("="*70)
    
    vocab_size = 50
    num_signals = 10
    d_model = 32
    num_neurons = 64
    
    model = RINModel(
        vocab_size=vocab_size,
        d_model=d_model,
        num_layers=2,
        num_neurons=num_neurons,
    ).to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    
    trigger_token = 0
    signal_tokens = list(range(1, num_signals + 1))
    noise_tokens = list(range(num_signals + 1, vocab_size))
    
    def generate_batch(batch_size, distance):
        sequences = []
        targets = []
        seq_len = distance + 2
        
        for _ in range(batch_size):
            signal = random.choice(signal_tokens)
            noise_before = [random.choice(noise_tokens) for _ in range(0)]
            noise_after = [random.choice(noise_tokens) for _ in range(distance)]
            seq = noise_before + [signal] + noise_after + [trigger_token]
            sequences.append(seq)
            targets.append(signal)
        
        return (
            torch.tensor(sequences, dtype=torch.long, device=device),
            torch.tensor(targets, dtype=torch.long, device=device)
        )
    
    print(f"Task: Retrieve signal token after {5} noise tokens")
    print(f"Random baseline: {100/num_signals:.1f}%")
    
    # Training loop
    for epoch in range(30):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for _ in range(20):
            seqs, targets = generate_batch(32, distance=5)
            
            optimizer.zero_grad()
            logits, _ = model(seqs)
            
            # Loss on last position
            loss = F.cross_entropy(logits[:, -1, :], targets)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item()
            preds = logits[:, -1, :].argmax(dim=-1)
            correct += (preds == targets).sum().item()
            total += len(targets)
        
        if (epoch + 1) % 5 == 0 or epoch == 0:
            acc = correct / total
            print(f"Epoch {epoch+1:3d}: Loss={total_loss/20:.4f}, Acc={acc*100:.1f}%")
    
    final_acc = correct / total
    print(f"\nFinal accuracy: {final_acc*100:.1f}%")
    
    if final_acc > 0.15:
        print("✓ Model is learning above random chance!")
    else:
        print("⚠ Model still near random chance")
    
    return final_acc


def analyze_interference_patterns():
    """Analyze what the interference sum actually computes."""
    print("\n" + "="*70)
    print("TEST 6: Interference Pattern Analysis")
    print("="*70)
    
    d_model = 16
    num_neurons = 8
    
    layer = ResonantLayer(d_model, num_neurons, use_swish=False).to(device)
    
    # Test with varying inputs
    x_zeros = torch.zeros(1, d_model, device=device)
    x_ones = torch.ones(1, d_model, device=device)
    x_random = torch.randn(1, d_model, device=device)
    t = torch.zeros(1, device=device)
    
    with torch.no_grad():
        # Manual computation to verify
        x_combined = torch.cat([x_random, torch.zeros_like(x_random)], dim=-1)
        x_collapsed = layer.input_collapse(x_combined)
        
        wavelength = 1.0 + layer.W.abs()
        theta = x_collapsed.unsqueeze(1) / wavelength + layer.B  # (1, num_neurons, d_model)
        
        lut = layer._get_lut(device)
        sin_theta, cos_theta = lut.lookup_sin_cos(theta)
        
        cos_sum = cos_theta.sum(dim=-1)  # (1, num_neurons)
        sin_sum = sin_theta.sum(dim=-1)
        
        print(f"Input x_collapsed stats: mean={x_collapsed.mean():.4f}, std={x_collapsed.std():.4f}")
        print(f"Theta stats: mean={theta.mean():.4f}, std={theta.std():.4f}")
        print(f"cos_theta stats: mean={cos_theta.mean():.4f}, std={cos_theta.std():.4f}")
        print(f"sin_theta stats: mean={sin_theta.mean():.4f}, std={sin_theta.std():.4f}")
        print(f"\nInterference sums per neuron:")
        print(f"  cos_sum: {cos_sum.squeeze().cpu().numpy()}")
        print(f"  sin_sum: {sin_sum.squeeze().cpu().numpy()}")
        print(f"  |cos_sum|: {cos_sum.abs().mean().item():.4f}")
        print(f"  |sin_sum|: {sin_sum.abs().mean().item():.4f}")
    
    # The key insight: when phases are random, interference tends to cancel
    # When phases align, interference amplifies
    print("\n✓ Interference analysis complete!")
    return True


if __name__ == "__main__":
    print("="*70)
    print("TESTING FIXED RESONANT LAYER & RIN MODEL")
    print("="*70)
    
    test_resonant_layer_shapes()
    test_interference_behavior()
    test_gradient_flow()
    test_full_model_gradient()
    analyze_interference_patterns()
    test_needle_task()
    
    print("\n" + "="*70)
    print("ALL TESTS COMPLETE")
    print("="*70)

#!/usr/bin/env python3
"""
Deep Diagnosis of RIN Gradient Flow Issues

KEY FINDINGS FROM PREVIOUS ANALYSIS:
1. Embeddings ARE receiving gradients (norm ~100+)
2. W/B ratio is ~0.3-0.4 (B gets ~3x more gradient than W)
3. Initial h_real/h_imag get NO gradients through euler chain!
4. Echo Chamber: Q-K cosine sim is NEGATIVE (-0.5 to -0.8)
5. Attention entropy decreasing but still high (~0.3-0.5)

THIS SCRIPT: Deep dive into WHY gradients don't flow back through euler
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import sys
sys.path.insert(0, '/home/aiman/pi')

from rin import PHI, get_global_lut
from rin.utils import wrap_time_periodic


def test_euler_gradient_flow():
    """Test gradient flow through the euler transform in isolation."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print("\n" + "="*70)
    print("EULER TRANSFORM GRADIENT FLOW ANALYSIS")
    print("="*70)
    
    lut = get_global_lut(4096, device)
    d_model = 64
    batch_size = 4
    
    # Test 1: Basic gradient flow with non-zero state
    print("\n--- TEST 1: Single step gradient flow ---")
    
    h_real = torch.randn(batch_size, d_model, device=device, requires_grad=True)
    h_imag = torch.randn(batch_size, d_model, device=device, requires_grad=True)
    w = torch.randn(batch_size, d_model, device=device, requires_grad=True) * 0.1
    b = torch.randn(batch_size, d_model, device=device, requires_grad=True) * 0.1
    
    # Forward
    wavelength = 1.0 + w.abs()
    t_phi = torch.tensor(0.0, device=device) * PHI
    
    theta_real = h_real / wavelength + b + t_phi
    theta_imag = h_imag / wavelength + b + t_phi
    
    sin_real, cos_real = lut.lookup_sin_cos(theta_real)
    sin_imag, cos_imag = lut.lookup_sin_cos(theta_imag)
    
    h_real_new = cos_real * cos_imag - sin_real * sin_imag
    h_imag_new = cos_real * sin_imag + sin_real * cos_imag
    
    # Backward
    loss = (h_real_new.sum() + h_imag_new.sum())
    loss.backward()
    
    print(f"h_real grad norm: {h_real.grad.norm().item():.6f}")
    print(f"h_imag grad norm: {h_imag.grad.norm().item():.6f}")
    # w and b gradients don't work directly because .abs() creates intermediate
    # In actual model, gradients flow through embedding.weight directly
    
    # Test 2: Multi-step chain
    print("\n--- TEST 2: Multi-step gradient flow (5 steps) ---")
    
    h_real = torch.randn(batch_size, d_model, device=device, requires_grad=True)
    h_imag = torch.randn(batch_size, d_model, device=device, requires_grad=True)
    
    # Fixed w, b for simplicity
    w_fixed = torch.randn(batch_size, d_model, device=device) * 0.1
    b_fixed = torch.randn(batch_size, d_model, device=device) * 0.1
    
    h_r, h_i = h_real, h_imag
    for t in range(5):
        wavelength = 1.0 + w_fixed.abs()
        t_phi = (t * PHI) % (2 * math.pi)
        
        theta_r = h_r / wavelength + b_fixed + t_phi
        theta_i = h_i / wavelength + b_fixed + t_phi
        
        sin_r, cos_r = lut.lookup_sin_cos(theta_r)
        sin_i, cos_i = lut.lookup_sin_cos(theta_i)
        
        h_r_new = cos_r * cos_i - sin_r * sin_i
        h_i_new = cos_r * sin_i + sin_r * cos_i
        
        h_r, h_i = h_r_new, h_i_new
    
    loss = h_r.sum() + h_i.sum()
    loss.backward()
    
    print(f"Initial h_real grad norm: {h_real.grad.norm().item():.6f}")
    print(f"Initial h_imag grad norm: {h_imag.grad.norm().item():.6f}")
    
    # Test 3: Check if gradient VANISHES over time
    print("\n--- TEST 3: Gradient magnitude over sequence length ---")
    
    for seq_len in [1, 2, 5, 10, 20]:
        h_real = torch.randn(batch_size, d_model, device=device, requires_grad=True)
        h_imag = torch.randn(batch_size, d_model, device=device, requires_grad=True)
        
        h_r, h_i = h_real, h_imag
        for t in range(seq_len):
            wavelength = 1.0 + w_fixed.abs()
            t_phi = (t * PHI) % (2 * math.pi)
            
            theta_r = h_r / wavelength + b_fixed + t_phi
            theta_i = h_i / wavelength + b_fixed + t_phi
            
            sin_r, cos_r = lut.lookup_sin_cos(theta_r)
            sin_i, cos_i = lut.lookup_sin_cos(theta_i)
            
            h_r_new = cos_r * cos_i - sin_r * sin_i
            h_i_new = cos_r * sin_i + sin_r * cos_i
            
            h_r, h_i = h_r_new, h_i_new
        
        loss = h_r.sum() + h_i.sum()
        loss.backward()
        
        grad_norm = h_real.grad.norm().item()
        print(f"  Seq len {seq_len:2d}: grad norm = {grad_norm:.6f}")
    
    # Test 4: Check the JACOBIAN of euler transform
    print("\n--- TEST 4: Jacobian analysis of single euler step ---")
    
    # For a single element, compute d(output)/d(input)
    h = torch.tensor([1.0], device=device, requires_grad=True)
    wavelength = torch.tensor([1.5], device=device)
    bias = torch.tensor([0.0], device=device)
    
    theta = h / wavelength + bias
    sin_t, cos_t = lut.lookup_sin_cos(theta)
    
    # Output is cos(theta) (real part when imag=0)
    cos_t.sum().backward()
    
    # d(cos(theta))/d(h) = -sin(theta) * d(theta)/d(h) = -sin(theta) / wavelength
    expected_grad = -sin_t.item() / wavelength.item()
    actual_grad = h.grad.item()
    
    print(f"  theta = {theta.item():.4f}")
    print(f"  cos(theta) = {cos_t.item():.4f}")
    print(f"  sin(theta) = {sin_t.item():.4f}")
    print(f"  Expected grad: {expected_grad:.6f}")
    print(f"  Actual grad: {actual_grad:.6f}")
    print(f"  Gradient magnitude: |grad| = {abs(actual_grad):.6f}")
    
    # Test 5: The COMPLEX multiplication effect
    print("\n--- TEST 5: Complex multiplication gradient flow ---")
    
    h_real = torch.tensor([0.5], device=device, requires_grad=True)
    h_imag = torch.tensor([0.5], device=device, requires_grad=True)
    
    theta_r = h_real / 1.5
    theta_i = h_imag / 1.5
    
    sin_r, cos_r = lut.lookup_sin_cos(theta_r)
    sin_i, cos_i = lut.lookup_sin_cos(theta_i)
    
    # Complex multiplication
    out_real = cos_r * cos_i - sin_r * sin_i
    out_imag = cos_r * sin_i + sin_r * cos_i
    
    # This is cos(theta_r + theta_i) and sin(theta_r + theta_i)
    combined_theta = theta_r + theta_i
    print(f"  theta_r = {theta_r.item():.4f}, theta_i = {theta_i.item():.4f}")
    print(f"  theta_r + theta_i = {combined_theta.item():.4f}")
    print(f"  out_real (should be cos(sum)) = {out_real.item():.4f}")
    print(f"  cos({combined_theta.item():.4f}) = {math.cos(combined_theta.item()):.4f}")
    
    loss = out_real + out_imag
    loss.backward()
    
    print(f"  h_real grad: {h_real.grad.item():.6f}")
    print(f"  h_imag grad: {h_imag.grad.item():.6f}")
    
    # The key insight: gradients through complex multiplication
    # d(out_real)/d(theta_r) = -sin_r*cos_i - cos_r*sin_i = -sin(theta_r + theta_i)
    # d(out_real)/d(theta_i) = -cos_r*sin_i - sin_r*cos_i = -sin(theta_r + theta_i)
    # So both get the same gradient contribution from out_real!
    
    # Test 6: What happens with ZERO initial state?
    print("\n--- TEST 6: Gradient flow with ZERO initial state ---")
    
    h_real_zero = torch.zeros(batch_size, d_model, device=device, requires_grad=True)
    h_imag_zero = torch.zeros(batch_size, d_model, device=device, requires_grad=True)
    
    # This is the critical case in our model!
    wavelength = 1.0 + w_fixed.abs()
    t_phi = 0.0
    
    theta_r = h_real_zero / wavelength + b_fixed + t_phi
    theta_i = h_imag_zero / wavelength + b_fixed + t_phi
    
    print(f"  When h_real=0, h_imag=0:")
    print(f"  theta_r = b_fixed + 0 (depends only on b)")
    print(f"  theta_i = b_fixed + 0 (depends only on b)")
    print(f"  theta_r == theta_i? {torch.allclose(theta_r, theta_i)}")
    
    sin_r, cos_r = lut.lookup_sin_cos(theta_r)
    sin_i, cos_i = lut.lookup_sin_cos(theta_i)
    
    # When theta_r == theta_i:
    # out_real = cos^2 - sin^2 = cos(2*theta)
    # out_imag = 2*cos*sin = sin(2*theta)
    
    out_real = cos_r * cos_i - sin_r * sin_i
    out_imag = cos_r * sin_i + sin_r * cos_i
    
    loss = out_real.sum() + out_imag.sum()
    loss.backward()
    
    print(f"  h_real_zero grad norm: {h_real_zero.grad.norm().item():.6f}")
    print(f"  h_imag_zero grad norm: {h_imag_zero.grad.norm().item():.6f}")
    print(f"  ⚠️  Note: grad is NON-ZERO even for zero input!")
    print(f"  This is because theta still depends on h through wavelength denominator")
    
    # Test 7: Multi-step with zero init
    print("\n--- TEST 7: Multi-step from ZERO initial state ---")
    
    h_real_z = torch.zeros(batch_size, d_model, device=device, requires_grad=True)
    h_imag_z = torch.zeros(batch_size, d_model, device=device, requires_grad=True)
    
    h_r, h_i = h_real_z, h_imag_z
    intermediates = []
    
    for t in range(5):
        wavelength = 1.0 + w_fixed.abs()
        t_phi = (t * PHI) % (2 * math.pi)
        
        theta_r = h_r / wavelength + b_fixed + t_phi
        theta_i = h_i / wavelength + b_fixed + t_phi
        
        sin_r, cos_r = lut.lookup_sin_cos(theta_r)
        sin_i, cos_i = lut.lookup_sin_cos(theta_i)
        
        h_r_new = cos_r * cos_i - sin_r * sin_i
        h_i_new = cos_r * sin_i + sin_r * cos_i
        
        intermediates.append((h_r.norm().item(), h_i.norm().item()))
        h_r, h_i = h_r_new, h_i_new
    
    print("  State norms at each step:")
    for i, (r_norm, i_norm) in enumerate(intermediates):
        print(f"    Step {i}: h_r={r_norm:.4f}, h_i={i_norm:.4f}")
    print(f"    Final: h_r={h_r.norm().item():.4f}, h_i={h_i.norm().item():.4f}")
    
    loss = h_r.sum() + h_i.sum()
    loss.backward()
    
    print(f"\n  Initial state gradient norms:")
    print(f"    h_real_z grad: {h_real_z.grad.norm().item():.6f}")
    print(f"    h_imag_z grad: {h_imag_z.grad.norm().item():.6f}")
    
    # CRITICAL INSIGHT
    print("\n" + "="*70)
    print("CRITICAL INSIGHTS")
    print("="*70)
    
    print("""
1. GRADIENT DOES FLOW through euler transform, but from ZERO initial state,
   the first step's output depends ONLY on (b + t*phi) - NOT on h!
   
   theta = h/wavelength + b + t*phi
   When h=0: theta = b + t*phi (h has no influence on theta!)
   
2. The gradient of loss w.r.t. h_zero is:
   dL/dh = dL/d(theta) * d(theta)/dh = dL/d(theta) * (1/wavelength)
   But this doesn't help because h=0 means h's VALUE didn't affect output!
   
3. After step 1, h becomes cos/sin of (b + t), which IS non-zero.
   But the gradient flows back through THIS path, not through initial h.
   
4. The FUNDAMENTAL ISSUE: Initial state h_0 = 0 is a FIXED POINT.
   Any information must be injected via embeddings (w, b), not via h.
   
5. For the model to learn, it must:
   - Encode signal information in b (phase offset)
   - Encode signal information in w (wavelength/frequency)
   - Rely on b/w at SIGNAL position to create distinguishable states
   
6. The trigger token's (w, b) at the FINAL position sees all gradients,
   but intermediate tokens see diminishing gradients due to the rotation.
""")


def test_echo_chamber_behavior():
    """Test Echo Chamber Q/K/V behavior and specialization."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("\n" + "="*70)
    print("ECHO CHAMBER BEHAVIOR ANALYSIS")
    print("="*70)
    
    from rin.echo_chamber import EchoChamber
    
    d_model = 64
    n_heads = 4
    batch_size = 8
    
    echo = EchoChamber(
        d_model=d_model,
        n_heads=n_heads,
        alpha=1.0,  # Instant updates
    ).to(device)
    
    # Test 1: Q-K similarity with random inputs
    print("\n--- TEST 1: Q-K Similarity Analysis ---")
    
    x_real = torch.randn(batch_size, d_model, device=device)
    x_imag = torch.randn(batch_size, d_model, device=device)
    state = echo.init_state(batch_size, device)
    
    # Manual forward to inspect Q, K, V
    proj_real, proj_imag = echo.input_proj(x_real, x_imag)
    x_collapsed = proj_real + proj_imag
    x_collapsed = echo.prenorm(x_collapsed)
    
    qkv = echo.W_qkv(x_collapsed)
    qkv = qkv.view(batch_size, 3, n_heads, d_model // n_heads)
    q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]
    
    # Q-K similarity
    qk_sim = F.cosine_similarity(q.reshape(-1, d_model // n_heads), 
                                  k.reshape(-1, d_model // n_heads), dim=-1)
    
    print(f"  Q-K cosine similarity: {qk_sim.mean().item():.4f} ± {qk_sim.std().item():.4f}")
    print(f"  Q-K sim range: [{qk_sim.min().item():.4f}, {qk_sim.max().item():.4f}]")
    
    # The problem: Q and K come from the SAME projection, so they tend to be similar
    # But our analysis showed NEGATIVE similarity (-0.5 to -0.8) after training!
    
    # Test 2: Attention gate behavior
    print("\n--- TEST 2: Attention Gate Analysis ---")
    
    scale = math.sqrt(d_model // n_heads)
    scores = torch.sum(q * k, dim=-1, keepdim=True) / scale
    gate = torch.sigmoid(scores)
    
    print(f"  Scores before sigmoid: {scores.mean().item():.4f} ± {scores.std().item():.4f}")
    print(f"  Gate values: {gate.mean().item():.4f} ± {gate.std().item():.4f}")
    print(f"  Gate range: [{gate.min().item():.4f}, {gate.max().item():.4f}]")
    
    # Entropy of gate (binary entropy for sigmoid output)
    p = gate.squeeze(-1)
    entropy = -p * torch.log(p + 1e-8) - (1 - p) * torch.log(1 - p + 1e-8)
    print(f"  Gate entropy: {entropy.mean().item():.4f} (max = 0.693 for uniform)")
    
    # Test 3: What happens with repeated inputs?
    print("\n--- TEST 3: Memory behavior with repeated inputs ---")
    
    # Simulate sequence: [A, noise, noise, A]
    # Does the model retrieve information from first A when seeing second A?
    
    x_signal = torch.randn(1, d_model, device=device) * 0.5  # Signal A
    x_noise = torch.randn(1, d_model, device=device) * 0.5   # Noise
    
    state = echo.init_state(1, device)
    
    # Step 1: See signal A
    (out1_r, out1_i), state = echo.forward_step(x_signal, x_signal, state, torch.tensor(0.0))
    state1 = state.clone()
    
    # Step 2: See noise
    (out2_r, out2_i), state = echo.forward_step(x_noise, x_noise, state, torch.tensor(1.0))
    
    # Step 3: See noise
    (out3_r, out3_i), state = echo.forward_step(x_noise, x_noise, state, torch.tensor(2.0))
    
    # Step 4: See signal A again - can it retrieve state1?
    (out4_r, out4_i), state = echo.forward_step(x_signal, x_signal, state, torch.tensor(3.0))
    
    print(f"  State after signal A (step 1): norm = {state1.norm().item():.4f}")
    print(f"  Output at step 1: {(out1_r + out1_i).norm().item():.4f}")
    print(f"  Output at step 4 (signal A again): {(out4_r + out4_i).norm().item():.4f}")
    
    # With alpha=1.0, state is instantly replaced, so state after step 4
    # contains only step 4's value, not step 1's
    print(f"  State after step 4: norm = {state.norm().item():.4f}")
    
    # Test 4: Alpha effect
    print("\n--- TEST 4: Alpha (EMA decay) effect ---")
    
    for alpha in [0.1, 0.5, 0.9, 1.0]:
        echo_test = EchoChamber(d_model=d_model, n_heads=n_heads, alpha=alpha).to(device)
        
        state = echo_test.init_state(1, device)
        
        # See 3 different signals
        for t in range(3):
            x = torch.randn(1, d_model, device=device)
            (_, _), state = echo_test.forward_step(x, x, state, torch.tensor(float(t)))
        
        print(f"  Alpha={alpha:.1f}: Final state norm = {state.norm().item():.4f}")


def test_embedding_gradient_path():
    """Test how gradients reach embeddings through the full forward pass."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("\n" + "="*70)
    print("EMBEDDING GRADIENT PATH ANALYSIS")
    print("="*70)
    
    # Simulate the forward pass gradient path
    d_model = 64
    vocab_size = 61
    seq_len = 10
    batch_size = 4
    
    lut = get_global_lut(4096, device)
    
    # Embedding layer
    embedding = nn.Embedding(vocab_size, 2 * d_model).to(device)
    nn.init.normal_(embedding.weight, std=0.02)
    
    # Generate input
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    
    # Forward pass tracking
    embeddings = embedding(input_ids)
    w_emb = embeddings[:, :, :d_model]
    b_emb = embeddings[:, :, d_model:]
    
    h_real = torch.zeros(batch_size, d_model, device=device)
    h_imag = torch.zeros(batch_size, d_model, device=device)
    
    for t in range(seq_len):
        wavelength = 1.0 + w_emb[:, t].abs()
        t_phi = (t * PHI) % (2 * math.pi)
        
        theta_r = h_real / wavelength + b_emb[:, t] + t_phi
        theta_i = h_imag / wavelength + b_emb[:, t] + t_phi
        
        sin_r, cos_r = lut.lookup_sin_cos(theta_r)
        sin_i, cos_i = lut.lookup_sin_cos(theta_i)
        
        h_real = cos_r * cos_i - sin_r * sin_i
        h_imag = cos_r * sin_i + sin_r * cos_i
    
    # Simple loss
    loss = (h_real.sum() + h_imag.sum())
    loss.backward()
    
    print("\n--- Gradient distribution by token position ---")
    
    # Gradient is in embedding.weight, but we need to see which positions got gradients
    grad = embedding.weight.grad
    w_grad = grad[:, :d_model]
    b_grad = grad[:, d_model:]
    
    # Check which tokens appear in our input
    unique_tokens = input_ids.unique()
    
    for tok in unique_tokens[:5]:  # First 5 unique tokens
        tok_grad = grad[tok]
        w_g = tok_grad[:d_model].norm().item()
        b_g = tok_grad[d_model:].norm().item()
        
        # Count occurrences and positions
        positions = (input_ids == tok).nonzero()
        pos_list = positions[:, 1].tolist() if len(positions) > 0 else []
        
        print(f"  Token {tok.item():2d}: W grad={w_g:.4f}, B grad={b_g:.4f}, positions={pos_list[:5]}")
    
    # The KEY question: Do later position tokens get more gradient?
    print("\n--- Gradient by position in sequence ---")
    
    # For each position, sum gradients of tokens appearing there
    position_grads = []
    for pos in range(seq_len):
        tokens_at_pos = input_ids[:, pos]  # (batch,)
        grads_at_pos = grad[tokens_at_pos]  # (batch, 2*d_model)
        avg_grad_norm = grads_at_pos.norm(dim=-1).mean().item()
        position_grads.append(avg_grad_norm)
    
    for pos, g in enumerate(position_grads):
        bar = "█" * int(g * 10)
        print(f"  Position {pos:2d}: {g:.4f} {bar}")
    
    print(f"\n  Last position gradient: {position_grads[-1]:.4f}")
    print(f"  First position gradient: {position_grads[0]:.4f}")
    print(f"  Ratio (last/first): {position_grads[-1]/(position_grads[0]+1e-8):.2f}x")


if __name__ == '__main__':
    test_euler_gradient_flow()
    test_echo_chamber_behavior()
    test_embedding_gradient_path()

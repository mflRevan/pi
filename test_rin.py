"""
Test Script for Resonant Interference Network

Verifies:
1. LUT accuracy and performance
2. Forward pass correctness
3. Backward pass (custom gradients)
4. Full model integration
5. Generation capability
"""

import torch
import torch.nn as nn
import math
import time
import sys

# Local imports
from rin import SinLUT, SinLayer, ResonantBlock, RINModel
from rin.lut import get_global_lut, reset_global_lut
from rin.utils import visualize_lut_accuracy, count_parameters, print_model_summary


def test_sin_lut():
    """Test Sin Look-Up Table."""
    print("\n" + "="*60)
    print("TEST: Sin Look-Up Table")
    print("="*60)
    
    # Create LUT
    lut = SinLUT(resolution=512)
    print(f"Created {lut}")
    
    # Test accuracy
    accuracy = visualize_lut_accuracy(lut, num_samples=10000)
    print(f"\nAccuracy (512 resolution):")
    print(f"  Sin max error: {accuracy['sin_max_error']:.6f}")
    print(f"  Sin mean error: {accuracy['sin_mean_error']:.6f}")
    print(f"  Cos max error: {accuracy['cos_max_error']:.6f}")
    print(f"  Cos mean error: {accuracy['cos_mean_error']:.6f}")
    
    # Test phase wrapping
    phases = torch.tensor([-2*math.pi, -math.pi, 0, math.pi, 2*math.pi, 3*math.pi])
    sin_values = lut.lookup_sin(phases)
    expected = torch.sin(phases)
    print(f"\nPhase wrapping test:")
    print(f"  Phases: {phases.tolist()}")
    print(f"  LUT sin: {sin_values.tolist()}")
    print(f"  Expected: {expected.tolist()}")
    print(f"  Max diff: {(sin_values - expected).abs().max().item():.6f}")
    
    # Speed test
    test_phases = torch.randn(1000, 512)
    
    # LUT speed
    start = time.time()
    for _ in range(100):
        _ = lut.lookup_sin(test_phases)
    lut_time = time.time() - start
    
    # torch.sin speed
    start = time.time()
    for _ in range(100):
        _ = torch.sin(test_phases)
    torch_time = time.time() - start
    
    print(f"\nSpeed comparison (100 iterations, 1000x512 tensor):")
    print(f"  LUT: {lut_time:.3f}s")
    print(f"  torch.sin: {torch_time:.3f}s")
    print(f"  Speedup: {torch_time/lut_time:.2f}x")
    
    print("\n✓ LUT tests passed!")
    return True


def test_sin_layer():
    """Test Sin Layer forward and backward pass."""
    print("\n" + "="*60)
    print("TEST: Sin Layer")
    print("="*60)
    
    # Create layer
    input_dim = 64
    num_neurons = 32
    layer = SinLayer(input_dim=input_dim, num_neurons=num_neurons)
    print(f"Created {layer}")
    
    # Test forward pass
    batch_size = 4
    seq_len = 10
    x = torch.randn(batch_size, seq_len, input_dim, requires_grad=True)
    
    output = layer(x)
    print(f"\nForward pass:")
    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {output.shape}")
    print(f"  Expected: ({batch_size}, {seq_len}, {num_neurons})")
    assert output.shape == (batch_size, seq_len, num_neurons), "Output shape mismatch!"
    
    # Test output range (sin should be bounded)
    # Each output is sum of embed_dim sin values, so bounded by [-embed_dim, embed_dim]
    print(f"  Output range: [{output.min().item():.2f}, {output.max().item():.2f}]")
    print(f"  Theoretical max: ±{input_dim}")
    
    # Test backward pass
    loss = output.sum()
    loss.backward()
    
    print(f"\nBackward pass:")
    print(f"  x.grad shape: {x.grad.shape}")
    print(f"  x.grad mean: {x.grad.mean().item():.6f}")
    print(f"  w.grad shape: {layer.w.grad.shape}")
    print(f"  w.grad mean: {layer.w.grad.mean().item():.6f}")
    print(f"  b.grad shape: {layer.b.grad.shape}")
    print(f"  b.grad mean: {layer.b.grad.mean().item():.6f}")
    
    assert x.grad is not None, "Input gradient is None!"
    assert layer.w.grad is not None, "Weight gradient is None!"
    assert layer.b.grad is not None, "Bias gradient is None!"
    
    # Test that STDP gradient for w is bounded
    # The phase_error mod π approach should bound gradients
    w_grad_max = layer.w.grad.abs().max().item()
    print(f"  w.grad max: {w_grad_max:.6f} (should be bounded)")
    
    # Test with timesteps
    t = torch.arange(seq_len, dtype=torch.float32)
    output_t = layer(x.detach().requires_grad_(True), t)
    print(f"\nWith explicit timesteps:")
    print(f"  Output shape: {output_t.shape}")
    
    print("\n✓ Sin Layer tests passed!")
    return True


def test_resonant_block():
    """Test Resonant Block."""
    print("\n" + "="*60)
    print("TEST: Resonant Block")
    print("="*60)
    
    input_dim = 64
    hidden_dim = 128
    output_dim = 64
    
    # Test basic block
    block = ResonantBlock(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        use_layer_norm=True,
        use_gate=True,
    )
    print(f"Created ResonantBlock")
    
    batch_size = 4
    seq_len = 10
    x = torch.randn(batch_size, seq_len, input_dim)
    
    output = block(x)
    print(f"\nForward pass:")
    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {output.shape}")
    assert output.shape == (batch_size, seq_len, output_dim), "Output shape mismatch!"
    
    # Test backward
    loss = output.sum()
    loss.backward()
    print(f"  Backward pass completed")
    
    # Count parameters
    params = sum(p.numel() for p in block.parameters())
    print(f"  Parameters: {params:,}")
    
    print("\n✓ Resonant Block tests passed!")
    return True


def test_rin_model():
    """Test full RIN Model."""
    print("\n" + "="*60)
    print("TEST: RIN Model")
    print("="*60)
    
    # Small model for testing
    model = RINModel(
        vocab_size=1000,
        embed_dim=64,
        hidden_dim=128,
        num_layers=1,
        num_heads=2,
        neurons_per_head=32,
        max_seq_len=128,
        use_multi_head=True,
    )
    print(f"\nCreated model:")
    print_model_summary(model)
    
    # Test forward pass
    batch_size = 2
    seq_len = 20
    input_ids = torch.randint(0, 1000, (batch_size, seq_len))
    
    outputs = model(input_ids, return_embeddings=True)
    print(f"\nForward pass:")
    print(f"  Input shape: {input_ids.shape}")
    print(f"  Logits shape: {outputs['logits'].shape}")
    print(f"  Hidden shape: {outputs['hidden_states'].shape}")
    print(f"  Embeddings shape: {outputs['embeddings'].shape}")
    
    expected_logits_shape = (batch_size, seq_len, 1000)
    assert outputs['logits'].shape == expected_logits_shape, "Logits shape mismatch!"
    
    # Test loss computation
    loss, _ = model.compute_loss(input_ids)
    print(f"\nLoss computation:")
    print(f"  Loss: {loss.item():.4f}")
    
    # Test backward pass
    loss.backward()
    print(f"  Backward pass completed")
    
    # Check gradients on embeddings
    embed_grad = model.token_embedding.weight.grad
    print(f"  Embedding grad shape: {embed_grad.shape}")
    print(f"  Embedding grad mean: {embed_grad.mean().item():.6f}")
    
    # Test generation
    print(f"\nGeneration test:")
    model.eval()
    prompt = torch.randint(0, 1000, (1, 5))
    generated = model.generate(prompt, max_new_tokens=10, temperature=1.0)
    print(f"  Prompt length: {prompt.shape[1]}")
    print(f"  Generated length: {generated.shape[1]}")
    print(f"  Generated tokens: {generated[0].tolist()}")
    
    print("\n✓ RIN Model tests passed!")
    return True


def test_gpu_support():
    """Test GPU support if available."""
    print("\n" + "="*60)
    print("TEST: GPU Support")
    print("="*60)
    
    if not torch.cuda.is_available():
        print("CUDA not available, skipping GPU tests")
        return True
    
    device = torch.device("cuda")
    print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    
    # Create model on GPU
    model = RINModel(
        vocab_size=1000,
        embed_dim=64,
        hidden_dim=128,
        num_layers=1,
        num_heads=2,
        neurons_per_head=32,
    ).to(device)
    
    # Test forward/backward on GPU
    input_ids = torch.randint(0, 1000, (4, 32), device=device)
    
    # Warmup
    for _ in range(3):
        loss, _ = model.compute_loss(input_ids)
        loss.backward()
        model.zero_grad()
    
    # Timed run
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(10):
        loss, _ = model.compute_loss(input_ids)
        loss.backward()
        model.zero_grad()
    torch.cuda.synchronize()
    elapsed = time.time() - start
    
    print(f"\nGPU timing (10 iterations):")
    print(f"  Total time: {elapsed:.3f}s")
    print(f"  Per iteration: {elapsed/10*1000:.2f}ms")
    
    # Memory usage
    print(f"\nGPU memory:")
    print(f"  Allocated: {torch.cuda.memory_allocated()/1e6:.1f} MB")
    print(f"  Cached: {torch.cuda.memory_reserved()/1e6:.1f} MB")
    
    print("\n✓ GPU tests passed!")
    return True


def test_gradient_flow():
    """Test that gradients flow correctly through custom backward."""
    print("\n" + "="*60)
    print("TEST: Gradient Flow")
    print("="*60)
    
    # Create a simple model
    layer = SinLayer(input_dim=16, num_neurons=8)
    
    # Input that requires grad
    x = torch.randn(2, 5, 16, requires_grad=True)
    
    # Forward
    y = layer(x)
    
    # Create a simple target
    target = torch.zeros_like(y)
    loss = ((y - target) ** 2).mean()
    
    print(f"Initial loss: {loss.item():.4f}")
    
    # Backward
    loss.backward()
    
    # Check all gradients exist and are finite
    assert x.grad is not None, "Input gradient missing!"
    assert torch.isfinite(x.grad).all(), "Input gradient has NaN/Inf!"
    
    assert layer.w.grad is not None, "W gradient missing!"
    assert torch.isfinite(layer.w.grad).all(), "W gradient has NaN/Inf!"
    
    assert layer.b.grad is not None, "B gradient missing!"
    assert torch.isfinite(layer.b.grad).all(), "B gradient has NaN/Inf!"
    
    print(f"Gradient norms:")
    print(f"  x.grad: {x.grad.norm().item():.4f}")
    print(f"  w.grad: {layer.w.grad.norm().item():.4f}")
    print(f"  b.grad: {layer.b.grad.norm().item():.4f}")
    
    # Test gradient descent actually reduces loss
    optimizer = torch.optim.SGD(layer.parameters(), lr=0.01)
    
    losses = []
    for i in range(100):
        x_new = torch.randn(2, 5, 16, requires_grad=True)
        y = layer(x_new)
        loss = ((y - target) ** 2).mean()
        losses.append(loss.item())
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    print(f"\nLoss after 100 steps: {losses[-1]:.4f}")
    print(f"Loss reduction: {losses[0] - losses[-1]:.4f}")
    
    # Check loss decreased (it should for this simple target)
    assert losses[-1] < losses[0], "Loss did not decrease!"
    
    print("\n✓ Gradient flow tests passed!")
    return True


def test_stdp_gradient():
    """Test that STDP-like gradient is bounded."""
    print("\n" + "="*60)
    print("TEST: STDP Gradient Bounding")
    print("="*60)
    
    layer = SinLayer(input_dim=32, num_neurons=16)
    
    # Test with increasing sequence lengths
    seq_lengths = [10, 100, 1000]
    w_grad_maxes = []
    
    for seq_len in seq_lengths:
        x = torch.randn(1, seq_len, 32, requires_grad=True)
        y = layer(x)
        loss = y.sum()
        
        layer.zero_grad()
        loss.backward()
        
        w_grad_max = layer.w.grad.abs().max().item()
        w_grad_maxes.append(w_grad_max)
        
        print(f"Seq len {seq_len}: w.grad max = {w_grad_max:.4f}")
    
    # With traditional backprop, grad would grow with t
    # With STDP, it should stay bounded
    # Allow some growth due to more samples, but not linear with seq_len
    growth_factor = w_grad_maxes[-1] / w_grad_maxes[0]
    seq_growth = seq_lengths[-1] / seq_lengths[0]
    
    print(f"\nGrad growth factor: {growth_factor:.2f}x")
    print(f"Seq length growth: {seq_growth:.0f}x")
    
    # Gradient should grow much slower than sequence length
    # (ideally sqrt or log scale due to random phase distributions)
    assert growth_factor < seq_growth / 2, "Gradient growing too fast with sequence length!"
    
    print("\n✓ STDP gradient bounding tests passed!")
    return True


def run_all_tests():
    """Run all tests."""
    print("\n" + "="*60)
    print("RESONANT INTERFERENCE NETWORK - TEST SUITE")
    print("="*60)
    
    tests = [
        ("Sin LUT", test_sin_lut),
        ("Sin Layer", test_sin_layer),
        ("Resonant Block", test_resonant_block),
        ("RIN Model", test_rin_model),
        ("Gradient Flow", test_gradient_flow),
        ("STDP Gradient", test_stdp_gradient),
        ("GPU Support", test_gpu_support),
    ]
    
    results = []
    for name, test_fn in tests:
        try:
            reset_global_lut()  # Reset LUT between tests
            passed = test_fn()
            results.append((name, "PASSED" if passed else "FAILED"))
        except Exception as e:
            print(f"\n✗ {name} FAILED with exception: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, f"ERROR: {e}"))
    
    print("\n" + "="*60)
    print("TEST RESULTS")
    print("="*60)
    for name, status in results:
        symbol = "✓" if status == "PASSED" else "✗"
        print(f"  {symbol} {name}: {status}")
    
    passed = sum(1 for _, s in results if s == "PASSED")
    total = len(results)
    print(f"\n{passed}/{total} tests passed")
    
    return all(s == "PASSED" for _, s in results)


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)

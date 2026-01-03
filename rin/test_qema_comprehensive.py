"""
Comprehensive test for Echo Chamber V2 with Q-EMA.

Tests:
1. Gradient stability - no NaN/Inf, reasonable magnitudes
2. EMA behavior - proper decay dynamics with w_eff
3. Interference score distribution - both constructive and destructive
4. Memory development - bounded growth, information retention
5. Output projection learning dynamics - w_out and log_beta evolution
6. Beta parameterization - stays positive, reasonable range
"""

import torch
import torch.nn as nn
import sys
sys.path.insert(0, '/home/aiman/pi')

from rin.echo_chamber_v2 import EchoChamberV2, EchoChamberModelV2

torch.manual_seed(42)


def test_beta_parameterization():
    """Test that beta parameterization is stable and reasonable."""
    print("=" * 80)
    print("TEST 1: Beta Parameterization")
    print("=" * 80)
    
    d_model = 64
    chamber = EchoChamberV2(d_model=d_model, n_heads=4)
    
    # Initial beta values
    beta_init = chamber.get_beta()
    w_eff_init = chamber.get_effective_wavelength()
    decay_init = chamber.compute_decay()
    
    print(f"\nInitial state:")
    print(f"  log_beta:  mean={chamber.log_beta.mean():.4f}, std={chamber.log_beta.std():.4f}")
    print(f"  beta:      mean={beta_init.mean():.4f}, min={beta_init.min():.4f}, max={beta_init.max():.4f}")
    print(f"  w_eff:     mean={w_eff_init.mean():.4f}, min={w_eff_init.min():.4f}, max={w_eff_init.max():.4f}")
    print(f"  decay:     mean={decay_init.mean():.4f}, min={decay_init.min():.4f}, max={decay_init.max():.4f}")
    
    # Verify beta is always positive
    assert (beta_init > 0).all(), "Beta should always be positive"
    assert (beta_init > 1e-4).all(), "Beta should be at least eps"
    
    # Verify initial beta is near golden ratio (~0.076)
    golden_target = 0.076
    assert (beta_init.mean() - golden_target).abs() < 0.01, f"Initial beta should be near {golden_target}"
    
    # Verify decay is in valid range
    assert (decay_init >= 0).all() and (decay_init <= 0.9999).all(), "Decay should be in [0, 0.9999]"
    
    print(f"\n✓ Beta parameterization is stable and properly initialized")
    return True


def test_gradient_stability():
    """Test gradient flow through the model."""
    print("\n" + "=" * 80)
    print("TEST 2: Gradient Stability")
    print("=" * 80)
    
    model = EchoChamberModelV2(
        vocab_size=100,
        d_model=64,
        num_layers=2,
        num_neurons=64,
        n_echo_heads=4,
    )
    
    # Create batch
    batch_size = 4
    seq_len = 16
    x = torch.randint(0, 100, (batch_size, seq_len))
    target = torch.randint(0, 100, (batch_size, seq_len))
    
    # Forward pass
    model.train()
    logits = model(x)  # Model returns just logits by default
    loss = nn.CrossEntropyLoss()(logits.view(-1, 100), target.view(-1))
    
    print(f"\nForward pass:")
    print(f"  Loss: {loss.item():.4f}")
    print(f"  Logits: mean={logits.mean():.4f}, std={logits.std():.4f}")
    
    # Backward pass
    loss.backward()
    
    # Check gradients
    print(f"\nGradient analysis:")
    grad_stats = {}
    
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad = param.grad
            grad_stats[name] = {
                'mean': grad.mean().item(),
                'std': grad.std().item(),
                'max': grad.abs().max().item(),
                'has_nan': torch.isnan(grad).any().item(),
                'has_inf': torch.isinf(grad).any().item(),
            }
    
    # Print key gradients
    key_params = ['log_beta', 'w_out', 'trigger_real', 'trigger_imag']
    for key in key_params:
        for name, stats in grad_stats.items():
            if key in name:
                print(f"  {name}:")
                print(f"    mean={stats['mean']:.6f}, std={stats['std']:.6f}, max={stats['max']:.6f}")
                assert not stats['has_nan'], f"NaN in gradient for {name}"
                assert not stats['has_inf'], f"Inf in gradient for {name}"
                break
    
    # Verify no NaN/Inf anywhere
    all_clean = all(not s['has_nan'] and not s['has_inf'] for s in grad_stats.values())
    assert all_clean, "Some gradients have NaN or Inf"
    
    print(f"\n✓ All gradients are clean (no NaN/Inf)")
    return True


def test_ema_behavior():
    """Test EMA decay dynamics with Q-factor."""
    print("\n" + "=" * 80)
    print("TEST 3: EMA Behavior and Q-Factor")
    print("=" * 80)
    
    d_model = 64
    chamber = EchoChamberV2(d_model=d_model, n_heads=4)
    chamber.eval()
    
    # Create input sequence
    batch_size = 1
    seq_len = 32
    
    # Initialize memory
    chamber.reset_memory(batch_size, torch.device('cpu'))
    
    # Track memory magnitude over time
    memory_mags = []
    decay_vals = []
    w_eff_vals = []
    
    for t in range(seq_len):
        # Constant input (to see pure decay behavior after initial write)
        if t == 0:
            # Strong initial input
            x_real = torch.randn(batch_size, d_model) * 2.0
            x_imag = torch.randn(batch_size, d_model) * 2.0
        else:
            # Zero input (pure decay)
            x_real = torch.zeros(batch_size, d_model)
            x_imag = torch.zeros(batch_size, d_model)
        
        t_tensor = torch.tensor([t], dtype=torch.float32)
        
        with torch.no_grad():
            out_r, out_i, diag = chamber(x_real, x_imag, t_tensor)
        
        memory_mags.append(diag['memory_mag'].item())
        decay_vals.append(diag['decay_mean'].item())
        w_eff_vals.append(diag['w_eff_mean'].item())
    
    print(f"\nDecay dynamics (after initial write at t=0):")
    print(f"  t=0:  mem_mag={memory_mags[0]:.4f}")
    print(f"  t=5:  mem_mag={memory_mags[5]:.4f}")
    print(f"  t=10: mem_mag={memory_mags[10]:.4f}")
    print(f"  t=20: mem_mag={memory_mags[20]:.4f}")
    print(f"  t=31: mem_mag={memory_mags[31]:.4f}")
    
    print(f"\nQ-factor parameters:")
    print(f"  decay_mean: {decay_vals[0]:.4f}")
    print(f"  w_eff_mean: {w_eff_vals[0]:.4f}")
    
    # Memory should decay but not explode
    assert memory_mags[31] < memory_mags[0], "Memory should decay over time with zero input"
    assert memory_mags[31] > 0, "Memory should not decay to exactly zero"
    
    # Compute effective half-life
    initial = memory_mags[0]
    for t, mag in enumerate(memory_mags):
        if mag < initial / 2:
            print(f"\n  Half-life: ~{t} timesteps")
            break
    else:
        print(f"\n  Half-life: >31 timesteps (slow decay)")
    
    print(f"\n✓ EMA behavior is correct (decay without explosion)")
    return True


def test_interference_distribution():
    """Test interference score distribution - both constructive and destructive."""
    print("\n" + "=" * 80)
    print("TEST 4: Interference Distribution")
    print("=" * 80)
    
    d_model = 64
    chamber = EchoChamberV2(d_model=d_model, n_heads=4)
    
    # Collect interference scores over many samples
    n_samples = 200
    int_reals = []
    int_mags = []
    
    chamber.eval()
    
    for i in range(n_samples):
        # Random input
        x_real = torch.randn(1, d_model) * 0.5
        x_imag = torch.randn(1, d_model) * 0.5
        t = torch.tensor([i % 20], dtype=torch.float32)
        
        chamber.reset_memory(1, torch.device('cpu'))
        
        with torch.no_grad():
            _, _, diag = chamber(x_real, x_imag, t)
        
        int_reals.append(diag['total_int_real'].item())
        int_mags.append(diag['total_int_mag'].item())
    
    int_reals = torch.tensor(int_reals)
    int_mags = torch.tensor(int_mags)
    
    print(f"\nInterference real part (before training):")
    print(f"  mean:   {int_reals.mean():.4f}")
    print(f"  std:    {int_reals.std():.4f}")
    print(f"  min:    {int_reals.min():.4f}")
    print(f"  max:    {int_reals.max():.4f}")
    print(f"  % negative: {(int_reals < 0).float().mean() * 100:.1f}%")
    print(f"  % positive: {(int_reals > 0).float().mean() * 100:.1f}%")
    
    print(f"\nInterference magnitude:")
    print(f"  mean:   {int_mags.mean():.4f}")
    print(f"  std:    {int_mags.std():.4f}")
    print(f"  min:    {int_mags.min():.4f}")
    print(f"  max:    {int_mags.max():.4f}")
    
    # Should have both positive and negative real parts
    assert (int_reals < 0).any(), "Should have some destructive interference"
    assert (int_reals > 0).any(), "Should have some constructive interference"
    
    # Magnitude should always be positive
    assert (int_mags >= 0).all(), "Magnitude should always be non-negative"
    
    print(f"\n✓ Both constructive and destructive interference present")
    return True


def test_learning_dynamics():
    """Test learning dynamics - how parameters evolve during training."""
    print("\n" + "=" * 80)
    print("TEST 5: Learning Dynamics")
    print("=" * 80)
    
    model = EchoChamberModelV2(
        vocab_size=100,
        d_model=64,
        num_layers=2,
        num_neurons=64,
        n_echo_heads=4,
    )
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    
    # Track parameter evolution
    history = {
        'loss': [],
        'log_beta_mean': [],
        'log_beta_std': [],
        'beta_mean': [],
        'w_out_mean': [],
        'w_out_std': [],
        'w_eff_mean': [],
        'trigger_mag_mean': [],
        'grad_log_beta_max': [],
        'grad_w_out_max': [],
    }
    
    # Training loop
    n_epochs = 100
    batch_size = 8
    seq_len = 20
    
    print(f"\nTraining for {n_epochs} epochs...")
    
    for epoch in range(n_epochs):
        # Generate batch
        x = torch.randint(0, 100, (batch_size, seq_len))
        target = torch.randint(0, 100, (batch_size, seq_len))
        
        # Forward (model resets memory internally)
        model.train()
        logits = model(x)  # No diagnostics needed
        loss = nn.CrossEntropyLoss()(logits.view(-1, 100), target.view(-1))
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        
        # Record before step
        with torch.no_grad():
            chamber = model.echo_chambers[0]  # First layer's echo chamber
            
            history['loss'].append(loss.item())
            history['log_beta_mean'].append(chamber.log_beta.mean().item())
            history['log_beta_std'].append(chamber.log_beta.std().item())
            history['beta_mean'].append(chamber.get_beta().mean().item())
            history['w_out_mean'].append(chamber.w_out.mean().item())
            history['w_out_std'].append(chamber.w_out.std().item())
            history['w_eff_mean'].append(chamber.get_effective_wavelength().mean().item())
            
            # Trigger magnitude
            trigger_mags = []
            for head in chamber.heads:
                trig_mag = (head.trigger_real**2 + head.trigger_imag**2).sqrt().mean()
                trigger_mags.append(trig_mag.item())
            history['trigger_mag_mean'].append(sum(trigger_mags) / len(trigger_mags))
            
            # Gradient magnitudes
            if chamber.log_beta.grad is not None:
                history['grad_log_beta_max'].append(chamber.log_beta.grad.abs().max().item())
            else:
                history['grad_log_beta_max'].append(0)
            
            if chamber.w_out.grad is not None:
                history['grad_w_out_max'].append(chamber.w_out.grad.abs().max().item())
            else:
                history['grad_w_out_max'].append(0)
        
        # Step
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
    
    # Print evolution
    print(f"\nParameter evolution:")
    print(f"  {'Metric':<20} {'Initial':>10} {'Final':>10} {'Change':>10}")
    print(f"  {'-'*50}")
    
    for key in ['log_beta_mean', 'beta_mean', 'w_out_std', 'w_eff_mean', 'trigger_mag_mean']:
        initial = history[key][0]
        final = history[key][-1]
        change = final - initial
        print(f"  {key:<20} {initial:>10.4f} {final:>10.4f} {change:>+10.4f}")
    
    print(f"\nGradient health:")
    print(f"  grad_log_beta_max: mean={sum(history['grad_log_beta_max'])/len(history['grad_log_beta_max']):.6f}")
    print(f"  grad_w_out_max:    mean={sum(history['grad_w_out_max'])/len(history['grad_w_out_max']):.6f}")
    
    print(f"\nLoss:")
    print(f"  Initial: {history['loss'][0]:.4f}")
    print(f"  Final:   {history['loss'][-1]:.4f}")
    
    # Verify beta stays positive throughout
    assert all(b > 0 for b in history['beta_mean']), "Beta should stay positive"
    
    print(f"\n✓ Learning dynamics are stable")
    return True


def test_memory_bounded():
    """Test that memory growth is bounded."""
    print("\n" + "=" * 80)
    print("TEST 6: Memory Bounded Growth")
    print("=" * 80)
    
    d_model = 64
    chamber = EchoChamberV2(d_model=d_model, n_heads=4)
    chamber.eval()
    
    # Long sequence with constant strong input
    seq_len = 100
    memory_mags = []
    
    chamber.reset_memory(1, torch.device('cpu'))
    
    for t in range(seq_len):
        # Strong constant input (worst case for growth)
        x_real = torch.ones(1, d_model) * 2.0
        x_imag = torch.ones(1, d_model) * 2.0
        t_tensor = torch.tensor([t], dtype=torch.float32)
        
        with torch.no_grad():
            _, _, diag = chamber(x_real, x_imag, t_tensor)
        
        memory_mags.append(diag['memory_mag'].item())
    
    print(f"\nMemory magnitude over time (constant strong input):")
    print(f"  t=0:   {memory_mags[0]:.4f}")
    print(f"  t=10:  {memory_mags[10]:.4f}")
    print(f"  t=50:  {memory_mags[50]:.4f}")
    print(f"  t=99:  {memory_mags[99]:.4f}")
    
    # Check for bounded growth
    # Memory should stabilize, not explode
    growth_rate = memory_mags[99] / max(memory_mags[50], 1e-6)
    print(f"\n  Growth rate (t=99/t=50): {growth_rate:.4f}")
    
    # Should stabilize (growth rate near 1.0)
    assert growth_rate < 2.0, "Memory should not explode"
    assert not torch.isnan(torch.tensor(memory_mags[-1])), "Memory should not become NaN"
    assert not torch.isinf(torch.tensor(memory_mags[-1])), "Memory should not become Inf"
    
    print(f"\n✓ Memory growth is bounded")
    return True


def test_memory_task():
    """Test actual memory task performance."""
    print("\n" + "=" * 80)
    print("TEST 7: Memory Task (Copy-Value-Query)")
    print("=" * 80)
    
    model = EchoChamberModelV2(
        vocab_size=100,
        d_model=64,
        num_layers=2,
        num_neurons=64,
        n_echo_heads=4,
    )
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    
    def create_task(batch_size=16):
        """Create copy-value-query task."""
        seq_len = 16
        x = torch.randint(10, 50, (batch_size, seq_len))
        target = torch.zeros_like(x)
        
        # Marker tokens
        MARKER = 1
        QUERY_MARKER = 2
        
        for b in range(batch_size):
            value = torch.randint(50, 100, (1,)).item()
            
            x[b, 4] = MARKER
            x[b, 5] = value
            x[b, 14] = QUERY_MARKER
            
            target[b, 15] = value
        
        return x, target
    
    # Training
    best_acc = 0
    n_epochs = 200
    
    for epoch in range(n_epochs):
        x, target = create_task(batch_size=32)
        
        # Forward (model resets memory internally)
        model.train()
        logits = model(x)
        
        loss = nn.CrossEntropyLoss()(logits[:, 15, :], target[:, 15])
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        # Accuracy
        with torch.no_grad():
            pred = logits[:, 15, :].argmax(dim=-1)
            acc = (pred == target[:, 15]).float().mean().item() * 100
            best_acc = max(best_acc, acc)
        
        if (epoch + 1) % 50 == 0:
            chamber = model.echo_chambers[0]
            beta = chamber.get_beta()
            w_eff = chamber.get_effective_wavelength()
            decay = chamber.compute_decay()
            
            print(f"  Epoch {epoch+1}: loss={loss.item():.4f}, acc={acc:.1f}%, "
                  f"β={beta.mean():.4f}, w_eff={w_eff.mean():.4f}, decay={decay.mean():.4f}")
    
    print(f"\nFinal: Best accuracy = {best_acc:.1f}%")
    print(f"\n✓ Memory task completed")
    return best_acc


if __name__ == '__main__':
    print("=" * 80)
    print("ECHO CHAMBER V2 - COMPREHENSIVE Q-EMA TEST")
    print("=" * 80)
    
    tests = [
        ('Beta Parameterization', test_beta_parameterization),
        ('Gradient Stability', test_gradient_stability),
        ('EMA Behavior', test_ema_behavior),
        ('Interference Distribution', test_interference_distribution),
        ('Learning Dynamics', test_learning_dynamics),
        ('Memory Bounded', test_memory_bounded),
        ('Memory Task', test_memory_task),
    ]
    
    results = []
    for name, test_fn in tests:
        try:
            result = test_fn()
            results.append((name, 'PASS', result))
        except Exception as e:
            results.append((name, 'FAIL', str(e)))
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    for name, status, result in results:
        print(f"  {name:<30} {status}")
    
    n_passed = sum(1 for _, s, _ in results if s == 'PASS')
    print(f"\n  {n_passed}/{len(tests)} tests passed")

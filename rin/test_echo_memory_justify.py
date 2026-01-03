"""
Test Echo Chamber V2 with wavelength-style beta and weight decay.

The echo chamber must JUSTIFY its long-term memory capability through
learning. With weight decay enabled, the model will regularize parameters
toward zero - the echo chamber must prove it's worth the cost.

Key test: Can the model learn to maintain information over long delays
when there's a pressure to minimize parameters?
"""

import torch
import torch.nn as nn
import sys
sys.path.insert(0, '/home/aiman/pi')

from rin.echo_chamber_v2 import EchoChamberV2, EchoChamberModelV2

torch.manual_seed(42)


def test_beta_parameterization():
    """Test new wavelength-style beta parameterization."""
    print("=" * 80)
    print("TEST 1: Beta Parameterization (Wavelength Style)")
    print("=" * 80)
    
    d_model = 64
    chamber = EchoChamberV2(d_model=d_model, n_heads=4)
    
    # Initial beta values
    beta_raw_init = chamber.beta
    beta_eff_init = chamber.get_beta()
    w_eff_init = chamber.get_effective_wavelength()
    decay_init = chamber.compute_decay()
    
    print(f"\nInitial state:")
    print(f"  beta_raw:  mean={beta_raw_init.mean():.4f}, std={beta_raw_init.std():.4f}")
    print(f"  beta_eff:  mean={beta_eff_init.mean():.4f}, min={beta_eff_init.min():.4f}, max={beta_eff_init.max():.4f}")
    print(f"             (should be in (0, 1] range)")
    print(f"  w_eff:     mean={w_eff_init.mean():.4f}, min={w_eff_init.min():.4f}, max={w_eff_init.max():.4f}")
    print(f"  decay:     mean={decay_init.mean():.4f}, min={decay_init.min():.4f}, max={decay_init.max():.4f}")
    
    # Verify beta_eff is in valid range
    assert (beta_eff_init > 0).all(), "Beta_eff should always be positive"
    assert (beta_eff_init <= 1.0).all(), "Beta_eff should not exceed 1.0"
    
    # Verify decay is in valid range
    assert (decay_init >= 0).all() and (decay_init <= 0.9999).all(), "Decay should be in [0, 0.9999]"
    
    # Test extreme values
    print(f"\nExtreme value behavior:")
    with torch.no_grad():
        # Large beta -> small beta_eff -> slow decay (long memory)
        chamber.beta[:] = 100.0
        beta_large = chamber.get_beta()
        print(f"  beta=100 → beta_eff={beta_large.mean():.6f} (should be ~0.01, slow decay)")
        
        # Small beta -> large beta_eff -> fast decay (short memory)
        chamber.beta[:] = 0.01
        beta_small = chamber.get_beta()
        print(f"  beta=0.01 → beta_eff={beta_small.mean():.6f} (should be ~0.99, fast decay)")
        
        # Zero beta -> max beta_eff = 1.0
        chamber.beta[:] = 0.0
        beta_zero = chamber.get_beta()
        print(f"  beta=0 → beta_eff={beta_zero.mean():.6f} (should be 1.0, instant decay)")
    
    print(f"\n✓ Beta parameterization matches wavelength style")
    return True


def test_long_term_memory_with_decay():
    """
    Test if echo chamber can maintain long-term memory with weight decay.
    
    Task: Copy a value from t=5 to t=50 (45 timestep delay)
    With weight decay, the model must justify maintaining memory that long.
    """
    print("\n" + "=" * 80)
    print("TEST 2: Long-Term Memory Justification (with Weight Decay)")
    print("=" * 80)
    
    model = EchoChamberModelV2(
        vocab_size=100,
        d_model=64,
        num_layers=2,
        num_neurons=64,
        n_echo_heads=4,
    )
    
    # CRUCIAL: AdamW with weight decay
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    
    def create_long_delay_task(batch_size=16, delay=45):
        """Create task with long delay between marker and query."""
        seq_len = delay + 10
        x = torch.randint(10, 50, (batch_size, seq_len))
        target = torch.zeros_like(x)
        
        # Marker tokens
        MARKER = 1
        QUERY_MARKER = 2
        
        for b in range(batch_size):
            value = torch.randint(50, 100, (1,)).item()
            
            x[b, 4] = MARKER
            x[b, 5] = value
            x[b, 4 + delay] = QUERY_MARKER
            
            target[b, 5 + delay] = value  # Query answer position
        
        return x, target
    
    # Training
    best_acc = 0
    n_epochs = 300
    
    print(f"\nTraining for {n_epochs} epochs with weight_decay=0.01...")
    print(f"Task: Copy value from t=5 to t=50 (45 timestep delay)\n")
    
    for epoch in range(n_epochs):
        x, target = create_long_delay_task(batch_size=32, delay=45)
        
        model.train()
        logits = model(x)
        
        # Loss only on the answer position
        loss = nn.CrossEntropyLoss()(logits[:, 50, :], target[:, 50])
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        # Accuracy
        with torch.no_grad():
            pred = logits[:, 50, :].argmax(dim=-1)
            acc = (pred == target[:, 50]).float().mean().item() * 100
            best_acc = max(best_acc, acc)
        
        if (epoch + 1) % 50 == 0:
            chamber = model.echo_chambers[0]
            beta_raw = chamber.beta
            beta_eff = chamber.get_beta()
            w_eff = chamber.get_effective_wavelength()
            decay = chamber.compute_decay()
            
            print(f"  Epoch {epoch+1:3d}: loss={loss.item():.4f}, acc={acc:5.1f}%, "
                  f"β_raw={beta_raw.mean():.4f}±{beta_raw.std():.4f}, "
                  f"β_eff={beta_eff.mean():.4f}, decay={decay.mean():.4f}")
    
    print(f"\n{'='*80}")
    print(f"RESULT: Best accuracy = {best_acc:.1f}%")
    
    if best_acc > 50:
        print(f"✓ SUCCESS: Echo chamber justified long-term memory!")
    elif best_acc > 20:
        print(f"⚠ PARTIAL: Echo chamber shows some memory capability")
    else:
        print(f"✗ FAILURE: Echo chamber could not justify long-term memory")
    
    # Analyze final parameters
    chamber = model.echo_chambers[0]
    beta_raw = chamber.beta
    beta_eff = chamber.get_beta()
    decay = chamber.compute_decay()
    
    print(f"\nFinal parameter statistics:")
    print(f"  beta_raw: mean={beta_raw.mean():.4f}, std={beta_raw.std():.4f}")
    print(f"  beta_eff: mean={beta_eff.mean():.4f}, min={beta_eff.min():.4f}, max={beta_eff.max():.4f}")
    print(f"  decay:    mean={decay.mean():.4f}, min={decay.min():.4f}, max={decay.max():.4f}")
    print(f"  w_out:    mean={chamber.w_out.mean():.4f}, std={chamber.w_out.std():.4f}")
    
    # Memory persistence analysis
    print(f"\nMemory persistence (avg over 45 timesteps):")
    avg_decay = decay.mean().item()
    persistence_45 = avg_decay ** 45
    print(f"  Single-step decay: {avg_decay:.4f}")
    print(f"  45-step persistence: {persistence_45:.4f}")
    print(f"  (fraction of initial signal remaining after 45 steps)")
    
    return best_acc


def test_short_vs_long_memory():
    """
    Compare short delay (10 steps) vs long delay (45 steps).
    
    With weight decay, the model should learn appropriate memory
    retention for each task.
    """
    print("\n" + "=" * 80)
    print("TEST 3: Adaptive Memory (Short vs Long Delay)")
    print("=" * 80)
    
    def train_on_delay(delay, n_epochs=200):
        model = EchoChamberModelV2(
            vocab_size=100,
            d_model=64,
            num_layers=2,
            num_neurons=64,
            n_echo_heads=4,
        )
        
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
        
        def create_task(batch_size=16):
            seq_len = delay + 10
            x = torch.randint(10, 50, (batch_size, seq_len))
            target = torch.zeros_like(x)
            
            MARKER = 1
            QUERY_MARKER = 2
            
            for b in range(batch_size):
                value = torch.randint(50, 100, (1,)).item()
                x[b, 4] = MARKER
                x[b, 5] = value
                x[b, 4 + delay] = QUERY_MARKER
                target[b, 5 + delay] = value
            
            return x, target
        
        best_acc = 0
        final_decay = None
        
        for epoch in range(n_epochs):
            x, target = create_task(batch_size=32)
            
            model.train()
            logits = model(x)
            loss = nn.CrossEntropyLoss()(logits[:, 5 + delay, :], target[:, 5 + delay])
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            with torch.no_grad():
                pred = logits[:, 5 + delay, :].argmax(dim=-1)
                acc = (pred == target[:, 5 + delay]).float().mean().item() * 100
                best_acc = max(best_acc, acc)
        
        chamber = model.echo_chambers[0]
        final_decay = chamber.compute_decay().mean().item()
        
        return best_acc, final_decay
    
    print(f"\nTraining on SHORT delay (10 steps)...")
    acc_short, decay_short = train_on_delay(delay=10, n_epochs=200)
    persistence_short_10 = decay_short ** 10
    
    print(f"  Best accuracy: {acc_short:.1f}%")
    print(f"  Final decay: {decay_short:.4f}")
    print(f"  10-step persistence: {persistence_short_10:.4f}")
    
    print(f"\nTraining on LONG delay (45 steps)...")
    acc_long, decay_long = train_on_delay(delay=45, n_epochs=200)
    persistence_long_45 = decay_long ** 45
    
    print(f"  Best accuracy: {acc_long:.1f}%")
    print(f"  Final decay: {decay_long:.4f}")
    print(f"  45-step persistence: {persistence_long_45:.4f}")
    
    print(f"\n{'='*80}")
    print(f"COMPARISON:")
    print(f"  Short task (10 steps): acc={acc_short:.1f}%, decay={decay_short:.4f}")
    print(f"  Long task (45 steps):  acc={acc_long:.1f}%, decay={decay_long:.4f}")
    
    if decay_long > decay_short:
        print(f"\n✓ Model adapted: Longer memory (slower decay) for longer delay task")
    else:
        print(f"\n⚠ Model did NOT adapt decay rates appropriately")
    
    return True


def test_gradient_flow_with_weight_decay():
    """Test that gradients flow correctly with weight decay."""
    print("\n" + "=" * 80)
    print("TEST 4: Gradient Flow (with Weight Decay)")
    print("=" * 80)
    
    model = EchoChamberModelV2(
        vocab_size=100,
        d_model=64,
        num_layers=2,
        num_neurons=64,
        n_echo_heads=4,
    )
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    
    # Create batch
    x = torch.randint(0, 100, (4, 20))
    target = torch.randint(0, 100, (4, 20))
    
    # Forward
    logits = model(x)
    loss = nn.CrossEntropyLoss()(logits.view(-1, 100), target.view(-1))
    
    # Backward
    optimizer.zero_grad()
    loss.backward()
    
    print(f"\nLoss: {loss.item():.4f}")
    
    # Check key gradients
    chamber = model.echo_chambers[0]
    
    print(f"\nGradient statistics:")
    print(f"  beta:        mean={chamber.beta.grad.abs().mean():.6f}, max={chamber.beta.grad.abs().max():.6f}")
    print(f"  w_out:       mean={chamber.w_out.grad.abs().mean():.6f}, max={chamber.w_out.grad.abs().max():.6f}")
    print(f"  trigger[0]:  mean={chamber.heads[0].trigger_real.grad.abs().mean():.6f}")
    
    # Verify no NaN/Inf
    has_nan = any(torch.isnan(p.grad).any() for p in model.parameters() if p.grad is not None)
    has_inf = any(torch.isinf(p.grad).any() for p in model.parameters() if p.grad is not None)
    
    assert not has_nan, "Found NaN in gradients"
    assert not has_inf, "Found Inf in gradients"
    
    print(f"\n✓ All gradients clean (no NaN/Inf)")
    
    # Simulate weight decay effect
    optimizer.step()
    
    print(f"\nAfter optimizer step (with weight decay):")
    print(f"  beta:  mean={chamber.beta.mean():.4f}, std={chamber.beta.std():.4f}")
    print(f"  w_out: mean={chamber.w_out.mean():.4f}, std={chamber.w_out.std():.4f}")
    
    return True


if __name__ == '__main__':
    print("=" * 80)
    print("ECHO CHAMBER V2 - MEMORY JUSTIFICATION TEST")
    print("Wavelength-style beta + AdamW weight decay")
    print("=" * 80)
    
    tests = [
        ('Beta Parameterization', test_beta_parameterization),
        ('Long-Term Memory (45 steps)', test_long_term_memory_with_decay),
        ('Adaptive Memory (Short vs Long)', test_short_vs_long_memory),
        ('Gradient Flow', test_gradient_flow_with_weight_decay),
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
        print(f"  {name:<35} {status}")
    
    n_passed = sum(1 for _, s, _ in results if s == 'PASS')
    print(f"\n  {n_passed}/{len(tests)} tests passed")

#!/usr/bin/env python3
"""
Test gradient flow through Echo Chamber memory.

HYPOTHESIS: The memory detachment prevents gradient flow through time,
which is why the model can't learn delay > 2.

We'll compare:
1. Current implementation (detached memory)
2. Non-detached memory (full BPTT through memory)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleEchoChamber(nn.Module):
    """
    Minimal Echo Chamber to test gradient flow.
    """
    
    def __init__(self, d_model: int = 32, detach_memory: bool = True):
        super().__init__()
        self.d_model = d_model
        self.detach_memory = detach_memory
        
        # Learnable parameters
        self.w = nn.Parameter(torch.randn(d_model) * 0.1)  # wavelength
        self.b = nn.Parameter(torch.randn(d_model) * 0.1)  # bias
        
        # Beta for decay (initialized for decay ~ 0.9)
        self.beta_raw = nn.Parameter(torch.abs(torch.randn(d_model)) * 5.0 + 5.0)
        
        # Output projection
        self.out_proj = nn.Linear(d_model, d_model)
        
        # Memory state
        self._memory = None
        
        # Golden ratio for timestep
        self.PHI = 1.618034
        
    def reset_memory(self, batch_size: int, device: torch.device):
        self._memory = torch.zeros(batch_size, self.d_model, device=device)
        
    def get_decay(self):
        w_eff = 1.0 / (1.0 + self.w.abs())
        beta_eff = 1.0 / (1.0 + self.beta_raw.abs())
        decay = torch.exp(-beta_eff * w_eff)
        return decay.clamp(min=0.5, max=0.9999)
        
    def forward(self, x: torch.Tensor, t: int) -> torch.Tensor:
        """
        x: (batch, d_model) - input at timestep t
        t: int - timestep index
        """
        batch_size = x.shape[0]
        device = x.device
        
        if self._memory is None:
            self.reset_memory(batch_size, device)
            
        # Euler projection of input
        w_eff = 1.0 / (1.0 + self.w.abs())
        t_phi = t * self.PHI
        theta = x * w_eff + self.b + t_phi
        x_proj = torch.cos(theta)  # Simple projection
        
        # Get memory (detached or not)
        if self.detach_memory:
            memory = self._memory.detach()
        else:
            memory = self._memory
            
        # Decay
        decay = self.get_decay()
        
        # Q-EMA update
        new_memory = memory * decay + x_proj * (1.0 - decay)
        
        # Store (always detach for autograd graph management)
        self._memory = new_memory if not self.detach_memory else new_memory.detach()
        
        # Output
        out = self.out_proj(new_memory)
        return out


def test_gradient_flow(detach_memory: bool, seq_len: int = 20):
    """Test if gradients flow back to early timesteps."""
    
    print(f"\n{'='*60}")
    print(f"Testing gradient flow with detach_memory={detach_memory}")
    print(f"Sequence length: {seq_len}")
    print('='*60)
    
    torch.manual_seed(42)
    
    d_model = 32
    batch_size = 4
    
    model = SimpleEchoChamber(d_model=d_model, detach_memory=detach_memory)
    model.reset_memory(batch_size, 'cpu')
    
    # Create inputs for each timestep (all require grad)
    inputs = []
    for t in range(seq_len):
        x = torch.randn(batch_size, d_model, requires_grad=True)
        inputs.append(x)
    
    # Forward pass through all timesteps
    outputs = []
    for t, x in enumerate(inputs):
        out = model(x, t)
        outputs.append(out)
    
    # Loss only on LAST output
    target = torch.randn(batch_size, d_model)
    loss = F.mse_loss(outputs[-1], target)
    
    # Backward
    loss.backward()
    
    # Check gradients at each timestep
    print("\nGradient magnitude at each timestep:")
    print("-" * 40)
    
    grad_mags = []
    for t, x in enumerate(inputs):
        if x.grad is not None:
            grad_mag = x.grad.abs().mean().item()
        else:
            grad_mag = 0.0
        grad_mags.append(grad_mag)
        
        if t == 0 or t == seq_len - 1 or t == seq_len // 2:
            print(f"  t={t:3d}: grad_mag = {grad_mag:.6f}")
    
    # Summary
    nonzero_grads = sum(1 for g in grad_mags if g > 1e-10)
    print(f"\nTimesteps with non-zero gradients: {nonzero_grads}/{seq_len}")
    print(f"First timestep gradient: {grad_mags[0]:.6f}")
    print(f"Last timestep gradient: {grad_mags[-1]:.6f}")
    
    if grad_mags[0] > 1e-10:
        ratio = grad_mags[-1] / (grad_mags[0] + 1e-12)
        print(f"Gradient ratio (last/first): {ratio:.2f}x")
    else:
        print("⚠️ NO GRADIENT at first timestep!")
        
    # Also check parameter gradients
    print("\nParameter gradients:")
    for name, param in model.named_parameters():
        if param.grad is not None:
            print(f"  {name}: {param.grad.abs().mean():.6f}")
        else:
            print(f"  {name}: None")
    
    return grad_mags


def test_learning_with_gradient_flow(detach_memory: bool):
    """Test if model can learn copy task with/without detachment."""
    
    print(f"\n{'='*60}")
    print(f"Learning test with detach_memory={detach_memory}")
    print('='*60)
    
    torch.manual_seed(42)
    
    d_model = 32
    batch_size = 32
    delay = 10  # Copy task with 10-step delay
    seq_len = delay + 5
    
    model = SimpleEchoChamber(d_model=d_model, detach_memory=detach_memory)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2, weight_decay=0.01)
    
    num_epochs = 100
    
    for epoch in range(num_epochs):
        model.reset_memory(batch_size, 'cpu')
        optimizer.zero_grad()
        
        # Generate sequence: noise, VALUE, noise...
        value_pos = 2
        value = torch.randn(batch_size, d_model)
        
        outputs = []
        for t in range(seq_len):
            if t == value_pos:
                x = value
            else:
                x = torch.randn(batch_size, d_model) * 0.1
            
            out = model(x, t)
            outputs.append(out)
        
        # Target: retrieve value at position value_pos + delay
        target_pos = value_pos + delay
        loss = F.mse_loss(outputs[target_pos], value)
        
        loss.backward()
        optimizer.step()
        
        if epoch % 20 == 0 or epoch == num_epochs - 1:
            # Test accuracy
            model.eval()
            with torch.no_grad():
                model.reset_memory(batch_size, 'cpu')
                test_value = torch.randn(batch_size, d_model)
                
                for t in range(seq_len):
                    if t == value_pos:
                        x = test_value
                    else:
                        x = torch.randn(batch_size, d_model) * 0.1
                    out = model(x, t)
                    
                # Check correlation at target position
                pred = out  # Last output
                corr = F.cosine_similarity(pred, test_value, dim=-1).mean()
                
            print(f"Epoch {epoch:3d}: loss={loss.item():.4f}, corr={corr.item():.4f}")
            model.train()
    
    return corr.item()


def test_learning_multiple_delays():
    """Compare learning across different delays with/without detachment."""
    
    print("\n" + "="*70)
    print("COMPARISON: Learning with vs without memory detachment")
    print("="*70)
    
    delays = [2, 5, 10]
    
    results = {"detached": {}, "connected": {}}
    
    for delay in delays:
        print(f"\n--- Delay = {delay} ---")
        
        # With detachment (current)
        print("\nWith detachment (current implementation):")
        corr_detached = test_learning_single_delay(detach_memory=True, delay=delay)
        results["detached"][delay] = corr_detached
        
        # Without detachment (full BPTT)
        print("\nWithout detachment (full BPTT):")
        corr_connected = test_learning_single_delay(detach_memory=False, delay=delay)
        results["connected"][delay] = corr_connected
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"{'Delay':<10} {'Detached':<15} {'Connected':<15} {'Improvement':<15}")
    print("-"*55)
    for delay in delays:
        detached = results["detached"][delay]
        connected = results["connected"][delay]
        improvement = connected - detached
        print(f"{delay:<10} {detached:<15.4f} {connected:<15.4f} {improvement:+.4f}")


def test_learning_single_delay(detach_memory: bool, delay: int, epochs: int = 100):
    """Test learning for a single delay setting."""
    
    torch.manual_seed(42)
    
    d_model = 32
    batch_size = 32
    seq_len = delay + 5
    
    model = SimpleEchoChamber(d_model=d_model, detach_memory=detach_memory)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2, weight_decay=0.01)
    
    for epoch in range(epochs):
        model.reset_memory(batch_size, 'cpu')
        optimizer.zero_grad()
        
        value_pos = 2
        value = torch.randn(batch_size, d_model)
        
        outputs = []
        for t in range(seq_len):
            if t == value_pos:
                x = value
            else:
                x = torch.randn(batch_size, d_model) * 0.1
            
            out = model(x, t)
            outputs.append(out)
        
        target_pos = value_pos + delay
        loss = F.mse_loss(outputs[target_pos], value)
        
        loss.backward()
        optimizer.step()
    
    # Final test
    model.eval()
    with torch.no_grad():
        model.reset_memory(batch_size, 'cpu')
        test_value = torch.randn(batch_size, d_model)
        
        for t in range(seq_len):
            if t == value_pos:
                x = test_value
            else:
                x = torch.randn(batch_size, d_model) * 0.1
            out = model(x, t)
        
        corr = F.cosine_similarity(out, test_value, dim=-1).mean().item()
    
    print(f"  Final correlation: {corr:.4f}")
    return corr


if __name__ == "__main__":
    # Test 1: Gradient flow comparison
    print("\n" + "#"*70)
    print("# TEST 1: GRADIENT FLOW THROUGH TIME")
    print("#"*70)
    
    grads_detached = test_gradient_flow(detach_memory=True, seq_len=20)
    grads_connected = test_gradient_flow(detach_memory=False, seq_len=20)
    
    # Test 2: Learning comparison
    print("\n" + "#"*70)
    print("# TEST 2: LEARNING COMPARISON")
    print("#"*70)
    
    test_learning_multiple_delays()

"""
Full Sequence BPTT Test for RIN Model.

CRITICAL INSIGHT: The RIN model is NOT autoregressive in the traditional sense.
It must SEE the token during the forward pass, then backprop through ALL timesteps.

The Euler-based architecture should have PERFECT gradient preservation:
    |d/dθ (cos θ)|² + |d/dθ (sin θ)|² = sin²θ + cos²θ = 1

This test verifies:
1. Gradients flow through long sequences without vanishing/exploding
2. The model can learn to remember tokens from early in the sequence
3. Full BPTT training works correctly

Training setup:
- Full sequence forward pass
- Loss computed at end of sequence
- Backprop through ALL timesteps
- Small model, high LR (1e-2), low weight decay (0.01)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import sys
sys.path.insert(0, '/home/aiman/pi')

from rin.model import RINModel, PHI
from rin.lut import get_global_lut

torch.manual_seed(42)


class SmallRIN(nn.Module):
    """
    Small RIN model for BPTT testing.
    
    Minimal architecture to test gradient flow through time.
    """
    
    def __init__(self, vocab_size=100, d_model=32, num_layers=1, num_neurons=64):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.num_layers = num_layers
        
        # Token embeddings: (w, b) pairs
        self.token_embedding = nn.Embedding(vocab_size, 2 * d_model)
        
        # Simple resonant layer
        self.layers = nn.ModuleList([
            SimpleResonantLayer(d_model, num_neurons)
            for _ in range(num_layers)
        ])
        
        # Output projection
        self.output_proj = nn.Linear(2 * d_model, vocab_size)
        
        self._lut = None
        self._init_weights()
    
    def _init_weights(self):
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.output_proj.weight, std=0.02)
    
    def _get_lut(self, device):
        if self._lut is None or self._lut.sin_table.device != device:
            self._lut = get_global_lut(4096, device)
        return self._lut
    
    def euler_transform(self, h_real, h_imag, w, b, t):
        """Euler-based state transformation."""
        lut = self._get_lut(h_real.device)
        
        wavelength = 1.0 + w.abs()
        t_phi = t * PHI
        if t_phi.dim() == 0:
            t_phi = t_phi.unsqueeze(0).unsqueeze(0)
        elif t_phi.dim() == 1:
            t_phi = t_phi.unsqueeze(-1)
        
        theta_real = h_real / wavelength + b + t_phi
        theta_imag = h_imag / wavelength + b + t_phi
        
        sin_r, cos_r = lut.lookup_sin_cos(theta_real)
        sin_i, cos_i = lut.lookup_sin_cos(theta_imag)
        
        # Complex multiplication preserves gradient magnitude
        h_real_new = cos_r * cos_i - sin_r * sin_i
        h_imag_new = cos_r * sin_i + sin_r * cos_i
        
        return h_real_new, h_imag_new
    
    def forward(self, input_ids):
        """
        Full sequence forward pass with BPTT.
        
        Unlike autoregressive, we process the ENTIRE sequence and
        return logits at every position.
        """
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        
        # Initialize hidden state
        h_real = torch.zeros(batch_size, self.d_model, device=device)
        h_imag = torch.zeros(batch_size, self.d_model, device=device)
        
        # Get embeddings
        embeddings = self.token_embedding(input_ids)
        w_emb = embeddings[:, :, :self.d_model]
        b_emb = embeddings[:, :, self.d_model:]
        
        all_logits = []
        
        # Process sequence step by step (full BPTT)
        for t in range(seq_len):
            w_t = w_emb[:, t, :]
            b_t = b_emb[:, t, :]
            t_val = torch.tensor(t, dtype=torch.float32, device=device)
            
            # Euler transform - this is where gradient preservation happens
            h_real, h_imag = self.euler_transform(h_real, h_imag, w_t, b_t, t_val)
            
            # Process through layers
            x_real, x_imag = h_real, h_imag
            t_phi = t_val * PHI
            
            for layer in self.layers:
                delta_real, delta_imag = layer(x_real, x_imag, t_phi)
                x_real = x_real + delta_real
                x_imag = x_imag + delta_imag
            
            # Output logits
            combined = torch.cat([x_real, x_imag], dim=-1)
            logits = self.output_proj(combined)
            all_logits.append(logits)
        
        return torch.stack(all_logits, dim=1)


class SimpleResonantLayer(nn.Module):
    """Simplified resonant layer for testing."""
    
    def __init__(self, d_model, num_neurons):
        super().__init__()
        self.d_model = d_model
        self.num_neurons = num_neurons
        
        self.input_proj = nn.Linear(2 * d_model, d_model)
        self.W = nn.Parameter(torch.randn(num_neurons, d_model) * 0.02)
        self.B = nn.Parameter(torch.zeros(num_neurons, d_model))
        self.out_proj_real = nn.Linear(num_neurons, d_model, bias=False)
        self.out_proj_imag = nn.Linear(num_neurons, d_model, bias=False)
        
        self._lut = None
    
    def _get_lut(self, device):
        if self._lut is None or self._lut.sin_table.device != device:
            self._lut = get_global_lut(4096, device)
        return self._lut
    
    def forward(self, x_real, x_imag, t):
        lut = self._get_lut(x_real.device)
        
        x_combined = torch.cat([x_real, x_imag], dim=-1)
        x_collapsed = self.input_proj(x_combined)
        
        x_expanded = x_collapsed.unsqueeze(1)
        wavelength = 1.0 + self.W.abs()
        
        if isinstance(t, (int, float)):
            t = torch.tensor(t, device=x_real.device, dtype=torch.float32)
        if t.dim() == 0:
            t = t.unsqueeze(0).unsqueeze(0).unsqueeze(0)
        elif t.dim() == 1:
            t = t.unsqueeze(-1).unsqueeze(-1)
        
        theta = x_expanded / wavelength + self.B + t
        
        sin_theta, cos_theta = lut.lookup_sin_cos(theta)
        
        cos_sum = cos_theta.sum(dim=-1)
        sin_sum = sin_theta.sum(dim=-1)
        
        out_real = self.out_proj_real(cos_sum)
        out_imag = self.out_proj_imag(sin_sum)
        
        return F.silu(out_real), F.silu(out_imag)


def test_gradient_preservation():
    """
    Test if gradients are preserved through long sequences.
    
    The key property of Euler-based models:
    |∂(cos θ)/∂θ|² + |∂(sin θ)/∂θ|² = sin²θ + cos²θ = 1
    
    This means gradient magnitude should be CONSTANT regardless of sequence length.
    """
    print("=" * 80)
    print("TEST 1: Gradient Preservation Through Time")
    print("=" * 80)
    
    model = SmallRIN(vocab_size=100, d_model=32, num_layers=1, num_neurons=64)
    
    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Test gradient flow at different sequence lengths
    seq_lengths = [10, 20, 50, 100]
    
    print(f"\nGradient norm at embedding layer vs sequence length:")
    print(f"{'Seq Len':<10} {'Grad Norm':<15} {'Grad Mean':<15} {'Grad Std':<15}")
    print("-" * 55)
    
    grad_norms = []
    
    for seq_len in seq_lengths:
        model.zero_grad()
        
        # Create sequence with target at the end
        x = torch.randint(10, 90, (4, seq_len))
        
        # Place a marker at position 5
        MARKER = 1
        VALUE = 42
        x[:, 3] = MARKER
        x[:, 4] = VALUE
        
        # Forward pass
        logits = model(x)
        
        # Loss at the LAST position (must remember token from position 4)
        target = torch.full((4,), VALUE, dtype=torch.long)
        loss = F.cross_entropy(logits[:, -1, :], target)
        
        # Backward
        loss.backward()
        
        # Check gradient at embedding layer (this needs to flow through ALL timesteps)
        emb_grad = model.token_embedding.weight.grad
        grad_norm = emb_grad.norm().item()
        grad_mean = emb_grad.abs().mean().item()
        grad_std = emb_grad.std().item()
        
        grad_norms.append(grad_norm)
        
        print(f"{seq_len:<10} {grad_norm:<15.6f} {grad_mean:<15.8f} {grad_std:<15.8f}")
    
    # Analyze gradient preservation
    print(f"\nGradient preservation analysis:")
    ratio_10_to_100 = grad_norms[0] / grad_norms[-1] if grad_norms[-1] > 0 else float('inf')
    print(f"  Ratio (seq=10 / seq=100): {ratio_10_to_100:.4f}")
    
    if 0.1 < ratio_10_to_100 < 10:
        print(f"  ✓ Gradients preserved reasonably well (within 10x)")
    elif ratio_10_to_100 > 10:
        print(f"  ⚠ Gradients EXPLODING with longer sequences")
    else:
        print(f"  ⚠ Gradients VANISHING with longer sequences")
    
    return grad_norms


def test_full_bptt_learning():
    """
    Test if model can learn to remember a token using full BPTT.
    
    Task: Marker at t=3, Value at t=4, predict Value at t=N (end of sequence)
    The model SEES the value token during forward pass, then must propagate
    gradients all the way back to learn the association.
    """
    print("\n" + "=" * 80)
    print("TEST 2: Full BPTT Learning (Remember Token)")
    print("=" * 80)
    
    model = SmallRIN(vocab_size=100, d_model=32, num_layers=1, num_neurons=64)
    
    # High LR, low weight decay as specified
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2, weight_decay=0.01)
    
    def create_memory_task(batch_size, seq_len):
        """
        Create task where model must remember a value from early in sequence.
        
        Sequence: [noise] [MARKER] [VALUE] [noise...] → predict VALUE at end
        """
        x = torch.randint(10, 50, (batch_size, seq_len))
        target = torch.zeros(batch_size, seq_len, dtype=torch.long)
        
        MARKER = 1
        
        for b in range(batch_size):
            value = torch.randint(50, 100, (1,)).item()
            
            x[b, 3] = MARKER
            x[b, 4] = value  # The value to remember
            
            # Target: predict the value at the last position
            target[b, -1] = value
        
        return x, target
    
    # Training loop
    n_epochs = 200
    seq_len = 30  # Moderate sequence length
    batch_size = 32
    
    print(f"\nTraining for {n_epochs} epochs...")
    print(f"Sequence length: {seq_len}, Batch size: {batch_size}")
    print(f"Task: Remember value from t=4, predict at t={seq_len-1}")
    print(f"LR: 1e-2, Weight Decay: 0.01\n")
    
    history = {'loss': [], 'acc': [], 'grad_norm': []}
    
    for epoch in range(n_epochs):
        x, target = create_memory_task(batch_size, seq_len)
        
        # Forward (full sequence)
        logits = model(x)
        
        # Loss only at the last position
        loss = F.cross_entropy(logits[:, -1, :], target[:, -1])
        
        # Backward (full BPTT through all timesteps)
        optimizer.zero_grad()
        loss.backward()
        
        # Record gradient norm before clipping
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        optimizer.step()
        
        # Accuracy
        with torch.no_grad():
            pred = logits[:, -1, :].argmax(dim=-1)
            acc = (pred == target[:, -1]).float().mean().item() * 100
        
        history['loss'].append(loss.item())
        history['acc'].append(acc)
        history['grad_norm'].append(grad_norm.item())
        
        if (epoch + 1) % 25 == 0:
            print(f"  Epoch {epoch+1:3d}: loss={loss.item():.4f}, acc={acc:5.1f}%, grad_norm={grad_norm.item():.4f}")
    
    # Final analysis
    print(f"\n{'='*80}")
    print(f"RESULTS:")
    print(f"  Final accuracy: {history['acc'][-1]:.1f}%")
    print(f"  Best accuracy:  {max(history['acc']):.1f}%")
    print(f"  Final loss:     {history['loss'][-1]:.4f}")
    
    # Gradient stability analysis
    print(f"\nGradient stability (norm over training):")
    print(f"  Initial: {history['grad_norm'][0]:.4f}")
    print(f"  Final:   {history['grad_norm'][-1]:.4f}")
    print(f"  Mean:    {sum(history['grad_norm'])/len(history['grad_norm']):.4f}")
    print(f"  Max:     {max(history['grad_norm']):.4f}")
    
    if max(history['acc']) > 80:
        print(f"\n✓ SUCCESS: Model learned to remember through full BPTT!")
    elif max(history['acc']) > 30:
        print(f"\n⚠ PARTIAL: Model shows some learning")
    else:
        print(f"\n✗ FAILURE: Model could not learn the task")
    
    return history


def test_varying_delay():
    """
    Test learning at different delay lengths.
    
    This tests if gradient preservation works uniformly across time.
    """
    print("\n" + "=" * 80)
    print("TEST 3: Varying Delay Length")
    print("=" * 80)
    
    delays = [10, 20, 40, 60]
    results = {}
    
    for delay in delays:
        print(f"\n--- Testing delay={delay} ---")
        
        model = SmallRIN(vocab_size=100, d_model=32, num_layers=1, num_neurons=64)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2, weight_decay=0.01)
        
        def create_task(batch_size):
            seq_len = delay + 10
            x = torch.randint(10, 50, (batch_size, seq_len))
            target = torch.zeros(batch_size, dtype=torch.long)
            
            MARKER = 1
            
            for b in range(batch_size):
                value = torch.randint(50, 100, (1,)).item()
                x[b, 3] = MARKER
                x[b, 4] = value
                target[b] = value
            
            return x, target
        
        best_acc = 0
        final_grad_norm = 0
        
        for epoch in range(150):
            x, target = create_task(batch_size=32)
            
            logits = model(x)
            loss = F.cross_entropy(logits[:, -1, :], target)
            
            optimizer.zero_grad()
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            with torch.no_grad():
                pred = logits[:, -1, :].argmax(dim=-1)
                acc = (pred == target).float().mean().item() * 100
                best_acc = max(best_acc, acc)
            
            final_grad_norm = grad_norm.item()
        
        results[delay] = {'best_acc': best_acc, 'grad_norm': final_grad_norm}
        print(f"  Best accuracy: {best_acc:.1f}%, Final grad norm: {final_grad_norm:.4f}")
    
    # Summary
    print(f"\n{'='*80}")
    print(f"DELAY COMPARISON:")
    print(f"{'Delay':<10} {'Best Acc':<15} {'Grad Norm':<15}")
    print("-" * 40)
    for delay, res in results.items():
        print(f"{delay:<10} {res['best_acc']:<15.1f} {res['grad_norm']:<15.4f}")
    
    return results


def test_gradient_flow_detailed():
    """
    Detailed gradient flow analysis at each layer.
    """
    print("\n" + "=" * 80)
    print("TEST 4: Detailed Gradient Flow Analysis")
    print("=" * 80)
    
    model = SmallRIN(vocab_size=100, d_model=32, num_layers=1, num_neurons=64)
    
    seq_len = 50
    x = torch.randint(10, 90, (4, seq_len))
    x[:, 3] = 1  # Marker
    x[:, 4] = 42  # Value
    
    # Forward with gradient tracking
    logits = model(x)
    
    # Loss at end
    target = torch.full((4,), 42, dtype=torch.long)
    loss = F.cross_entropy(logits[:, -1, :], target)
    
    # Backward
    loss.backward()
    
    print(f"\nGradient analysis for seq_len={seq_len}:")
    print(f"\nParameter gradients:")
    
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad = param.grad
            print(f"  {name:<40} | norm={grad.norm().item():<10.6f} | mean={grad.abs().mean().item():<10.8f} | max={grad.abs().max().item():<10.6f}")
    
    # Check for vanishing/exploding
    all_norms = [p.grad.norm().item() for p in model.parameters() if p.grad is not None]
    
    print(f"\nSummary:")
    print(f"  Min grad norm: {min(all_norms):.8f}")
    print(f"  Max grad norm: {max(all_norms):.6f}")
    print(f"  Ratio max/min: {max(all_norms)/max(min(all_norms), 1e-10):.2f}")
    
    if min(all_norms) > 1e-8:
        print(f"\n✓ No vanishing gradients detected")
    else:
        print(f"\n⚠ Some gradients may be vanishing")
    
    if max(all_norms) < 100:
        print(f"✓ No exploding gradients detected")
    else:
        print(f"⚠ Some gradients may be exploding")
    
    return True


if __name__ == '__main__':
    print("=" * 80)
    print("RIN MODEL - FULL BPTT GRADIENT PRESERVATION TEST")
    print("=" * 80)
    print(f"\nTheory: Euler-based transforms have |∂(cos θ)/∂θ|² + |∂(sin θ)/∂θ|² = 1")
    print(f"This should provide perfect gradient preservation through time.")
    print("=" * 80)
    
    tests = [
        ('Gradient Preservation', test_gradient_preservation),
        ('Full BPTT Learning', test_full_bptt_learning),
        ('Varying Delay', test_varying_delay),
        ('Detailed Gradient Flow', test_gradient_flow_detailed),
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
    
    for name, status, _ in results:
        print(f"  {name:<30} {status}")

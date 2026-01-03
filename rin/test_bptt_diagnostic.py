"""
Deep diagnostic test for BPTT learning in RIN model.

The gradient preservation is working, but the model isn't learning.
This test analyzes:
1. State evolution through the sequence
2. How the "value" token affects state
3. Whether the state at the end contains information from the value
4. Gradient flow specifically to the value token's embedding
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


class DiagnosticRIN(nn.Module):
    """
    RIN model with diagnostic hooks to track state evolution.
    """
    
    def __init__(self, vocab_size=100, d_model=32, num_layers=1, num_neurons=32):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.d_model = d_model
        
        self.token_embedding = nn.Embedding(vocab_size, 2 * d_model)
        
        # Simpler: just Euler transform + linear output
        # No resonant layers to reduce complexity
        self.output_proj = nn.Linear(2 * d_model, vocab_size)
        
        self._lut = None
        self._init_weights()
    
    def _init_weights(self):
        nn.init.normal_(self.token_embedding.weight, std=0.1)  # Larger init
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
        
        h_real_new = cos_r * cos_i - sin_r * sin_i
        h_imag_new = cos_r * sin_i + sin_r * cos_i
        
        return h_real_new, h_imag_new
    
    def forward(self, input_ids, return_states=False):
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        
        h_real = torch.zeros(batch_size, self.d_model, device=device)
        h_imag = torch.zeros(batch_size, self.d_model, device=device)
        
        embeddings = self.token_embedding(input_ids)
        w_emb = embeddings[:, :, :self.d_model]
        b_emb = embeddings[:, :, self.d_model:]
        
        all_logits = []
        states_real = [h_real.clone()] if return_states else None
        states_imag = [h_imag.clone()] if return_states else None
        
        for t in range(seq_len):
            w_t = w_emb[:, t, :]
            b_t = b_emb[:, t, :]
            t_val = torch.tensor(t, dtype=torch.float32, device=device)
            
            h_real, h_imag = self.euler_transform(h_real, h_imag, w_t, b_t, t_val)
            
            if return_states:
                states_real.append(h_real.clone())
                states_imag.append(h_imag.clone())
            
            combined = torch.cat([h_real, h_imag], dim=-1)
            logits = self.output_proj(combined)
            all_logits.append(logits)
        
        result = torch.stack(all_logits, dim=1)
        
        if return_states:
            return result, states_real, states_imag
        return result


def test_state_evolution():
    """Trace how state evolves and if value token affects it."""
    print("=" * 80)
    print("TEST 1: State Evolution Analysis")
    print("=" * 80)
    
    model = DiagnosticRIN(vocab_size=100, d_model=32)
    model.eval()
    
    # Create sequence
    seq_len = 20
    x = torch.randint(10, 50, (1, seq_len))
    
    MARKER = 1
    VALUE = 42
    x[0, 3] = MARKER
    x[0, 4] = VALUE
    
    print(f"\nSequence: marker at t=3, value={VALUE} at t=4")
    print(f"Analyzing state evolution...\n")
    
    with torch.no_grad():
        logits, states_r, states_i = model(x, return_states=True)
    
    # Compute state magnitudes
    print(f"{'Step':<6} {'|h_real|':<12} {'|h_imag|':<12} {'|h|':<12} {'h.sum':<12}")
    print("-" * 54)
    
    for t in range(min(seq_len + 1, 15)):
        h_r = states_r[t][0]
        h_i = states_i[t][0]
        mag_r = h_r.norm().item()
        mag_i = h_i.norm().item()
        mag_total = (h_r**2 + h_i**2).sqrt().sum().item()
        h_sum = (h_r + h_i).sum().item()
        
        marker = ""
        if t == 4:
            marker = " <- MARKER"
        elif t == 5:
            marker = " <- VALUE"
        
        print(f"{t:<6} {mag_r:<12.4f} {mag_i:<12.4f} {mag_total:<12.4f} {h_sum:<12.4f}{marker}")
    
    # Check if state at VALUE position is different
    h_before_value = states_r[4][0]
    h_after_value = states_r[5][0]
    
    diff = (h_after_value - h_before_value).norm().item()
    print(f"\nState change at value position: {diff:.4f}")
    
    # Check final state vs state right after value
    h_final = states_r[-1][0]
    h_at_value = states_r[5][0]
    
    similarity = F.cosine_similarity(h_final.unsqueeze(0), h_at_value.unsqueeze(0)).item()
    print(f"Cosine similarity (final vs after-value state): {similarity:.4f}")
    
    return True


def test_gradient_to_value_token():
    """
    Check if gradient flows specifically to the VALUE token's embedding.
    """
    print("\n" + "=" * 80)
    print("TEST 2: Gradient Flow to Value Token")
    print("=" * 80)
    
    model = DiagnosticRIN(vocab_size=100, d_model=32)
    
    seq_len = 20
    x = torch.randint(10, 50, (4, seq_len))
    
    VALUES = [50, 60, 70, 80]  # Different values for each batch
    for b, v in enumerate(VALUES):
        x[b, 3] = 1  # Marker
        x[b, 4] = v  # Value
    
    # Target: predict the value at the end
    target = torch.tensor(VALUES, dtype=torch.long)
    
    # Forward
    logits = model(x)
    
    # Loss at the last position
    loss = F.cross_entropy(logits[:, -1, :], target)
    
    # Backward
    loss.backward()
    
    # Check gradient at embedding for specific tokens
    emb_grad = model.token_embedding.weight.grad
    
    print(f"\nEmbedding gradient analysis:")
    print(f"  Total embedding grad norm: {emb_grad.norm().item():.4f}")
    
    # Check gradient for VALUE tokens specifically
    for v in VALUES:
        grad_v = emb_grad[v]
        print(f"  Grad for token {v}: norm={grad_v.norm().item():.6f}")
    
    # Check gradient for random non-value tokens
    random_tokens = [15, 25, 35, 45]
    for rt in random_tokens:
        grad_rt = emb_grad[rt]
        print(f"  Grad for random token {rt}: norm={grad_rt.norm().item():.6f}")
    
    # Check if value tokens have stronger gradients
    value_grad_mean = sum(emb_grad[v].norm().item() for v in VALUES) / len(VALUES)
    random_grad_mean = sum(emb_grad[rt].norm().item() for rt in random_tokens) / len(random_tokens)
    
    print(f"\n  Mean value token grad: {value_grad_mean:.6f}")
    print(f"  Mean random token grad: {random_grad_mean:.6f}")
    print(f"  Ratio (value/random): {value_grad_mean/max(random_grad_mean, 1e-10):.2f}")
    
    if value_grad_mean > random_grad_mean * 1.5:
        print(f"\n✓ Gradients ARE flowing preferentially to value tokens")
    else:
        print(f"\n⚠ Gradients NOT preferentially flowing to value tokens")
    
    return True


def test_minimal_memory():
    """
    Simplest possible memory test: remember token from t=1, predict at t=2.
    """
    print("\n" + "=" * 80)
    print("TEST 3: Minimal Memory Test (delay=1)")
    print("=" * 80)
    
    model = DiagnosticRIN(vocab_size=100, d_model=32)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2, weight_decay=0.01)
    
    def create_simple_task(batch_size):
        # Sequence: [MARKER, VALUE, ?] -> predict VALUE
        x = torch.zeros(batch_size, 3, dtype=torch.long)
        target = torch.zeros(batch_size, dtype=torch.long)
        
        for b in range(batch_size):
            value = torch.randint(50, 100, (1,)).item()
            x[b, 0] = 1  # Marker
            x[b, 1] = value
            x[b, 2] = 2  # Query
            target[b] = value
        
        return x, target
    
    print(f"\nTask: [MARKER, VALUE, QUERY] -> predict VALUE at position 2")
    print(f"This is the minimal memory task (1 step delay)\n")
    
    for epoch in range(100):
        x, target = create_simple_task(batch_size=32)
        
        logits = model(x)
        loss = F.cross_entropy(logits[:, 2, :], target)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        with torch.no_grad():
            pred = logits[:, 2, :].argmax(dim=-1)
            acc = (pred == target).float().mean().item() * 100
        
        if (epoch + 1) % 20 == 0:
            print(f"  Epoch {epoch+1}: loss={loss.item():.4f}, acc={acc:.1f}%")
    
    if acc > 80:
        print(f"\n✓ Model CAN learn minimal memory task")
    else:
        print(f"\n✗ Model CANNOT even learn minimal memory task")
    
    return acc


def test_increasing_delay():
    """
    Test with increasing delay to find the limit.
    """
    print("\n" + "=" * 80)
    print("TEST 4: Finding Memory Limit (Increasing Delay)")
    print("=" * 80)
    
    delays = [1, 2, 5, 10, 15, 20]
    results = {}
    
    for delay in delays:
        model = DiagnosticRIN(vocab_size=100, d_model=32)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2, weight_decay=0.01)
        
        def create_task(batch_size):
            seq_len = delay + 3
            x = torch.randint(10, 50, (batch_size, seq_len))
            target = torch.zeros(batch_size, dtype=torch.long)
            
            for b in range(batch_size):
                value = torch.randint(50, 100, (1,)).item()
                x[b, 0] = 1  # Marker
                x[b, 1] = value
                x[b, -1] = 2  # Query at end
                target[b] = value
            
            return x, target
        
        best_acc = 0
        for epoch in range(150):
            x, target = create_task(batch_size=32)
            
            logits = model(x)
            loss = F.cross_entropy(logits[:, -1, :], target)
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            with torch.no_grad():
                pred = logits[:, -1, :].argmax(dim=-1)
                acc = (pred == target).float().mean().item() * 100
                best_acc = max(best_acc, acc)
        
        results[delay] = best_acc
        print(f"  Delay {delay:2d}: Best accuracy = {best_acc:.1f}%")
    
    print(f"\n{'='*80}")
    print(f"MEMORY LIMIT ANALYSIS:")
    for delay, acc in results.items():
        status = "✓" if acc > 50 else "~" if acc > 20 else "✗"
        print(f"  {status} Delay {delay}: {acc:.1f}%")
    
    return results


def test_pure_euler_chain():
    """
    Test a PURE Euler chain with NO processing layers.
    Just: embed -> euler transform chain -> output
    
    This tests if the fundamental Euler transform preserves information.
    """
    print("\n" + "=" * 80)
    print("TEST 5: Pure Euler Transform Chain (No Layers)")
    print("=" * 80)
    
    class PureEulerModel(nn.Module):
        def __init__(self, vocab_size=100, d_model=32):
            super().__init__()
            self.d_model = d_model
            self.token_embedding = nn.Embedding(vocab_size, 2 * d_model)
            self.output_proj = nn.Linear(2 * d_model, vocab_size)
            self._lut = None
            
            nn.init.normal_(self.token_embedding.weight, std=0.1)
        
        def _get_lut(self, device):
            if self._lut is None or self._lut.sin_table.device != device:
                self._lut = get_global_lut(4096, device)
            return self._lut
        
        def forward(self, input_ids):
            batch_size, seq_len = input_ids.shape
            device = input_ids.device
            lut = self._get_lut(device)
            
            h_real = torch.zeros(batch_size, self.d_model, device=device)
            h_imag = torch.zeros(batch_size, self.d_model, device=device)
            
            embeddings = self.token_embedding(input_ids)
            w_emb = embeddings[:, :, :self.d_model]
            b_emb = embeddings[:, :, self.d_model:]
            
            all_logits = []
            
            for t in range(seq_len):
                w_t = w_emb[:, t, :]
                b_t = b_emb[:, t, :]
                
                wavelength = 1.0 + w_t.abs()
                t_phi = t * PHI
                
                theta_real = h_real / wavelength + b_t + t_phi
                theta_imag = h_imag / wavelength + b_t + t_phi
                
                sin_r, cos_r = lut.lookup_sin_cos(theta_real)
                sin_i, cos_i = lut.lookup_sin_cos(theta_imag)
                
                h_real = cos_r * cos_i - sin_r * sin_i
                h_imag = cos_r * sin_i + sin_r * cos_i
                
                combined = torch.cat([h_real, h_imag], dim=-1)
                logits = self.output_proj(combined)
                all_logits.append(logits)
            
            return torch.stack(all_logits, dim=1)
    
    model = PureEulerModel(vocab_size=100, d_model=32)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2, weight_decay=0.01)
    
    print(f"\nPure Euler model (no resonant layers)")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    def create_task(batch_size, delay=10):
        seq_len = delay + 3
        x = torch.randint(10, 50, (batch_size, seq_len))
        target = torch.zeros(batch_size, dtype=torch.long)
        
        for b in range(batch_size):
            value = torch.randint(50, 100, (1,)).item()
            x[b, 0] = 1
            x[b, 1] = value
            target[b] = value
        
        return x, target
    
    print(f"\nTraining on delay=10 task...")
    
    best_acc = 0
    for epoch in range(200):
        x, target = create_task(batch_size=32, delay=10)
        
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
        
        if (epoch + 1) % 40 == 0:
            print(f"  Epoch {epoch+1}: loss={loss.item():.4f}, acc={acc:.1f}%, grad_norm={grad_norm.item():.4f}")
    
    print(f"\nBest accuracy: {best_acc:.1f}%")
    
    if best_acc > 50:
        print(f"\n✓ Pure Euler chain CAN learn memory task")
    else:
        print(f"\n✗ Pure Euler chain cannot learn memory task")
        print(f"  → The issue is NOT gradient preservation, but information encoding")
    
    return best_acc


if __name__ == '__main__':
    print("=" * 80)
    print("DEEP DIAGNOSTIC: WHY ISN'T BPTT LEARNING WORKING?")
    print("=" * 80)
    
    tests = [
        ('State Evolution', test_state_evolution),
        ('Gradient to Value Token', test_gradient_to_value_token),
        ('Minimal Memory (delay=1)', test_minimal_memory),
        ('Increasing Delay', test_increasing_delay),
        ('Pure Euler Chain', test_pure_euler_chain),
    ]
    
    for name, test_fn in tests:
        try:
            test_fn()
        except Exception as e:
            print(f"\n{name} FAILED: {e}")
            import traceback
            traceback.print_exc()

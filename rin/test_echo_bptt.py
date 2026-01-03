"""
Test if Echo Chamber enables long-term memory in BPTT training.

The pure Euler model loses information after ~3 steps because each
token's embedding rotates the state, scrambling previous information.

The Echo Chamber should:
1. Selectively write important information to memory
2. Preserve it through subsequent steps
3. Allow retrieval when needed

This test uses full BPTT to train the model end-to-end.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import sys
sys.path.insert(0, '/home/aiman/pi')

from rin.model import PHI
from rin.lut import get_global_lut
from rin.echo_chamber_v2 import EchoChamberV2

torch.manual_seed(42)


class EulerWithEchoModel(nn.Module):
    """
    Euler transform model WITH Echo Chamber for explicit memory.
    
    The Echo Chamber provides a separate memory pathway that:
    - Writes when interference is high (selective attention)
    - Maintains state with learned decay (persistence)
    - Adds to output (memory retrieval)
    """
    
    def __init__(self, vocab_size=100, d_model=32, n_echo_heads=4):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.d_model = d_model
        
        self.token_embedding = nn.Embedding(vocab_size, 2 * d_model)
        
        # Echo Chamber for explicit memory
        self.echo = EchoChamberV2(d_model, n_heads=n_echo_heads)
        
        # Learnable scale for echo contribution
        self.echo_scale = nn.Parameter(torch.tensor(0.5))
        
        # Output projection
        self.output_proj = nn.Linear(2 * d_model, vocab_size)
        
        self._lut = None
        self._init_weights()
    
    def _init_weights(self):
        nn.init.normal_(self.token_embedding.weight, std=0.1)
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
    
    def forward(self, input_ids, return_diagnostics=False):
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        
        # Reset echo chamber memory
        self.echo.reset_memory(batch_size, device)
        
        h_real = torch.zeros(batch_size, self.d_model, device=device)
        h_imag = torch.zeros(batch_size, self.d_model, device=device)
        
        embeddings = self.token_embedding(input_ids)
        w_emb = embeddings[:, :, :self.d_model]
        b_emb = embeddings[:, :, self.d_model:]
        
        all_logits = []
        all_diags = [] if return_diagnostics else None
        
        for t in range(seq_len):
            w_t = w_emb[:, t, :]
            b_t = b_emb[:, t, :]
            t_val = torch.tensor(t, dtype=torch.float32, device=device)
            
            # Euler transform for instant state
            h_real, h_imag = self.euler_transform(h_real, h_imag, w_t, b_t, t_val)
            
            # Echo Chamber for long-term memory
            echo_real, echo_imag, diag = self.echo(h_real, h_imag, t_val)
            
            if return_diagnostics:
                all_diags.append(diag)
            
            # Combine instant state with echo memory
            out_real = h_real + self.echo_scale * echo_real
            out_imag = h_imag + self.echo_scale * echo_imag
            
            combined = torch.cat([out_real, out_imag], dim=-1)
            logits = self.output_proj(combined)
            all_logits.append(logits)
        
        result = torch.stack(all_logits, dim=1)
        
        if return_diagnostics:
            return result, all_diags
        return result


def test_echo_memory_bptt():
    """
    Test if Echo Chamber enables memory beyond the Euler transform's ~3 step limit.
    """
    print("=" * 80)
    print("TEST 1: Echo Chamber Memory with BPTT")
    print("=" * 80)
    
    delays = [2, 5, 10, 15, 20]
    
    for delay in delays:
        print(f"\n--- Testing delay={delay} ---")
        
        model = EulerWithEchoModel(vocab_size=100, d_model=32, n_echo_heads=4)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2, weight_decay=0.01)
        
        print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        def create_task(batch_size):
            seq_len = delay + 3
            x = torch.randint(10, 50, (batch_size, seq_len))
            target = torch.zeros(batch_size, dtype=torch.long)
            
            for b in range(batch_size):
                value = torch.randint(50, 100, (1,)).item()
                x[b, 0] = 1  # Marker
                x[b, 1] = value
                target[b] = value
            
            return x, target
        
        best_acc = 0
        for epoch in range(200):
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
            
            if (epoch + 1) % 50 == 0:
                # Get echo diagnostics
                logits, diags = model(x, return_diagnostics=True)
                decay = diags[-1]['decay_mean'].item()
                int_mag = diags[-1]['total_int_mag'].mean().item()
                
                print(f"    Epoch {epoch+1}: loss={loss.item():.4f}, acc={acc:.1f}%, "
                      f"decay={decay:.4f}, int_mag={int_mag:.4f}")
        
        status = "✓" if best_acc > 50 else "~" if best_acc > 20 else "✗"
        print(f"  {status} Delay {delay}: Best accuracy = {best_acc:.1f}%")


def test_echo_vs_pure_euler():
    """
    Direct comparison: Pure Euler vs Euler + Echo on same task.
    """
    print("\n" + "=" * 80)
    print("TEST 2: Pure Euler vs Euler+Echo Comparison")
    print("=" * 80)
    
    delay = 10
    n_epochs = 200
    
    # Pure Euler (baseline - we know this fails)
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
                wavelength = 1.0 + w_emb[:, t, :].abs()
                t_phi = t * PHI
                
                theta_real = h_real / wavelength + b_emb[:, t, :] + t_phi
                theta_imag = h_imag / wavelength + b_emb[:, t, :] + t_phi
                
                sin_r, cos_r = lut.lookup_sin_cos(theta_real)
                sin_i, cos_i = lut.lookup_sin_cos(theta_imag)
                
                h_real = cos_r * cos_i - sin_r * sin_i
                h_imag = cos_r * sin_i + sin_r * cos_i
                
                combined = torch.cat([h_real, h_imag], dim=-1)
                logits = self.output_proj(combined)
                all_logits.append(logits)
            
            return torch.stack(all_logits, dim=1)
    
    def create_task(batch_size):
        seq_len = delay + 3
        x = torch.randint(10, 50, (batch_size, seq_len))
        target = torch.zeros(batch_size, dtype=torch.long)
        
        for b in range(batch_size):
            value = torch.randint(50, 100, (1,)).item()
            x[b, 0] = 1
            x[b, 1] = value
            target[b] = value
        
        return x, target
    
    print(f"\nTask: Remember value from t=1, predict at t={delay+2}")
    print(f"Training both models for {n_epochs} epochs...\n")
    
    # Train Pure Euler
    print("Pure Euler Model:")
    model_pure = PureEulerModel(vocab_size=100, d_model=32)
    opt_pure = torch.optim.AdamW(model_pure.parameters(), lr=1e-2, weight_decay=0.01)
    
    best_pure = 0
    for epoch in range(n_epochs):
        x, target = create_task(batch_size=32)
        logits = model_pure(x)
        loss = F.cross_entropy(logits[:, -1, :], target)
        opt_pure.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model_pure.parameters(), 1.0)
        opt_pure.step()
        
        with torch.no_grad():
            pred = logits[:, -1, :].argmax(dim=-1)
            acc = (pred == target).float().mean().item() * 100
            best_pure = max(best_pure, acc)
        
        if (epoch + 1) % 50 == 0:
            print(f"  Epoch {epoch+1}: loss={loss.item():.4f}, acc={acc:.1f}%")
    
    print(f"  Best: {best_pure:.1f}%")
    
    # Train Euler + Echo
    print("\nEuler + Echo Model:")
    model_echo = EulerWithEchoModel(vocab_size=100, d_model=32, n_echo_heads=4)
    opt_echo = torch.optim.AdamW(model_echo.parameters(), lr=1e-2, weight_decay=0.01)
    
    best_echo = 0
    for epoch in range(n_epochs):
        x, target = create_task(batch_size=32)
        logits = model_echo(x)
        loss = F.cross_entropy(logits[:, -1, :], target)
        opt_echo.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model_echo.parameters(), 1.0)
        opt_echo.step()
        
        with torch.no_grad():
            pred = logits[:, -1, :].argmax(dim=-1)
            acc = (pred == target).float().mean().item() * 100
            best_echo = max(best_echo, acc)
        
        if (epoch + 1) % 50 == 0:
            # Get diagnostics
            logits, diags = model_echo(x, return_diagnostics=True)
            decay = diags[-1]['decay_mean'].item()
            mem_mag = diags[-1]['memory_mag'].mean().item()
            print(f"  Epoch {epoch+1}: loss={loss.item():.4f}, acc={acc:.1f}%, decay={decay:.4f}, mem={mem_mag:.4f}")
    
    print(f"  Best: {best_echo:.1f}%")
    
    print(f"\n{'='*80}")
    print(f"COMPARISON (delay={delay}):")
    print(f"  Pure Euler: {best_pure:.1f}%")
    print(f"  Euler+Echo: {best_echo:.1f}%")
    
    if best_echo > best_pure + 10:
        print(f"\n✓ Echo Chamber SIGNIFICANTLY helps memory!")
    elif best_echo > best_pure:
        print(f"\n~ Echo Chamber helps somewhat")
    else:
        print(f"\n✗ Echo Chamber does NOT help")


def test_echo_diagnostics():
    """
    Detailed analysis of Echo Chamber behavior during training.
    """
    print("\n" + "=" * 80)
    print("TEST 3: Echo Chamber Behavior Analysis")
    print("=" * 80)
    
    model = EulerWithEchoModel(vocab_size=100, d_model=32, n_echo_heads=4)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2, weight_decay=0.01)
    
    delay = 10
    
    def create_task(batch_size):
        seq_len = delay + 3
        x = torch.randint(10, 50, (batch_size, seq_len))
        target = torch.zeros(batch_size, dtype=torch.long)
        
        for b in range(batch_size):
            value = torch.randint(50, 100, (1,)).item()
            x[b, 0] = 1
            x[b, 1] = value
            target[b] = value
        
        return x, target
    
    print(f"\nTracking Echo Chamber metrics during training...")
    print(f"{'Epoch':<8} {'Loss':<10} {'Acc':<8} {'Decay':<10} {'Int_Mag':<10} {'Mem_Mag':<10} {'β_eff':<10}")
    print("-" * 66)
    
    for epoch in range(150):
        x, target = create_task(batch_size=32)
        
        logits, diags = model(x, return_diagnostics=True)
        loss = F.cross_entropy(logits[:, -1, :], target)
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        with torch.no_grad():
            pred = logits[:, -1, :].argmax(dim=-1)
            acc = (pred == target).float().mean().item() * 100
        
        if (epoch + 1) % 30 == 0:
            d = diags[-1]
            decay = d['decay_mean'].item()
            int_mag = d['total_int_mag'].mean().item()
            mem_mag = d['memory_mag'].mean().item()
            beta_eff = d['beta_eff_mean'].item()
            
            print(f"{epoch+1:<8} {loss.item():<10.4f} {acc:<8.1f} {decay:<10.4f} {int_mag:<10.4f} {mem_mag:<10.4f} {beta_eff:<10.4f}")
    
    # Final detailed per-timestep analysis
    print(f"\n\nPer-timestep analysis (final model):")
    x, target = create_task(batch_size=4)
    
    with torch.no_grad():
        logits, diags = model(x, return_diagnostics=True)
    
    print(f"{'t':<4} {'Int_Mag':<10} {'Mem_Mag':<10} {'Write':<10}")
    print("-" * 34)
    
    for t, d in enumerate(diags[:min(15, len(diags))]):
        int_mag = d['total_int_mag'].mean().item()
        mem_mag = d['memory_mag'].mean().item()
        write = d.get('write_scale_mean', torch.tensor(0)).item() if isinstance(d.get('write_scale_mean'), torch.Tensor) else 0
        
        marker = ""
        if t == 0:
            marker = " <- MARKER"
        elif t == 1:
            marker = " <- VALUE"
        elif t == len(diags) - 1:
            marker = " <- PREDICT"
        
        print(f"{t:<4} {int_mag:<10.4f} {mem_mag:<10.4f} {write:<10.4f}{marker}")


if __name__ == '__main__':
    print("=" * 80)
    print("ECHO CHAMBER WITH FULL BPTT TRAINING")
    print("=" * 80)
    
    tests = [
        ('Echo Memory BPTT', test_echo_memory_bptt),
        ('Echo vs Pure Euler', test_echo_vs_pure_euler),
        ('Echo Diagnostics', test_echo_diagnostics),
    ]
    
    for name, test_fn in tests:
        try:
            test_fn()
        except Exception as e:
            print(f"\n{name} FAILED: {e}")
            import traceback
            traceback.print_exc()

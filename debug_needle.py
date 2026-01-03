#!/usr/bin/env python3
"""Debug why the needle task isn't learning."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
sys.path.insert(0, '/home/aiman/pi')

from rin import PHI, get_global_lut
from rin.utils import wrap_time_periodic


class SimpleNeedleRIN(nn.Module):
    """Simplified RIN to debug learning."""
    
    def __init__(self, vocab_size, num_signals, d_model=64):
        super().__init__()
        self.vocab_size = vocab_size
        self.num_signals = num_signals
        self.d_model = d_model
        
        self.token_embedding = nn.Embedding(vocab_size, 2 * d_model)
        self.output_proj = nn.Linear(d_model * 2, num_signals)  # Use both real and imag
        
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        
        self._lut = None
    
    def _get_lut(self, device):
        if self._lut is None or self._lut.sin_table.device != device:
            self._lut = get_global_lut(4096, device)
        return self._lut
    
    def euler_transform(self, h_real, h_imag, w, b, t):
        lut = self._get_lut(h_real.device)
        
        wavelength = 1.0 + w.abs()
        t_phi = t.unsqueeze(-1) * PHI if t.dim() == 1 else t * PHI
        t_phi = wrap_time_periodic(t_phi)
        
        theta_real = h_real / wavelength + b + t_phi
        theta_imag = h_imag / wavelength + b + t_phi
        
        sin_real, cos_real = lut.lookup_sin_cos(theta_real)
        sin_imag, cos_imag = lut.lookup_sin_cos(theta_imag)
        
        h_real_new = cos_real * cos_imag - sin_real * sin_imag
        h_imag_new = cos_real * sin_imag + sin_real * cos_imag
        
        return h_real_new, h_imag_new
    
    def forward(self, input_ids):
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        
        h_real = torch.zeros(batch_size, self.d_model, device=device)
        h_imag = torch.zeros(batch_size, self.d_model, device=device)
        
        embeddings = self.token_embedding(input_ids)
        w_emb = embeddings[:, :, :self.d_model]
        b_emb = embeddings[:, :, self.d_model:]
        
        t_indices = torch.arange(seq_len, device=device, dtype=torch.float32)
        
        for t in range(seq_len):
            t_val = t_indices[t].expand(batch_size)
            h_real, h_imag = self.euler_transform(h_real, h_imag, w_emb[:, t], b_emb[:, t], t_val)
        
        # Concat both real and imag for output
        h_combined = torch.cat([h_real, h_imag], dim=-1)
        return self.output_proj(h_combined)


def test_gradient_flow():
    """Test if gradients flow through the model."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = SimpleNeedleRIN(vocab_size=61, num_signals=10, d_model=64).to(device)
    
    # Simple input
    input_ids = torch.tensor([[0, 5, 10, 15, 0]], device=device)  # [trigger, signal, noise, noise, trigger]
    target = torch.tensor([4], device=device)  # Signal was 5 - 1 = 4
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    
    print("Testing gradient flow...")
    print(f"Input shape: {input_ids.shape}")
    
    for step in range(100):
        optimizer.zero_grad()
        logits = model(input_ids)
        loss = F.cross_entropy(logits, target)
        loss.backward()
        
        # Check gradients
        if step == 0:
            print(f"\nStep 0:")
            print(f"  Loss: {loss.item():.4f}")
            print(f"  Logits: {logits[0].detach().cpu().numpy()}")
            print(f"  Embedding grad norm: {model.token_embedding.weight.grad.norm():.6f}")
            print(f"  Output proj grad norm: {model.output_proj.weight.grad.norm():.6f}")
        
        optimizer.step()
    
    print(f"\nAfter 100 steps:")
    logits = model(input_ids)
    print(f"  Loss: {F.cross_entropy(logits, target).item():.4f}")
    print(f"  Prediction: {logits.argmax(-1).item()} (target: {target.item()})")
    print(f"  Logits: {logits[0].detach().cpu().numpy()}")


def test_baseline():
    """Test with a simple MLP baseline."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    class MLPBaseline(nn.Module):
        def __init__(self, vocab_size, num_signals, d_model=64):
            super().__init__()
            self.embed = nn.Embedding(vocab_size, d_model)
            self.fc1 = nn.Linear(d_model, d_model)
            self.fc2 = nn.Linear(d_model, num_signals)
        
        def forward(self, x):
            # Sum embeddings along sequence
            h = self.embed(x).sum(dim=1)
            h = F.relu(self.fc1(h))
            return self.fc2(h)
    
    model = MLPBaseline(vocab_size=61, num_signals=10, d_model=64).to(device)
    
    input_ids = torch.tensor([[0, 5, 10, 15, 0]], device=device)
    target = torch.tensor([4], device=device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    
    print("\nMLP Baseline test...")
    for step in range(100):
        optimizer.zero_grad()
        logits = model(input_ids)
        loss = F.cross_entropy(logits, target)
        loss.backward()
        optimizer.step()
    
    logits = model(input_ids)
    print(f"  Final loss: {F.cross_entropy(logits, target).item():.4f}")
    print(f"  Prediction: {logits.argmax(-1).item()} (target: {target.item()})")


def test_signal_preservation():
    """Test if the signal token's information is preserved through the sequence."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = SimpleNeedleRIN(vocab_size=61, num_signals=10, d_model=64).to(device)
    
    # Two sequences with same noise but different signals
    seq1 = torch.tensor([[0, 3, 10, 15, 20, 0]], device=device)  # signal=3
    seq2 = torch.tensor([[0, 7, 10, 15, 20, 0]], device=device)  # signal=7
    
    with torch.no_grad():
        logits1 = model(seq1)
        logits2 = model(seq2)
        
        print("\nSignal preservation test:")
        print(f"  Seq1 (signal=3) logits std: {logits1.std().item():.4f}")
        print(f"  Seq2 (signal=7) logits std: {logits2.std().item():.4f}")
        print(f"  Difference in logits: {(logits1 - logits2).abs().mean().item():.4f}")
        
        # If signal info is preserved, outputs should differ
        if (logits1 - logits2).abs().mean() < 0.01:
            print("  ⚠️ Signal information is NOT being preserved!")
        else:
            print("  ✓ Signal information IS being preserved")


if __name__ == "__main__":
    test_gradient_flow()
    test_baseline()
    test_signal_preservation()

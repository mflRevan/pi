"""Detailed analysis of Echo Chamber V2 learning dynamics."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import time

import sys
sys.path.insert(0, '/home/aiman/pi')

from rin.echo_chamber_v2 import EchoChamberModelV2

def main():
    print("="*80)
    print("ECHO CHAMBER V2 - DETAILED LEARNING ANALYSIS")
    print("="*80)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    vocab_size = 64
    marker = vocab_size - 1
    seq_len = 16
    batch_size = 32
    
    model = EchoChamberModelV2(
        vocab_size=vocab_size,
        d_model=64,
        num_layers=1,
        num_neurons=64,
        n_echo_heads=4,
    ).to(device)
    
    print(f"Parameters: {model.get_num_params():,}")
    print(f"d_head = 16, Expected interference range: ~[-4, +4] (|trigger|~2, |query|~1)")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.1)
    
    # Fixed test sequence for tracking
    test_seq = torch.randint(0, vocab_size-2, (1, seq_len), device=device)
    test_seq[0, 4] = marker  # marker position
    test_seq[0, 5] = 42  # value to remember
    test_seq[0, -2] = marker  # query position
    test_target = torch.tensor([42], device=device)
    
    print(f"\nTest sequence: marker at t=4, value=42 at t=5, query marker at t=14")
    
    def analyze(epoch_str):
        model.eval()
        with torch.no_grad():
            logits, diag = model(test_seq, return_diagnostics=True)
            pred = logits[:, -1, :].argmax(dim=-1).item()
            
        # Get per-timestep interference
        int_reals = [d['layer0_echo']['total_int_real'].item() for d in diag]
        int_mags = [d['layer0_echo']['total_int_mag'].item() for d in diag]
        mem_mags = [d['layer0_echo']['memory_mag'].item() for d in diag]
        
        decay = diag[0]['layer0_echo']['decay_mean'].item()
        beta = model.echo_chambers[0].beta.item()
        
        print(f"\n{epoch_str}: pred={pred}, β={beta:.4f}, decay={decay:.4f}")
        print(f"  t |  int_real |  int_mag |   mem_mag | Notes")
        print(f"  --|-----------|----------|-----------|------")
        
        for t in range(seq_len):
            note = ""
            if t == 4:
                note = "← MARKER"
            elif t == 5:
                note = "← VALUE (42)"
            elif t == 14:
                note = "← QUERY MARKER"
            elif t == 15:
                note = "← PREDICT"
            
            print(f" {t:2d} | {int_reals[t]:9.4f} | {int_mags[t]:8.4f} | {mem_mags[t]:9.4f} | {note}")
        
        # Key positions summary
        print(f"\n  Key positions:")
        print(f"    Marker (t=4):  int={int_reals[4]:+.4f}, mem={mem_mags[4]:.4f}")
        print(f"    Value (t=5):   int={int_reals[5]:+.4f}, mem={mem_mags[5]:.4f}")
        print(f"    Query (t=14):  int={int_reals[14]:+.4f}, mem={mem_mags[14]:.4f}")
        print(f"    Final (t=15):  int={int_reals[15]:+.4f}, mem={mem_mags[15]:.4f}")
    
    # Before training
    analyze("BEFORE TRAINING")
    
    # Train
    best_acc = 0.0
    for epoch in range(200):
        model.train()
        
        seq = torch.randint(0, vocab_size-2, (batch_size, seq_len), device=device)
        targets = torch.randint(0, vocab_size-2, (batch_size,), device=device)
        
        for i in range(batch_size):
            pos = torch.randint(2, seq_len//2, (1,)).item()
            seq[i, pos] = marker
            seq[i, pos+1] = targets[i]
        
        seq[:, -2] = marker
        
        logits = model(seq)
        loss = F.cross_entropy(logits[:, -1, :], targets)
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        if epoch % 50 == 49:
            model.eval()
            with torch.no_grad():
                pred = logits[:, -1, :].argmax(dim=-1)
                acc = (pred == targets).float().mean().item()
                best_acc = max(best_acc, acc)
            
            analyze(f"EPOCH {epoch+1} (loss={loss.item():.4f}, acc={acc:.1%})")
    
    print(f"\n" + "="*80)
    print(f"FINAL: Best accuracy = {best_acc:.1%}")
    print("="*80)

if __name__ == "__main__":
    main()

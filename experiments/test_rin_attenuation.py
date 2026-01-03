"""
Test the RIN model with attenuation fix
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import sys
sys.path.insert(0, '/home/aiman/pi')

from rin.model import RINModel


def test_gradient_flow():
    """Test gradient flow through attenuation weights."""
    print("="*60)
    print("RIN MODEL - GRADIENT FLOW TEST")
    print("="*60)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    model = RINModel(
        vocab_size=64,
        d_model=64,
        num_layers=2,
        num_neurons=128,
    ).to(device)
    
    print(f"Parameters: {model.get_num_params():,}")
    
    x = torch.randint(0, 64, (4, 8), device=device)
    logits, _ = model(x)
    loss = logits.sum()
    loss.backward()
    
    print("\nKey gradients:")
    for name, param in model.named_parameters():
        if param.grad is None:
            print(f"  ⚠️  NO GRAD: {name}")
            continue
        
        if 'attn' in name or 'W' in name or 'B' in name:
            grad_norm = param.grad.norm().item()
            print(f"  {name}: {grad_norm:.6f}")
    
    print("\n✓ Gradients flowing!")


def test_retrieval_task():
    """Test marker-based retrieval task with weight decay."""
    print("\n" + "="*60)
    print("RIN MODEL - RETRIEVAL TASK (with weight decay)")
    print("="*60)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    vocab_size = 64
    marker = vocab_size - 1
    seq_len = 16
    batch_size = 32
    num_epochs = 500
    
    model = RINModel(
        vocab_size=vocab_size,
        d_model=64,
        num_layers=2,
        num_neurons=128,
    ).to(device)
    
    print(f"Parameters: {model.get_num_params():,}")
    
    # WITH WEIGHT DECAY (crucial!)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.1)
    
    start = time.time()
    
    for epoch in range(num_epochs):
        model.train()
        
        # Generate batch
        seq = torch.randint(0, vocab_size-2, (batch_size, seq_len), device=device)
        targets = torch.randint(0, vocab_size-2, (batch_size,), device=device)
        
        for i in range(batch_size):
            pos = torch.randint(2, seq_len//2, (1,)).item()
            seq[i, pos] = marker
            seq[i, pos+1] = targets[i]
        
        seq[:, -2] = marker
        
        logits, _ = model(seq)
        loss = F.cross_entropy(logits[:, -1, :], targets)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if epoch % 100 == 99:
            model.eval()
            with torch.no_grad():
                pred = logits[:, -1, :].argmax(dim=-1)
                acc = (pred == targets).float().mean().item()
            print(f"Epoch {epoch+1}: loss={loss.item():.4f}, acc={acc:.1%}")
    
    elapsed = time.time() - start
    print(f"\nTime: {elapsed:.1f}s")
    
    # Final eval
    model.eval()
    with torch.no_grad():
        test_seq = torch.randint(0, vocab_size-2, (batch_size, seq_len), device=device)
        test_targets = torch.randint(0, vocab_size-2, (batch_size,), device=device)
        
        for i in range(batch_size):
            pos = torch.randint(2, seq_len//2, (1,)).item()
            test_seq[i, pos] = marker
            test_seq[i, pos+1] = test_targets[i]
        
        test_seq[:, -2] = marker
        
        test_logits, _ = model(test_seq)
        test_pred = test_logits[:, -1, :].argmax(dim=-1)
        test_acc = (test_pred == test_targets).float().mean().item()
    
    print(f"Final test acc: {test_acc:.1%}")


def compare_with_vs_without_weight_decay():
    """Compare performance with and without weight decay."""
    print("\n" + "="*60)
    print("WEIGHT DECAY COMPARISON")
    print("="*60)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    vocab_size = 64
    marker = vocab_size - 1
    seq_len = 16
    batch_size = 32
    num_epochs = 300
    
    results = {}
    
    for wd in [0.0, 0.01, 0.1]:
        print(f"\n--- Weight Decay: {wd} ---")
        
        model = RINModel(
            vocab_size=vocab_size,
            d_model=64,
            num_layers=2,
            num_neurons=128,
        ).to(device)
        
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=wd)
        
        start = time.time()
        final_acc = 0.0
        
        for epoch in range(num_epochs):
            model.train()
            
            seq = torch.randint(0, vocab_size-2, (batch_size, seq_len), device=device)
            targets = torch.randint(0, vocab_size-2, (batch_size,), device=device)
            
            for i in range(batch_size):
                pos = torch.randint(2, seq_len//2, (1,)).item()
                seq[i, pos] = marker
                seq[i, pos+1] = targets[i]
            
            seq[:, -2] = marker
            
            logits, _ = model(seq)
            loss = F.cross_entropy(logits[:, -1, :], targets)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if epoch % 100 == 99:
                model.eval()
                with torch.no_grad():
                    pred = logits[:, -1, :].argmax(dim=-1)
                    final_acc = (pred == targets).float().mean().item()
                print(f"  Epoch {epoch+1}: loss={loss.item():.4f}, acc={final_acc:.1%}")
        
        elapsed = time.time() - start
        results[wd] = {'acc': final_acc, 'time': elapsed}
        print(f"  Time: {elapsed:.1f}s")
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"{'Weight Decay':<15} {'Accuracy':<12}")
    print("-"*30)
    for wd, r in results.items():
        print(f"{wd:<15} {r['acc']:.1%}")


if __name__ == "__main__":
    test_gradient_flow()
    test_retrieval_task()
    compare_with_vs_without_weight_decay()

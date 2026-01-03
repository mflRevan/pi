"""
Comprehensive Model Comparison - With Weight Decay

Tests:
1. SwiGLU Transformer (baseline)
2. Echo V2 (parallel Euler)
3. RIN (recurrent, now with attenuation)
4. Episodic Echo V2 (optimized)

All with weight_decay=0.1 and same hyperparameters.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import sys
sys.path.insert(0, '/home/aiman/pi')

from rin.model import RINModel
from rin.transformer import SwiGLUTransformer
from rin.echo_v2 import EchoModelV2
from rin.episodic_echo_v2 import EpisodicEchoModelV2


def test_marker_retrieval(model, name, device, vocab_size=64, seq_len=16, 
                          batch_size=32, num_epochs=500, weight_decay=0.1):
    """Test marker-based retrieval task."""
    marker = vocab_size - 1
    
    # CRUCIAL: weight decay for holographic patterns
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=weight_decay)
    
    best_acc = 0.0
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
        
        # Forward - handle tuple returns
        output = model(seq)
        if isinstance(output, tuple):
            logits = output[0]
        else:
            logits = output
        
        loss = F.cross_entropy(logits[:, -1, :], targets)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if epoch % 100 == 99:
            model.eval()
            with torch.no_grad():
                pred = logits[:, -1, :].argmax(dim=-1)
                acc = (pred == targets).float().mean().item()
                best_acc = max(best_acc, acc)
            print(f"  {name} Epoch {epoch+1}: loss={loss.item():.4f}, acc={acc:.1%}")
    
    elapsed = time.time() - start
    
    # Final test
    model.eval()
    test_accs = []
    with torch.no_grad():
        for _ in range(5):
            test_seq = torch.randint(0, vocab_size-2, (batch_size, seq_len), device=device)
            test_targets = torch.randint(0, vocab_size-2, (batch_size,), device=device)
            
            for i in range(batch_size):
                pos = torch.randint(2, seq_len//2, (1,)).item()
                test_seq[i, pos] = marker
                test_seq[i, pos+1] = test_targets[i]
            
            test_seq[:, -2] = marker
            
            output = model(test_seq)
            if isinstance(output, tuple):
                test_logits = output[0]
            else:
                test_logits = output
            
            test_pred = test_logits[:, -1, :].argmax(dim=-1)
            test_accs.append((test_pred == test_targets).float().mean().item())
    
    final_acc = sum(test_accs) / len(test_accs)
    
    return {
        'best_acc': best_acc,
        'final_acc': final_acc,
        'time': elapsed,
    }


def main():
    print("="*70)
    print("COMPREHENSIVE MODEL COMPARISON (with weight_decay=0.1)")
    print("="*70)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    
    vocab_size = 64
    d_model = 64
    num_layers = 2
    num_neurons = 128
    
    results = {}
    
    # 1. SwiGLU Transformer (baseline)
    print("\n" + "="*50)
    print("1. SwiGLU Transformer")
    print("="*50)
    
    model = SwiGLUTransformer(
        vocab_size=vocab_size,
        d_model=d_model,
        n_heads=4,
        num_layers=num_layers,
        d_ff=d_model * 4,
    ).to(device)
    print(f"Parameters: {model.get_num_params():,}")
    results['transformer'] = test_marker_retrieval(model, "Transformer", device, vocab_size=vocab_size)
    
    # 2. Echo V2 (parallel)
    print("\n" + "="*50)
    print("2. Echo V2 (parallel)")
    print("="*50)
    
    model = EchoModelV2(
        vocab_size=vocab_size,
        d_model=d_model,
        num_layers=num_layers,
        num_neurons=num_neurons,
        n_heads=4,
    ).to(device)
    print(f"Parameters: {model.get_num_params():,}")
    results['echo_v2'] = test_marker_retrieval(model, "EchoV2", device, vocab_size=vocab_size)
    
    # 3. RIN (recurrent with attenuation) - shorter training
    print("\n" + "="*50)
    print("3. RIN (recurrent with attenuation)")
    print("="*50)
    
    model = RINModel(
        vocab_size=vocab_size,
        d_model=d_model,
        num_layers=num_layers,
        num_neurons=num_neurons,
    ).to(device)
    print(f"Parameters: {model.get_num_params():,}")
    # Fewer epochs because it's slow
    results['rin'] = test_marker_retrieval(model, "RIN", device, vocab_size=vocab_size, num_epochs=300)
    
    # 4. Episodic Echo V2
    print("\n" + "="*50)
    print("4. Episodic Echo V2")
    print("="*50)
    
    model = EpisodicEchoModelV2(
        vocab_size=vocab_size,
        d_model=d_model,
        num_layers=num_layers,
        num_neurons=num_neurons // 2,  # Smaller to match param count
        n_heads=2,
    ).to(device)
    print(f"Parameters: {model.get_num_params():,}")
    results['episodic'] = test_marker_retrieval(model, "Episodic", device, vocab_size=vocab_size, num_epochs=300)
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"{'Model':<20} {'Best Acc':<12} {'Final Acc':<12} {'Time':<10}")
    print("-"*54)
    for name, r in results.items():
        print(f"{name:<20} {r['best_acc']:.1%}        {r['final_acc']:.1%}        {r['time']:.1f}s")
    
    print("\n" + "="*70)
    print("ANALYSIS")
    print("="*70)
    
    best_model = max(results.keys(), key=lambda k: results[k]['final_acc'])
    print(f"Best performing: {best_model} ({results[best_model]['final_acc']:.1%})")
    
    fastest_model = min(results.keys(), key=lambda k: results[k]['time'])
    print(f"Fastest training: {fastest_model} ({results[fastest_model]['time']:.1f}s)")


if __name__ == "__main__":
    main()

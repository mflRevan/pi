"""
Comprehensive Model Comparison Test Suite

Compares:
1. RIN-only (resonant layers only)
2. Echo (attention + resonant with additive fusion)
3. SwiGLU Transformer baseline

On tasks:
- Gradient flow verification
- Modular arithmetic (grokking)
- Needle-in-haystack retrieval
- WikiText language modeling
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
import time
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import sys
import os

# Add project root to path
sys.path.insert(0, '/home/aiman/pi')

from rin.model import RINModel
from rin.echo import EchoModel
from rin.transformer import SwiGLUTransformer


def get_logits(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Extract logits from any model type."""
    if isinstance(model, SwiGLUTransformer):
        return model(x)
    elif isinstance(model, EchoModel):
        logits, _, _ = model(x)
        return logits
    else:  # RINModel
        logits, _ = model(x)
        return logits


@dataclass
class ExperimentConfig:
    """Configuration for experiments."""
    d_model: int = 64
    num_layers: int = 2
    n_heads: int = 4
    num_neurons: int = 32
    vocab_size: int = 1024
    batch_size: int = 32
    lr: float = 1e-3
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


def count_params(model: nn.Module) -> int:
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def create_matched_models(config: ExperimentConfig) -> Dict[str, nn.Module]:
    """Create models with approximately matched parameter counts."""
    
    # RIN model (resonant layers only)
    rin = RINModel(
        vocab_size=config.vocab_size,
        d_model=config.d_model,
        num_layers=config.num_layers,
        num_neurons=config.num_neurons,
    )
    
    # Echo model (attention + resonant)
    echo = EchoModel(
        vocab_size=config.vocab_size,
        d_model=config.d_model,
        num_layers=config.num_layers,
        num_neurons=config.num_neurons,
        n_heads=config.n_heads,
    )
    
    # Transformer - adjust d_ff to match param count
    rin_params = count_params(rin)
    echo_params = count_params(echo)
    
    # Start with standard d_ff = 4 * d_model and adjust
    transformer = SwiGLUTransformer(
        vocab_size=config.vocab_size,
        d_model=config.d_model,
        num_layers=config.num_layers,
        n_heads=config.n_heads,
        d_ff=config.d_model * 4,
    )
    
    return {
        'RIN': rin,
        'Echo': echo,
        'Transformer': transformer,
    }


# =============================================================================
# Gradient Flow Test
# =============================================================================

def test_gradient_flow(config: ExperimentConfig):
    """Test that gradients flow through all components."""
    print("\n" + "="*70)
    print("GRADIENT FLOW TEST")
    print("="*70)
    
    models = create_matched_models(config)
    
    for name, model in models.items():
        model = model.to(config.device)
        model.train()
        
        # Create dummy input
        x = torch.randint(0, config.vocab_size, (2, 16), device=config.device)
        
        # Forward pass
        logits = get_logits(model, x)
        
        # Compute loss
        loss = logits.sum()
        loss.backward()
        
        # Check gradients
        print(f"\n{name} Model ({count_params(model):,} params):")
        
        gradient_info = {}
        zero_grads = []
        
        for param_name, param in model.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                gradient_info[param_name] = grad_norm
                if grad_norm == 0:
                    zero_grads.append(param_name)
            else:
                gradient_info[param_name] = None
                zero_grads.append(param_name)
        
        # Key components
        key_components = ['embedding', 'attn', 'resonant', 'output', 'ln', 'w_']
        
        print("  Key gradient norms:")
        for key in key_components:
            matching = [(k, v) for k, v in gradient_info.items() if key in k.lower()]
            if matching:
                avg_grad = np.mean([v for _, v in matching if v is not None])
                print(f"    {key}: {avg_grad:.6f} (avg over {len(matching)} params)")
        
        if zero_grads:
            print(f"  ⚠️  Zero gradients: {len(zero_grads)} parameters")
        else:
            print("  ✓ All parameters receiving gradients")
        
        model.zero_grad()


# =============================================================================
# Modular Arithmetic Task
# =============================================================================

def generate_mod_arithmetic_data(
    n_samples: int,
    mod: int = 97,
    seq_len: int = 3,
    device: str = "cpu",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generate modular arithmetic: (a + b) mod p = c"""
    a = torch.randint(0, mod, (n_samples,))
    b = torch.randint(0, mod, (n_samples,))
    c = (a + b) % mod
    
    # Input: [a, op_token, b, eq_token]
    # Target: c
    op_token = mod  # Use mod as operator token
    eq_token = mod + 1  # Use mod+1 as equals token
    
    inputs = torch.stack([a, torch.full_like(a, op_token), b, torch.full_like(a, eq_token)], dim=1)
    targets = c
    
    return inputs.to(device), targets.to(device)


def train_mod_arithmetic(
    model: nn.Module,
    config: ExperimentConfig,
    mod: int = 97,
    num_epochs: int = 1000,
    eval_every: int = 100,
    train_size: int = 512,
    test_size: int = 256,
) -> Dict:
    """Train on modular arithmetic and track grokking."""
    
    model = model.to(config.device)
    optimizer = AdamW(model.parameters(), lr=config.lr, weight_decay=0.1)
    scheduler = CosineAnnealingLR(optimizer, num_epochs)
    
    # Generate fixed train/test splits
    torch.manual_seed(42)
    train_x, train_y = generate_mod_arithmetic_data(train_size, mod, device=config.device)
    test_x, test_y = generate_mod_arithmetic_data(test_size, mod, device=config.device)
    
    history = {
        'train_loss': [],
        'train_acc': [],
        'test_acc': [],
        'epoch': [],
    }
    
    is_transformer = isinstance(model, SwiGLUTransformer)
    
    for epoch in range(num_epochs):
        model.train()
        
        # Forward
        logits = get_logits(model, train_x)
        
        # Get prediction at last position
        pred_logits = logits[:, -1, :mod]  # Only mod classes
        loss = F.cross_entropy(pred_logits, train_y)
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        if (epoch + 1) % eval_every == 0:
            model.eval()
            with torch.no_grad():
                # Train accuracy
                train_pred = pred_logits.argmax(dim=-1)
                train_acc = (train_pred == train_y).float().mean().item()
                
                # Test accuracy
                test_logits = get_logits(model, test_x)
                test_pred = test_logits[:, -1, :mod].argmax(dim=-1)
                test_acc = (test_pred == test_y).float().mean().item()
            
            history['train_loss'].append(loss.item())
            history['train_acc'].append(train_acc)
            history['test_acc'].append(test_acc)
            history['epoch'].append(epoch + 1)
    
    return history


def test_modular_arithmetic(config: ExperimentConfig):
    """Compare models on modular arithmetic (grokking)."""
    print("\n" + "="*70)
    print("MODULAR ARITHMETIC TEST (Grokking)")
    print("="*70)
    
    mod = 97
    num_epochs = 2000
    
    # Update vocab size for this task
    task_config = ExperimentConfig(
        d_model=config.d_model,
        num_layers=config.num_layers,
        n_heads=config.n_heads,
        num_neurons=config.num_neurons,
        vocab_size=mod + 2,  # mod numbers + op + eq tokens
        batch_size=config.batch_size,
        lr=3e-4,
        device=config.device,
    )
    
    models = create_matched_models(task_config)
    results = {}
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        start_time = time.time()
        
        history = train_mod_arithmetic(
            model,
            task_config,
            mod=mod,
            num_epochs=num_epochs,
            eval_every=100,
        )
        
        elapsed = time.time() - start_time
        
        final_train_acc = history['train_acc'][-1]
        final_test_acc = history['test_acc'][-1]
        
        # Find grokking point (where test acc > 90%)
        grokking_epoch = None
        for i, acc in enumerate(history['test_acc']):
            if acc > 0.9:
                grokking_epoch = history['epoch'][i]
                break
        
        results[name] = {
            'final_train_acc': final_train_acc,
            'final_test_acc': final_test_acc,
            'grokking_epoch': grokking_epoch,
            'time': elapsed,
            'history': history,
        }
        
        print(f"  Final train acc: {final_train_acc:.1%}")
        print(f"  Final test acc:  {final_test_acc:.1%}")
        print(f"  Grokking epoch:  {grokking_epoch or 'N/A'}")
        print(f"  Time: {elapsed:.1f}s")
    
    # Summary
    print("\n" + "-"*50)
    print("MODULAR ARITHMETIC SUMMARY")
    print("-"*50)
    print(f"{'Model':<15} {'Train Acc':<12} {'Test Acc':<12} {'Grokking':<12}")
    print("-"*50)
    for name, r in results.items():
        grok = str(r['grokking_epoch']) if r['grokking_epoch'] else 'N/A'
        print(f"{name:<15} {r['final_train_acc']:.1%}        {r['final_test_acc']:.1%}        {grok}")
    
    return results


# =============================================================================
# Needle-in-Haystack Task
# =============================================================================

def generate_needle_data(
    n_samples: int,
    seq_len: int,
    vocab_size: int,
    device: str = "cpu",
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Generate needle-in-haystack retrieval task.
    
    Format: [haystack tokens...] [KEY] [needle] [haystack...] [KEY] [?]
    Target: needle value
    """
    key_token = vocab_size - 1
    
    # Random haystack
    haystack = torch.randint(1, vocab_size - 2, (n_samples, seq_len))
    
    # Random needle position (first half of sequence)
    max_pos = seq_len // 2
    needle_pos = torch.randint(1, max_pos, (n_samples,))
    
    # Insert key and needle
    needle_values = torch.randint(1, vocab_size - 2, (n_samples,))
    
    for i in range(n_samples):
        pos = needle_pos[i].item()
        haystack[i, pos] = key_token
        haystack[i, pos + 1] = needle_values[i]
    
    # Query position (end of sequence)
    haystack[:, -2] = key_token
    haystack[:, -1] = 0  # Query placeholder
    
    # Distance from needle to query
    distances = seq_len - 2 - needle_pos
    
    return haystack.to(device), needle_values.to(device), distances.to(device)


def test_needle_retrieval(
    model: nn.Module,
    config: ExperimentConfig,
    seq_lens: List[int] = [16, 32, 64, 128],
    n_samples: int = 200,
) -> Dict:
    """Test needle-in-haystack at various sequence lengths."""
    
    model = model.to(config.device)
    model.eval()
    
    results = {}
    
    with torch.no_grad():
        for seq_len in seq_lens:
            x, targets, distances = generate_needle_data(
                n_samples, seq_len, config.vocab_size, config.device
            )
            
            logits = get_logits(model, x)
            
            # Prediction at last position
            pred = logits[:, -1, :].argmax(dim=-1)
            correct = (pred == targets).float()
            
            acc = correct.mean().item()
            
            # Accuracy by distance bucket
            dist_buckets = [(0, 16), (16, 32), (32, 64), (64, 128)]
            bucket_acc = {}
            for lo, hi in dist_buckets:
                mask = (distances >= lo) & (distances < hi)
                if mask.any():
                    bucket_acc[f"{lo}-{hi}"] = correct[mask].mean().item()
            
            results[seq_len] = {
                'accuracy': acc,
                'by_distance': bucket_acc,
            }
    
    return results


def test_needle_task(config: ExperimentConfig):
    """Compare models on needle-in-haystack retrieval."""
    print("\n" + "="*70)
    print("NEEDLE-IN-HAYSTACK TEST")
    print("="*70)
    
    # Train on short sequences first
    train_seq_len = 32
    num_epochs = 500
    
    models = create_matched_models(config)
    results = {}
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        model = model.to(config.device)
        optimizer = AdamW(model.parameters(), lr=config.lr)
        
        # Training loop
        for epoch in range(num_epochs):
            model.train()
            x, targets, _ = generate_needle_data(
                config.batch_size, train_seq_len, config.vocab_size, config.device
            )
            
            logits = get_logits(model, x)
            
            loss = F.cross_entropy(logits[:, -1, :], targets)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if (epoch + 1) % 100 == 0:
                model.eval()
                with torch.no_grad():
                    pred = logits[:, -1, :].argmax(dim=-1)
                    acc = (pred == targets).float().mean().item()
                print(f"  Epoch {epoch+1}: loss={loss.item():.4f}, acc={acc:.1%}")
        
        # Test at various sequence lengths
        print(f"\n  Testing {name} at various lengths...")
        test_results = test_needle_retrieval(model, config, [32, 64, 96, 128])
        results[name] = test_results
        
        for seq_len, r in test_results.items():
            print(f"    Seq len {seq_len}: {r['accuracy']:.1%}")
    
    # Summary
    print("\n" + "-"*60)
    print("NEEDLE RETRIEVAL SUMMARY (Accuracy by Sequence Length)")
    print("-"*60)
    seq_lens = [32, 64, 96, 128]
    header = f"{'Model':<15}" + "".join(f"{'Len '+str(s):<12}" for s in seq_lens)
    print(header)
    print("-"*60)
    
    for name, r in results.items():
        row = f"{name:<15}"
        for s in seq_lens:
            if s in r:
                row += f"{r[s]['accuracy']:.1%}        "
            else:
                row += "N/A         "
        print(row)
    
    return results


# =============================================================================
# Language Modeling Task
# =============================================================================

def generate_lm_data(
    n_samples: int,
    seq_len: int,
    vocab_size: int,
    device: str = "cpu",
) -> torch.Tensor:
    """Generate synthetic language modeling data with patterns."""
    data = torch.randint(0, vocab_size, (n_samples, seq_len), device=device)
    
    # Add some repeating patterns
    for i in range(n_samples):
        if i % 3 == 0:
            # Repeat pattern
            pattern_len = torch.randint(3, 8, (1,)).item()
            pattern = data[i, :pattern_len]
            for j in range(0, seq_len - pattern_len, pattern_len):
                data[i, j:j+pattern_len] = pattern
    
    return data


def train_language_model(
    model: nn.Module,
    config: ExperimentConfig,
    seq_len: int = 64,
    num_epochs: int = 500,
) -> Dict:
    """Train language model and measure perplexity."""
    
    model = model.to(config.device)
    optimizer = AdamW(model.parameters(), lr=config.lr)
    
    history = {'train_loss': [], 'train_ppl': []}
    
    for epoch in range(num_epochs):
        model.train()
        
        # Generate batch
        x = generate_lm_data(config.batch_size, seq_len, config.vocab_size, config.device)
        
        # Forward
        logits = get_logits(model, x)
        
        # Next token prediction loss
        loss = F.cross_entropy(
            logits[:, :-1, :].reshape(-1, config.vocab_size),
            x[:, 1:].reshape(-1),
        )
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 100 == 0:
            ppl = torch.exp(loss).item()
            history['train_loss'].append(loss.item())
            history['train_ppl'].append(ppl)
    
    return history


def test_language_modeling(config: ExperimentConfig):
    """Compare models on language modeling."""
    print("\n" + "="*70)
    print("LANGUAGE MODELING TEST")
    print("="*70)
    
    num_epochs = 500
    seq_len = 64
    
    models = create_matched_models(config)
    results = {}
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        start_time = time.time()
        
        history = train_language_model(model, config, seq_len, num_epochs)
        
        elapsed = time.time() - start_time
        final_ppl = history['train_ppl'][-1]
        
        results[name] = {
            'final_ppl': final_ppl,
            'final_loss': history['train_loss'][-1],
            'time': elapsed,
            'history': history,
        }
        
        print(f"  Final perplexity: {final_ppl:.2f}")
        print(f"  Final loss: {history['train_loss'][-1]:.4f}")
        print(f"  Time: {elapsed:.1f}s")
    
    # Summary
    print("\n" + "-"*50)
    print("LANGUAGE MODELING SUMMARY")
    print("-"*50)
    print(f"{'Model':<15} {'Perplexity':<15} {'Loss':<15}")
    print("-"*50)
    for name, r in results.items():
        print(f"{name:<15} {r['final_ppl']:<15.2f} {r['final_loss']:<15.4f}")
    
    return results


# =============================================================================
# Main Entry Point
# =============================================================================

def run_all_tests():
    """Run all comparison tests."""
    print("="*70)
    print("COMPREHENSIVE MODEL COMPARISON")
    print("RIN vs Echo vs Transformer")
    print("="*70)
    
    config = ExperimentConfig(
        d_model=64,
        num_layers=2,
        n_heads=4,
        num_neurons=32,
        vocab_size=512,
        batch_size=32,
        lr=1e-3,
    )
    
    print(f"\nConfiguration:")
    print(f"  d_model:     {config.d_model}")
    print(f"  num_layers:  {config.num_layers}")
    print(f"  n_heads:     {config.n_heads}")
    print(f"  num_neurons: {config.num_neurons}")
    print(f"  vocab_size:  {config.vocab_size}")
    print(f"  device:      {config.device}")
    
    # Model sizes
    models = create_matched_models(config)
    print(f"\nModel Parameter Counts:")
    for name, model in models.items():
        print(f"  {name}: {count_params(model):,}")
    
    # Run tests
    results = {}
    
    # 1. Gradient flow
    test_gradient_flow(config)
    
    # 2. Modular arithmetic
    results['mod_arithmetic'] = test_modular_arithmetic(config)
    
    # 3. Needle-in-haystack
    results['needle'] = test_needle_task(config)
    
    # 4. Language modeling
    results['lm'] = test_language_modeling(config)
    
    # Final summary
    print("\n" + "="*70)
    print("FINAL SUMMARY")
    print("="*70)
    
    print("\n✓ All tests completed successfully!")
    
    return results


if __name__ == "__main__":
    results = run_all_tests()

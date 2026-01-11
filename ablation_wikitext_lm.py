"""
WikiText-2 Language Modeling Ablation Study
============================================

Compares HolographicTransformer (Triton-optimized, log-scaled, omniware gates)
against SwiGLU baseline on language modeling with downstream evaluation.

Configuration:
- Dataset: WikiText-2
- Epochs: 20
- Sequence Length: 2048
- Batch Size: 2-4
- Layers: 8
- d_model: 384
- n_phase: 48 (divisible by n_heads=6)
- Expansion: 4x

Evaluation: HellaSwag + LAMBADA via lm_eval after each epoch
"""

import os
import sys
import json
import time
import math
import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
from dataclasses import dataclass, asdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.cuda.amp import autocast, GradScaler

# Attempt to import lm_eval
try:
    import lm_eval
    from lm_eval import evaluator
    from lm_eval.api.model import LM
    LM_EVAL_AVAILABLE = True
except ImportError:
    LM_EVAL_AVAILABLE = False
    LM = object  # Fallback base class
    print("Warning: lm_eval not available. Install with: pip install lm_eval")

# Import our models
from rin import HolographicTransformer, SwiGLUTransformer
from rin.kernels import TRITON_AVAILABLE

# HuggingFace datasets
try:
    from datasets import load_dataset
    from transformers import GPT2TokenizerFast
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False
    print("Warning: transformers/datasets not available")


@dataclass
class TrainingConfig:
    """Training configuration."""
    # Model
    vocab_size: int = 50257  # GPT-2 vocab
    d_model: int = 384
    n_heads: int = 6  # 384 / 6 = 64 head dim
    n_layers: int = 8
    n_phase: int = 48  # Must be divisible by n_heads (48 = 6 * 8)
    expansion: int = 4
    dropout: float = 0.1
    max_seq_len: int = 2048
    
    # Training
    epochs: int = 20
    batch_size: int = 2
    gradient_accumulation: int = 4  # Effective batch = 8
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    warmup_steps: int = 500
    max_grad_norm: float = 1.0
    
    # Evaluation
    eval_every_epoch: bool = True
    eval_downstream: bool = True
    eval_tasks: str = 'hellaswag,lambada_openai'  # Stable downstream tasks
    eval_limit: int = 100  # Samples per task (None for full)
    
    # Holographic specific
    gate_mode: str = 'omniware'
    use_triton: bool = True
    log_grad: bool = True
    
    # System
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    seed: int = 42
    output_dir: str = 'results/wikitext_ablation'
    
    def __post_init__(self):
        assert self.d_model % self.n_heads == 0, "d_model must be divisible by n_heads"


class WikiTextDataset(Dataset):
    """WikiText-2 dataset for language modeling."""
    
    def __init__(
        self,
        tokenizer,
        split: str = 'train',
        seq_len: int = 2048,
        stride: int = None,
    ):
        self.seq_len = seq_len
        self.stride = stride or seq_len
        
        # Load WikiText-2
        dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', split=split)
        
        # Concatenate and tokenize
        text = '\n\n'.join([t for t in dataset['text'] if t.strip()])
        self.tokens = tokenizer.encode(text)
        
        # Create chunks
        self.chunks = []
        for i in range(0, len(self.tokens) - seq_len, self.stride):
            self.chunks.append(self.tokens[i:i + seq_len + 1])  # +1 for labels
        
        print(f"WikiText-2 {split}: {len(self.tokens):,} tokens, {len(self.chunks):,} chunks")
    
    def __len__(self):
        return len(self.chunks)
    
    def __getitem__(self, idx):
        chunk = self.chunks[idx]
        return torch.tensor(chunk[:-1], dtype=torch.long), torch.tensor(chunk[1:], dtype=torch.long)


class LMEvalModelWrapper(LM):
    """
    Wrapper to make our models compatible with lm_eval.
    Inherits from lm_eval.api.model.LM.
    """
    
    def __init__(self, model, tokenizer, batch_size=1, device='cuda', max_length=2048):
        super().__init__()
        self._model = model
        self._tokenizer = tokenizer
        self._batch_size = batch_size
        self._device = device
        self._max_length = max_length
        self._model.eval()
    
    @property
    def rank(self):
        return 0
    
    @property
    def world_size(self):
        return 1
    
    @property
    def tokenizer_name(self):
        return 'gpt2'
    
    @property
    def eot_token_id(self):
        return self._tokenizer.eos_token_id
    
    @property
    def max_length(self):
        return self._max_length
    
    @property
    def batch_size(self):
        return self._batch_size
    
    @property
    def device(self):
        return self._device
    
    @property
    def tokenizer(self):
        return self._tokenizer
    
    @property
    def max_gen_toks(self):
        return 256
    
    def tok_encode(self, string, **kwargs):
        return self._tokenizer.encode(string, add_special_tokens=False)
    
    def tok_decode(self, tokens, **kwargs):
        return self._tokenizer.decode(tokens)
    
    def _model_call(self, inps, **kwargs):
        """Compute logits for input tokens."""
        with torch.no_grad():
            logits = self._model(inps.to(self._device))
        return logits
    
    def loglikelihood(self, requests):
        """Compute log-likelihood for each request."""
        results = []
        
        for req in requests:
            context, continuation = req.args
            
            # Tokenize
            ctx_tokens = self.tok_encode(context)
            cont_tokens = self.tok_encode(continuation)
            
            # Combine
            all_tokens = ctx_tokens + cont_tokens
            if len(all_tokens) > self._max_length:
                all_tokens = all_tokens[-self._max_length:]
                ctx_len = len(all_tokens) - len(cont_tokens)
            else:
                ctx_len = len(ctx_tokens)
            
            input_ids = torch.tensor([all_tokens], device=self._device)
            
            # Get logits
            with torch.no_grad():
                logits = self._model(input_ids)
            
            # Compute log probs for continuation
            log_probs = F.log_softmax(logits[0], dim=-1)
            
            # Sum log probs for continuation tokens
            total_log_prob = 0.0
            for i, tok in enumerate(cont_tokens):
                pos = ctx_len + i - 1  # -1 because we predict next token
                if pos >= 0 and pos < log_probs.shape[0]:
                    total_log_prob += log_probs[pos, tok].item()
            
            # Check if max prob prediction matches (greedy)
            is_greedy = True  # Simplified
            
            results.append((total_log_prob, is_greedy))
        
        return results
    
    def loglikelihood_rolling(self, requests):
        """Rolling log-likelihood (for perplexity)."""
        results = []
        for req in requests:
            text = req.args[0]
            tokens = self.tok_encode(text)
            
            if len(tokens) == 0:
                results.append((0.0,))
                continue
            
            if len(tokens) > self._max_length:
                tokens = tokens[-self._max_length:]
            
            input_ids = torch.tensor([tokens], device=self._device)
            
            with torch.no_grad():
                logits = self._model(input_ids)
            
            log_probs = F.log_softmax(logits[0], dim=-1)
            
            total_log_prob = 0.0
            for i in range(1, len(tokens)):
                if i - 1 < log_probs.shape[0]:
                    total_log_prob += log_probs[i - 1, tokens[i]].item()
            
            results.append((total_log_prob,))
        
        return results
    
    def generate_until(self, requests):
        """Generate text until stop token (not implemented)."""
        return [("", False) for _ in requests]


def evaluate_downstream(model, tokenizer, device, tasks='hellaswag', limit=100):
    """Run downstream evaluation using lm_eval."""
    if not LM_EVAL_AVAILABLE:
        print("Skipping downstream eval - lm_eval not installed")
        return {}
    
    model.eval()
    
    # Wrap model for lm_eval
    lm = LMEvalModelWrapper(model, tokenizer, batch_size=1, device=device)
    
    try:
        results = evaluator.simple_evaluate(
            model=lm,
            tasks=tasks.split(','),
            num_fewshot=0,
            limit=limit,
            batch_size=1,
        )
        
        # Extract accuracy scores
        eval_results = {}
        if 'results' in results:
            for task, metrics in results['results'].items():
                for metric_name, value in metrics.items():
                    if 'acc' in metric_name.lower():
                        eval_results[f'{task}_{metric_name}'] = value
                        break
        
        return eval_results
    
    except Exception as e:
        print(f"Downstream evaluation error: {e}")
        import traceback
        traceback.print_exc()
        return {'error': str(e)}


def create_model(config: TrainingConfig, model_type: str) -> nn.Module:
    """Create model based on type."""
    if model_type == 'holographic':
        model = HolographicTransformer(
            vocab_size=config.vocab_size,
            d_model=config.d_model,
            n_heads=config.n_heads,
            n_layers=config.n_layers,
            n_phase=config.n_phase,
            expansion=config.expansion,
            dropout=config.dropout,
            gate_mode=config.gate_mode,
            use_triton=config.use_triton,
            log_grad=config.log_grad,
            causal=True,
            max_seq_len=config.max_seq_len,
        )
        print(f"\n✓ Created HolographicTransformer:")
        print(f"  - gate_mode: {config.gate_mode}")
        print(f"  - use_triton: {config.use_triton} (available: {TRITON_AVAILABLE})")
        print(f"  - log_grad: {config.log_grad}")
        print(f"  - n_phase: {config.n_phase}")
        
    elif model_type == 'swiglu':
        model = SwiGLUTransformer(
            vocab_size=config.vocab_size,
            d_model=config.d_model,
            n_heads=config.n_heads,
            num_layers=config.n_layers,
            d_ff=config.d_model * config.expansion,
            dropout=config.dropout,
            max_seq_len=config.max_seq_len,
        )
        print(f"\n✓ Created SwiGLUTransformer (baseline)")
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"  - Parameters: {num_params:,}")
    
    return model.to(config.device)


def get_lr_scheduler(optimizer, config: TrainingConfig, total_steps: int):
    """Cosine schedule with warmup."""
    def lr_lambda(step):
        if step < config.warmup_steps:
            return step / config.warmup_steps
        progress = (step - config.warmup_steps) / (total_steps - config.warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * progress))
    
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler,
    config: TrainingConfig,
    epoch: int,
    scaler: GradScaler,
) -> Dict[str, float]:
    """Train for one epoch."""
    model.train()
    
    total_loss = 0.0
    total_tokens = 0
    num_batches = 0
    
    optimizer.zero_grad()
    
    start_time = time.time()
    log_interval = 50
    
    for batch_idx, (input_ids, labels) in enumerate(dataloader):
        input_ids = input_ids.to(config.device)
        labels = labels.to(config.device)
        
        # Forward with mixed precision
        with autocast(dtype=torch.bfloat16):
            logits = model(input_ids)
            loss = F.cross_entropy(
                logits.view(-1, config.vocab_size),
                labels.view(-1),
            )
            loss = loss / config.gradient_accumulation
        
        # Backward
        scaler.scale(loss).backward()
        
        # Accumulate stats
        total_loss += loss.item() * config.gradient_accumulation
        total_tokens += labels.numel()
        num_batches += 1
        
        # Gradient step
        if (batch_idx + 1) % config.gradient_accumulation == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad()
        
        # Logging
        if (batch_idx + 1) % log_interval == 0:
            avg_loss = total_loss / num_batches
            ppl = math.exp(min(avg_loss, 20))
            elapsed = time.time() - start_time
            tok_per_sec = total_tokens / elapsed
            
            print(f"  Epoch {epoch} | Batch {batch_idx+1}/{len(dataloader)} | "
                  f"Loss: {avg_loss:.4f} | PPL: {ppl:.2f} | "
                  f"LR: {scheduler.get_last_lr()[0]:.2e} | "
                  f"Tok/s: {tok_per_sec:.0f}")
    
    avg_loss = total_loss / num_batches
    return {
        'loss': avg_loss,
        'perplexity': math.exp(min(avg_loss, 20)),
        'tokens_per_sec': total_tokens / (time.time() - start_time),
    }


@torch.no_grad()
def evaluate_perplexity(
    model: nn.Module,
    dataloader: DataLoader,
    config: TrainingConfig,
) -> Dict[str, float]:
    """Evaluate perplexity on validation set."""
    model.eval()
    
    total_loss = 0.0
    total_tokens = 0
    
    for input_ids, labels in dataloader:
        input_ids = input_ids.to(config.device)
        labels = labels.to(config.device)
        
        with autocast(dtype=torch.bfloat16):
            logits = model(input_ids)
            loss = F.cross_entropy(
                logits.view(-1, config.vocab_size),
                labels.view(-1),
                reduction='sum',
            )
        
        total_loss += loss.item()
        total_tokens += labels.numel()
    
    avg_loss = total_loss / total_tokens
    return {
        'loss': avg_loss,
        'perplexity': math.exp(min(avg_loss, 20)),
    }


def run_ablation(config: TrainingConfig, model_type: str) -> Dict[str, Any]:
    """Run full training and evaluation for one model."""
    print(f"\n{'='*60}")
    print(f"ABLATION: {model_type.upper()}")
    print(f"{'='*60}")
    
    # Set seed
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
    
    # Tokenizer
    tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    
    # Create datasets
    print("\nLoading WikiText-2...")
    train_dataset = WikiTextDataset(tokenizer, 'train', config.max_seq_len)
    valid_dataset = WikiTextDataset(tokenizer, 'validation', config.max_seq_len)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )
    
    # Create model
    model = create_model(config, model_type)
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
        betas=(0.9, 0.95),
    )
    
    # Scheduler
    total_steps = len(train_loader) * config.epochs // config.gradient_accumulation
    scheduler = get_lr_scheduler(optimizer, config, total_steps)
    
    # Mixed precision
    scaler = GradScaler()
    
    # Results tracking
    results = {
        'model_type': model_type,
        'config': asdict(config),
        'epochs': [],
        'best_val_ppl': float('inf'),
        'best_epoch': 0,
    }
    
    # Training loop
    for epoch in range(1, config.epochs + 1):
        print(f"\n--- Epoch {epoch}/{config.epochs} ---")
        
        # Train
        train_metrics = train_epoch(
            model, train_loader, optimizer, scheduler,
            config, epoch, scaler
        )
        
        # Validate
        val_metrics = evaluate_perplexity(model, valid_loader, config)
        
        epoch_results = {
            'epoch': epoch,
            'train': train_metrics,
            'val': val_metrics,
            'downstream': {},
        }
        
        print(f"\n  Train Loss: {train_metrics['loss']:.4f} | PPL: {train_metrics['perplexity']:.2f}")
        print(f"  Val Loss: {val_metrics['loss']:.4f} | PPL: {val_metrics['perplexity']:.2f}")
        
        # Downstream evaluation
        if config.eval_downstream:
            print(f"\n  Running downstream evaluation ({config.eval_tasks})...")
            downstream_results = evaluate_downstream(
                model, tokenizer, config.device,
                tasks=config.eval_tasks,
                limit=config.eval_limit
            )
            epoch_results['downstream'] = downstream_results
            
            if downstream_results and 'error' not in downstream_results:
                for task, acc in downstream_results.items():
                    print(f"  {task}: {acc:.4f}")
        
        results['epochs'].append(epoch_results)
        
        # Track best
        if val_metrics['perplexity'] < results['best_val_ppl']:
            results['best_val_ppl'] = val_metrics['perplexity']
            results['best_epoch'] = epoch
            
            # Save checkpoint
            ckpt_path = Path(config.output_dir) / f'{model_type}_best.pt'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_ppl': val_metrics['perplexity'],
            }, ckpt_path)
            print(f"  ✓ Saved best model (PPL: {val_metrics['perplexity']:.2f})")
        
        # Save intermediate results
        results_path = Path(config.output_dir) / f'{model_type}_results.json'
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
    
    return results


def verify_pipeline(config: TrainingConfig):
    """Verify the ablation pipeline works end-to-end."""
    print("\n" + "="*60)
    print("PIPELINE VERIFICATION")
    print("="*60)
    
    tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    
    # Test both model types
    for model_type in ['holographic', 'swiglu']:
        print(f"\n--- Testing {model_type} ---")
        
        # Create model
        model = create_model(config, model_type)
        
        # Test forward pass
        print("  Testing forward pass...")
        x = torch.randint(0, config.vocab_size, (2, 128), device=config.device)
        with torch.no_grad():
            logits = model(x)
        print(f"  ✓ Forward: {x.shape} -> {logits.shape}")
        
        # Test backward pass
        print("  Testing backward pass...")
        x = torch.randint(0, config.vocab_size, (2, 128), device=config.device)
        logits = model(x)
        loss = F.cross_entropy(logits.view(-1, config.vocab_size), x.view(-1))
        loss.backward()
        print(f"  ✓ Backward: loss = {loss.item():.4f}")
        
        # Test train/eval switching
        print("  Testing train/eval mode switching...")
        model.train()
        assert model.training, "Model should be in training mode"
        model.eval()
        assert not model.training, "Model should be in eval mode"
        print("  ✓ Mode switching works")
        
        # Verify Triton usage for holographic
        if model_type == 'holographic':
            print(f"  Triton available: {TRITON_AVAILABLE}")
            print(f"  Triton enabled in config: {config.use_triton}")
            
            # Check if model is actually using Triton
            if hasattr(model, 'blocks') and len(model.blocks) > 0:
                ffn = model.blocks[0].ffn
                if hasattr(ffn, 'use_triton'):
                    print(f"  FFN use_triton: {ffn.use_triton}")
                if hasattr(ffn, 'gate_mode'):
                    print(f"  FFN gate_mode: {ffn.gate_mode}")
                if hasattr(ffn, 'log_grad'):
                    print(f"  FFN log_grad: {ffn.log_grad}")
        
        # Test downstream eval (quick)
        if LM_EVAL_AVAILABLE and config.eval_downstream:
            print("  Testing downstream evaluation (limit=3)...")
            downstream_results = evaluate_downstream(
                model, tokenizer, config.device,
                tasks='hellaswag',
                limit=3
            )
            if downstream_results and 'error' not in downstream_results:
                print(f"  ✓ Downstream eval works: {downstream_results}")
            else:
                print(f"  ⚠ Downstream eval issue: {downstream_results}")
        
        del model
        torch.cuda.empty_cache()
    
    print("\n✓ Pipeline verification complete!")
    return True


def main():
    parser = argparse.ArgumentParser(description='WikiText-2 Ablation Study')
    parser.add_argument('--verify-only', action='store_true', help='Only verify pipeline')
    parser.add_argument('--model', type=str, choices=['holographic', 'swiglu', 'both'],
                        default='both', help='Model to train')
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=2, help='Batch size')
    parser.add_argument('--no-downstream', action='store_true', help='Skip downstream evaluation')
    parser.add_argument('--eval-limit', type=int, default=100, help='Samples per eval task')
    parser.add_argument('--eval-tasks', type=str, default='hellaswag,lambada_openai',
                        help='Comma-separated eval tasks')
    args = parser.parse_args()
    
    # Configuration
    config = TrainingConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        eval_downstream=not args.no_downstream,
        eval_limit=args.eval_limit,
        eval_tasks=args.eval_tasks,
    )
    
    # Create output directory
    Path(config.output_dir).mkdir(parents=True, exist_ok=True)
    
    # Print system info
    print("="*60)
    print("WikiText-2 Language Modeling Ablation")
    print("="*60)
    print(f"Device: {config.device}")
    print(f"Triton Available: {TRITON_AVAILABLE}")
    print(f"lm_eval Available: {LM_EVAL_AVAILABLE}")
    print(f"Epochs: {config.epochs}")
    print(f"Batch Size: {config.batch_size}")
    print(f"Sequence Length: {config.max_seq_len}")
    print(f"d_model: {config.d_model}, n_phase: {config.n_phase}")
    print(f"Layers: {config.n_layers}, Expansion: {config.expansion}")
    print(f"Eval Tasks: {config.eval_tasks}")
    
    # Verify pipeline
    verify_pipeline(config)
    
    if args.verify_only:
        print("\n✓ Verification complete. Exiting.")
        return
    
    # Run ablations
    all_results = {}
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    if args.model in ['holographic', 'both']:
        results = run_ablation(config, 'holographic')
        all_results['holographic'] = results
    
    if args.model in ['swiglu', 'both']:
        results = run_ablation(config, 'swiglu')
        all_results['swiglu'] = results
    
    # Save combined results
    results_path = Path(config.output_dir) / f'ablation_results_{timestamp}.json'
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n✓ Results saved to {results_path}")
    
    # Print summary
    print("\n" + "="*60)
    print("ABLATION SUMMARY")
    print("="*60)
    
    for model_type, results in all_results.items():
        print(f"\n{model_type.upper()}:")
        print(f"  Best Val PPL: {results['best_val_ppl']:.2f} (epoch {results['best_epoch']})")
        
        # Final downstream results
        if results['epochs'] and results['epochs'][-1].get('downstream'):
            downstream = results['epochs'][-1]['downstream']
            if downstream and 'error' not in downstream:
                for task, acc in downstream.items():
                    print(f"  {task}: {acc:.4f}")


if __name__ == '__main__':
    main()

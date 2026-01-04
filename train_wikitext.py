#!/usr/bin/env python3
"""
Train RIN on WikiText-2 Language Modeling

Language modeling with the Euler-based Resonant Interference Network.
Uses continuous waveform transformations with golden ratio timesteps.

By default, uses EchoChamberModel with ResonantBlocks (Echo memory + Resonant layers).
Can also use the original RINModel (resonant-only, no memory) with --model rin.

BPTT (Backpropagation Through Time):
    Sequences are processed in chunks (default 32 tokens) to reduce memory usage.
    Memory state is detached between chunks for truncated BPTT.
    Echo chamber memory persists across chunks but gradients are truncated.

Usage:
    python train_wikitext.py                           # EchoChamber model (default)
    python train_wikitext.py --model rin               # Original RIN model
    python train_wikitext.py --bptt_chunk 64           # Larger BPTT chunks
    python train_wikitext.py --d_model 256 --num_layers 4 --epochs 20
    python train_wikitext.py --n_echo_heads 8          # More echo heads
"""

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import argparse
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# Enable TF32 for better performance on Ampere+ GPUs
torch.set_float32_matmul_precision('high')

from transformers import GPT2TokenizerFast
from datasets import load_dataset

from rin import RINModel, PHI
from rin.echo_chamber import EchoChamberModel


class TextDataset(Dataset):
    """Chunked text sequences for language modeling."""
    
    def __init__(self, tokens: list, seq_len: int):
        self.tokens = tokens
        self.seq_len = seq_len
        self.num_sequences = (len(tokens) - 1) // seq_len
    
    def __len__(self):
        return self.num_sequences
    
    def __getitem__(self, idx):
        start = idx * self.seq_len
        end = start + self.seq_len + 1
        chunk = self.tokens[start:end]
        return torch.tensor(chunk[:-1], dtype=torch.long), torch.tensor(chunk[1:], dtype=torch.long)


def load_data(tokenizer, max_tokens: int = 1_000_000):
    """Load and tokenize WikiText-2."""
    print("\nLoading WikiText-2...")
    
    try:
        dataset = load_dataset("wikitext", "wikitext-2-raw-v1", trust_remote_code=True)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("Trying alternative...")
        dataset = load_dataset("wikitext", "wikitext-2-v1", trust_remote_code=True)
    
    def tokenize_split(split_name):
        texts = dataset[split_name]["text"]
        all_tokens = []
        for text in tqdm(texts, desc=f"Tokenizing {split_name}"):
            if text.strip():
                tokens = tokenizer.encode(text)
                all_tokens.extend(tokens)
            if max_tokens and len(all_tokens) >= max_tokens:
                break
        return all_tokens[:max_tokens] if max_tokens else all_tokens
    
    train_tokens = tokenize_split("train")
    val_tokens = tokenize_split("validation")
    test_tokens = tokenize_split("test")
    
    print(f"Train tokens: {len(train_tokens):,}")
    print(f"Val tokens: {len(val_tokens):,}")
    print(f"Test tokens: {len(test_tokens):,}")
    
    return train_tokens, val_tokens, test_tokens


def evaluate(model, dataloader, device):
    """Evaluate model and return average loss and perplexity."""
    model.eval()
    total_loss = 0
    total_tokens = 0
    
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            logits, _ = model(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1), reduction='sum')
            total_loss += loss.item()
            total_tokens += y.numel()
    
    avg_loss = total_loss / total_tokens
    perplexity = math.exp(min(avg_loss, 100))  # Cap to avoid overflow
    return avg_loss, perplexity


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Golden ratio φ = {PHI:.6f}")
    
    # Tokenizer
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    vocab_size = tokenizer.vocab_size
    print(f"Vocab size: {vocab_size}")
    
    # Data
    train_tokens, val_tokens, test_tokens = load_data(tokenizer, args.max_tokens)
    
    train_ds = TextDataset(train_tokens, args.seq_len)
    val_ds = TextDataset(val_tokens, args.seq_len)
    test_ds = TextDataset(test_tokens, args.seq_len)
    
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, num_workers=2, pin_memory=True)
    
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    
    # Model
    if args.model == "echo":
        print(f"\nUsing EchoChamberModel (ResonantBlocks with Echo memory)")
        model = EchoChamberModel(
            vocab_size=vocab_size,
            d_model=args.d_model,
            num_layers=args.num_layers,
            num_neurons=args.num_neurons,
            n_echo_heads=args.n_echo_heads,
        ).to(device)
    else:
        print(f"\nUsing RINModel (Resonant-only, no memory)")
        model = RINModel(
            vocab_size=vocab_size,
            d_model=args.d_model,
            num_layers=args.num_layers,
            num_neurons=args.num_neurons,
            use_swish=args.use_swish,
            wrap_time=True
        ).to(device)
    
    print(f"\n{model}")
    
    if hasattr(torch, 'compile') and args.compile:
        print("Compiling model...")
        model = torch.compile(model)
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=args.lr, 
        weight_decay=args.weight_decay,
        betas=(0.9, 0.95),
    )
    
    # Learning rate scheduler
    total_steps = len(train_loader) * args.epochs
    warmup_steps = min(2000, total_steps // 10)
    
    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        progress = (step - warmup_steps) / (total_steps - warmup_steps)
        return 0.1 + 0.9 * (1 + math.cos(math.pi * progress)) / 2
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    # Training
    best_val_loss = float('inf')
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(exist_ok=True)
    
    global_step = 0
    
    print(f"\n{'='*60}")
    print("Training")
    print(f"{'='*60}")
    
    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0
        epoch_start = time.time()
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        for batch_idx, (x, y) in enumerate(pbar):
            x, y = x.to(device), y.to(device)
            
            optimizer.zero_grad(set_to_none=True)
            
            # BPTT with chunking to reduce memory usage
            if args.bptt_chunk > 0 and x.size(1) > args.bptt_chunk:
                # Process sequence in chunks
                total_loss = 0
                num_chunks = 0
                hidden = None
                
                for chunk_start in range(0, x.size(1), args.bptt_chunk):
                    chunk_end = min(chunk_start + args.bptt_chunk, x.size(1))
                    x_chunk = x[:, chunk_start:chunk_end].contiguous()
                    y_chunk = y[:, chunk_start:chunk_end].contiguous()
                    
                    if args.model == "echo":
                        # Reset memory only on first chunk
                        logits, hidden = model(x_chunk, hidden=hidden, reset_memory=(chunk_start == 0))
                        # Detach hidden to truncate BPTT
                        hidden = (hidden[0].detach(), hidden[1].detach())
                    else:
                        logits, hidden = model(x_chunk, hidden=hidden)
                        # Detach hidden to truncate BPTT
                        hidden = (hidden[0].detach(), hidden[1].detach())
                    
                    chunk_loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y_chunk.reshape(-1))
                    chunk_loss.backward()
                    total_loss += chunk_loss.item()
                    num_chunks += 1
                
                loss = torch.tensor(total_loss / num_chunks)  # For logging
            else:
                # Full sequence (no chunking)
                if args.model == "echo":
                    logits, _ = model(x, reset_memory=True)
                else:
                    logits, _ = model(x)
                
                loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
                loss.backward()
            
            # Gradient clipping
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            
            optimizer.step()
            scheduler.step()
            
            epoch_loss += loss.item()
            global_step += 1
            
            if batch_idx % 50 == 0:
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'ppl': f'{math.exp(min(loss.item(), 10)):.1f}',
                    'lr': f'{scheduler.get_last_lr()[0]:.2e}',
                })
        
        epoch_time = time.time() - epoch_start
        avg_train_loss = epoch_loss / len(train_loader)
        
        # Validation
        val_loss, val_ppl = evaluate(model, val_loader, device)
        
        print(f"\nEpoch {epoch+1}: train_loss={avg_train_loss:.4f}, val_loss={val_loss:.4f}, "
              f"val_ppl={val_ppl:.2f}, time={epoch_time:.1f}s")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_ppl': val_ppl,
            }, checkpoint_dir / "best_model.pt")
            print(f"  Saved best model (val_ppl={val_ppl:.2f})")
    
    # Final evaluation
    print(f"\n{'='*60}")
    print("Final Evaluation")
    print(f"{'='*60}")
    
    # Load best model
    checkpoint = torch.load(checkpoint_dir / "best_model.pt")
    model.load_state_dict(checkpoint['model_state_dict'])
    
    test_loss, test_ppl = evaluate(model, test_loader, device)
    print(f"Test perplexity: {test_ppl:.2f}")
    
    return {"test_ppl": test_ppl, "best_val_ppl": math.exp(best_val_loss)}


def main():
    parser = argparse.ArgumentParser(description="Train RIN on WikiText-2")
    parser.add_argument("--model", type=str, default="echo", choices=["echo", "rin"],
                        help="Model type: 'echo' (EchoChamberModel with memory) or 'rin' (resonant-only)")
    parser.add_argument("--d_model", type=int, default=256, help="Model dimension")
    parser.add_argument("--num_layers", type=int, default=4, help="Number of layers")
    parser.add_argument("--num_neurons", type=int, default=512, help="Neurons per layer")
    parser.add_argument("--n_echo_heads", type=int, default=4, help="Number of echo heads (echo model only)")
    parser.add_argument("--seq_len", type=int, default=128, help="Sequence length")
    parser.add_argument("--bptt_chunk", type=int, default=32, help="BPTT chunk size (0=full sequence)")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--epochs", type=int, default=10, help="Training epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="Gradient clipping")
    parser.add_argument("--max_tokens", type=int, default=1_000_000, help="Max tokens to load")
    parser.add_argument("--use_swish", action="store_true", default=True, help="Use swish")
    parser.add_argument("--no_swish", action="store_false", dest="use_swish")
    parser.add_argument("--compile", action="store_true", help="Use torch.compile")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints", help="Checkpoint directory")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("RIN - WikiText-2 Language Modeling")
    print("=" * 60)
    
    train(args)


if __name__ == "__main__":
    main()

"""
Training Script for Resonant Interference Network

Implements training loop for language modeling with:
- GPT-2 tokenizer
- Mixed precision training
- Learning rate scheduling
- Gradient clipping
- Logging and checkpointing
"""

import os
import sys
import math
import time
import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.cuda.amp import autocast, GradScaler

# Transformers for tokenizer and datasets
from transformers import GPT2Tokenizer, GPT2TokenizerFast
from datasets import load_dataset

# Local imports
from rin import RINModel
from rin.config import RINConfig, ModelConfig, TrainingConfig, DataConfig, TINY_CONFIG


class TextDataset(Dataset):
    """
    Simple text dataset that tokenizes and chunks text for language modeling.
    """
    
    def __init__(
        self,
        texts: list,
        tokenizer,
        max_length: int = 512,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Tokenize all texts and concatenate
        all_tokens = []
        for text in texts:
            if text.strip():  # Skip empty strings
                tokens = tokenizer.encode(text, add_special_tokens=False)
                all_tokens.extend(tokens)
        
        # Create chunks of max_length
        self.examples = []
        for i in range(0, len(all_tokens) - max_length, max_length // 2):
            chunk = all_tokens[i:i + max_length]
            if len(chunk) == max_length:
                self.examples.append(torch.tensor(chunk, dtype=torch.long))
        
        print(f"Created {len(self.examples)} training examples from {len(all_tokens)} tokens")
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        return self.examples[idx]


def collate_fn(batch):
    """Collate function for DataLoader."""
    return torch.stack(batch, dim=0)


def get_lr_scheduler(optimizer, config: TrainingConfig, total_steps: int):
    """Create learning rate scheduler."""
    
    def lr_lambda(step):
        # Warmup
        if step < config.warmup_steps:
            return step / config.warmup_steps
        
        # Decay
        progress = (step - config.warmup_steps) / (total_steps - config.warmup_steps)
        
        if config.lr_scheduler == "cosine":
            return config.min_lr / config.learning_rate + (1 - config.min_lr / config.learning_rate) * (
                0.5 * (1 + math.cos(math.pi * progress))
            )
        elif config.lr_scheduler == "linear":
            return 1.0 - progress * (1 - config.min_lr / config.learning_rate)
        else:  # constant
            return 1.0
    
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def train_epoch(
    model: RINModel,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler,
    scaler: GradScaler,
    config: RINConfig,
    epoch: int,
    global_step: int,
) -> Tuple[float, int]:
    """
    Train for one epoch.
    
    Returns:
        Tuple of (average_loss, global_step)
    """
    model.train()
    device = next(model.parameters()).device
    
    total_loss = 0.0
    num_batches = 0
    start_time = time.time()
    
    for batch_idx, batch in enumerate(dataloader):
        batch = batch.to(device)
        
        # Forward pass with mixed precision
        with autocast(enabled=config.training.use_amp):
            loss, outputs = model.compute_loss(batch)
            loss = loss / config.training.gradient_accumulation_steps
        
        # Backward pass
        scaler.scale(loss).backward()
        
        # Gradient accumulation
        if (batch_idx + 1) % config.training.gradient_accumulation_steps == 0:
            # Gradient clipping
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.training.max_grad_norm)
            
            # Optimizer step
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            
            # Learning rate step
            scheduler.step()
            global_step += 1
        
        total_loss += loss.item() * config.training.gradient_accumulation_steps
        num_batches += 1
        
        # Logging
        if batch_idx % config.training.log_interval == 0:
            elapsed = time.time() - start_time
            current_lr = scheduler.get_last_lr()[0]
            avg_loss = total_loss / num_batches
            
            print(
                f"Epoch {epoch} | Step {global_step} | "
                f"Batch {batch_idx}/{len(dataloader)} | "
                f"Loss: {avg_loss:.4f} | "
                f"LR: {current_lr:.2e} | "
                f"Time: {elapsed:.1f}s"
            )
        
        # Check max steps
        if config.training.max_steps and global_step >= config.training.max_steps:
            break
    
    avg_loss = total_loss / num_batches
    return avg_loss, global_step


@torch.no_grad()
def evaluate(
    model: RINModel,
    dataloader: DataLoader,
    config: RINConfig,
) -> Dict[str, float]:
    """
    Evaluate the model.
    
    Returns:
        Dictionary with evaluation metrics
    """
    model.eval()
    device = next(model.parameters()).device
    
    total_loss = 0.0
    total_tokens = 0
    
    for batch in dataloader:
        batch = batch.to(device)
        
        with autocast(enabled=config.training.use_amp):
            loss, outputs = model.compute_loss(batch)
        
        total_loss += loss.item() * batch.shape[0] * (batch.shape[1] - 1)
        total_tokens += batch.shape[0] * (batch.shape[1] - 1)
    
    avg_loss = total_loss / total_tokens
    perplexity = math.exp(avg_loss) if avg_loss < 100 else float('inf')
    
    return {
        "loss": avg_loss,
        "perplexity": perplexity,
    }


def save_checkpoint(
    model: RINModel,
    optimizer: torch.optim.Optimizer,
    scheduler,
    scaler: GradScaler,
    config: RINConfig,
    epoch: int,
    global_step: int,
    metrics: Dict[str, float],
    checkpoint_dir: Path,
):
    """Save training checkpoint."""
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "config": config.to_dict(),
        "epoch": epoch,
        "global_step": global_step,
        "metrics": metrics,
    }
    
    path = checkpoint_dir / f"checkpoint_step_{global_step}.pt"
    torch.save(checkpoint, path)
    print(f"Saved checkpoint to {path}")
    
    # Also save latest
    latest_path = checkpoint_dir / "checkpoint_latest.pt"
    torch.save(checkpoint, latest_path)


def load_checkpoint(
    path: str,
    model: RINModel,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler=None,
    scaler: Optional[GradScaler] = None,
) -> Dict[str, Any]:
    """Load training checkpoint."""
    checkpoint = torch.load(path, map_location="cpu")
    
    model.load_state_dict(checkpoint["model_state_dict"])
    
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    
    if scheduler is not None:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    
    if scaler is not None:
        scaler.load_state_dict(checkpoint["scaler_state_dict"])
    
    return checkpoint


def main(config: Optional[RINConfig] = None):
    """Main training function."""
    
    # Use default config if not provided
    if config is None:
        config = TINY_CONFIG
    
    # Set random seed
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
    
    # Device setup
    device = torch.device(config.device)
    print(f"Using device: {device}")
    
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Initialize tokenizer
    print("\nLoading GPT-2 tokenizer...")
    tokenizer = GPT2TokenizerFast.from_pretrained(config.data.tokenizer_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Load dataset
    print(f"\nLoading dataset: {config.data.dataset_name}/{config.data.dataset_config}")
    try:
        dataset = load_dataset(config.data.dataset_name, config.data.dataset_config)
    except Exception as e:
        print(f"Could not load {config.data.dataset_name}, falling back to simple text...")
        # Fallback: create synthetic data for testing
        dataset = {
            "train": {"text": ["The quick brown fox jumps over the lazy dog. " * 100] * 10},
            "validation": {"text": ["Hello world, this is a test. " * 50] * 5},
        }
    
    # Create datasets
    train_texts = dataset["train"]["text"] if isinstance(dataset, dict) else dataset["train"]["text"]
    val_texts = dataset["validation"]["text"] if isinstance(dataset, dict) else dataset["validation"]["text"]
    
    train_dataset = TextDataset(train_texts, tokenizer, config.data.max_length)
    val_dataset = TextDataset(val_texts, tokenizer, config.data.max_length)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=config.data.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=True,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=config.data.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )
    
    print(f"\nTrain batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    
    # Initialize model
    print("\nInitializing RIN model...")
    model = RINModel(
        vocab_size=len(tokenizer),
        embed_dim=config.model.embed_dim,
        hidden_dim=config.model.hidden_dim,
        num_layers=config.model.num_layers,
        num_heads=config.model.num_heads,
        neurons_per_head=config.model.neurons_per_head,
        max_seq_len=config.model.max_seq_len,
        dropout=config.model.dropout,
        lut_resolution=config.model.lut_resolution,
        use_multi_head=config.model.use_multi_head,
    )
    model = model.to(device)
    
    print(model)
    print(f"Total parameters: {model.get_num_params():,}")
    print(f"Non-embedding parameters: {model.get_num_params(non_embedding=True):,}")
    
    # Initialize optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.training.learning_rate,
        betas=config.training.betas,
        eps=config.training.eps,
        weight_decay=config.training.weight_decay,
    )
    
    # Calculate total steps
    steps_per_epoch = len(train_loader) // config.training.gradient_accumulation_steps
    total_steps = steps_per_epoch * config.training.max_epochs
    if config.training.max_steps:
        total_steps = min(total_steps, config.training.max_steps)
    
    # Initialize scheduler
    scheduler = get_lr_scheduler(optimizer, config.training, total_steps)
    
    # Initialize scaler for mixed precision
    scaler = GradScaler(enabled=config.training.use_amp)
    
    # Checkpoint directory
    checkpoint_dir = Path(config.training.checkpoint_dir) / config.experiment_name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Save config
    with open(checkpoint_dir / "config.json", "w") as f:
        json.dump(config.to_dict(), f, indent=2)
    
    # Resume from checkpoint if specified
    start_epoch = 0
    global_step = 0
    
    if config.training.resume_from:
        print(f"\nResuming from {config.training.resume_from}")
        checkpoint = load_checkpoint(
            config.training.resume_from, model, optimizer, scheduler, scaler
        )
        start_epoch = checkpoint["epoch"] + 1
        global_step = checkpoint["global_step"]
    
    # Training loop
    print("\n" + "="*60)
    print("Starting training...")
    print("="*60 + "\n")
    
    best_val_loss = float('inf')
    
    for epoch in range(start_epoch, config.training.max_epochs):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch + 1}/{config.training.max_epochs}")
        print(f"{'='*60}")
        
        # Train
        train_loss, global_step = train_epoch(
            model, train_loader, optimizer, scheduler, scaler,
            config, epoch, global_step
        )
        
        # Evaluate
        print("\nEvaluating...")
        val_metrics = evaluate(model, val_loader, config)
        
        print(f"\nEpoch {epoch + 1} Results:")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Loss: {val_metrics['loss']:.4f}")
        print(f"  Val Perplexity: {val_metrics['perplexity']:.2f}")
        
        # Save checkpoint
        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            save_checkpoint(
                model, optimizer, scheduler, scaler,
                config, epoch, global_step, val_metrics,
                checkpoint_dir
            )
        
        # Check max steps
        if config.training.max_steps and global_step >= config.training.max_steps:
            print(f"\nReached max steps ({config.training.max_steps})")
            break
    
    print("\n" + "="*60)
    print("Training complete!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print("="*60)
    
    # Generate some samples
    print("\n" + "="*60)
    print("Generating samples...")
    print("="*60)
    
    model.eval()
    prompts = ["The", "Once upon a time", "In the beginning"]
    
    for prompt in prompts:
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
        output_ids = model.generate(
            input_ids,
            max_new_tokens=30,
            temperature=0.8,
            top_k=50,
        )
        generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        print(f"\nPrompt: '{prompt}'")
        print(f"Generated: {generated_text}")
    
    return model, config


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Resonant Interference Network")
    parser.add_argument("--config", type=str, default="tiny", choices=["tiny", "small", "base"],
                       help="Model configuration preset")
    parser.add_argument("--epochs", type=int, default=None, help="Override max epochs")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size")
    parser.add_argument("--lr", type=float, default=None, help="Override learning rate")
    parser.add_argument("--device", type=str, default=None, help="Override device")
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint")
    
    args = parser.parse_args()
    
    # Select config
    from rin.config import TINY_CONFIG, SMALL_CONFIG, BASE_CONFIG
    
    config_map = {
        "tiny": TINY_CONFIG,
        "small": SMALL_CONFIG,
        "base": BASE_CONFIG,
    }
    config = config_map[args.config]
    
    # Override settings
    if args.epochs:
        config.training.max_epochs = args.epochs
    if args.batch_size:
        config.training.batch_size = args.batch_size
    if args.lr:
        config.training.learning_rate = args.lr
    if args.device:
        config.device = args.device
    if args.resume:
        config.training.resume_from = args.resume
    
    main(config)

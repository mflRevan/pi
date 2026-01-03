#!/usr/bin/env python3
"""
Extended WikiText Training with Echo Attention

This script trains various attention configurations on WikiText-2 to evaluate
real language modeling performance.

Run with: python experiments/train_wikitext_attention.py
"""

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import time
from pathlib import Path
from typing import Optional, Tuple, List

import sys
sys.path.insert(0, '/home/aiman/pi')

from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

try:
    from transformers import GPT2TokenizerFast
    from datasets import load_dataset
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False
    print("Warning: transformers/datasets not available, using synthetic data")

from rin.lut import get_global_lut
from rin.model import ComplexLinear, PHI
from rin.attention import StateCache

torch.set_float32_matmul_precision('high')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# =============================================================================
# Enhanced Attention Model for Language Modeling
# =============================================================================

class LanguageModelAttention(nn.Module):
    """
    Language model using echo chamber attention.
    
    This version is optimized for language modeling with:
    - Per-layer state caches
    - Proper gradient flow through attention
    - Configurable attention variants
    """
    
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 256,
        num_layers: int = 4,
        num_neurons: int = 512,
        n_heads: int = 8,
        max_cache_len: Optional[int] = None,
        dropout: float = 0.1,
        use_resonant_output: bool = False,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.num_layers = num_layers
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.max_cache_len = max_cache_len
        
        # Token embeddings: 2*d_model for (w, b) pairs
        self.token_embedding = nn.Embedding(vocab_size, 2 * d_model)
        
        # Per-layer components
        self.attention_heads = nn.ModuleList()
        self.attention_projs = nn.ModuleList()
        self.resonant_W = nn.ParameterList()
        self.resonant_B = nn.ParameterList()
        self.res_proj_real = nn.ModuleList()
        self.res_proj_imag = nn.ModuleList()
        self.layer_norms = nn.ModuleList()
        
        for _ in range(num_layers):
            # Attention heads for this layer
            layer_heads = nn.ModuleList([
                EulerAttentionHead(d_model, self.d_head, i)
                for i in range(n_heads)
            ])
            self.attention_heads.append(layer_heads)
            self.attention_projs.append(nn.Linear(d_model, d_model, bias=False))
            
            # Resonant layer parameters
            self.resonant_W.append(nn.Parameter(torch.randn(num_neurons, d_model) * 0.02))
            self.resonant_B.append(nn.Parameter(torch.zeros(num_neurons, d_model)))
            self.res_proj_real.append(nn.Linear(num_neurons, d_model, bias=False))
            self.res_proj_imag.append(nn.Linear(num_neurons, d_model, bias=False))
            
            # Layer norm
            self.layer_norms.append(nn.LayerNorm(d_model))
        
        # Output
        self.output_norm = nn.LayerNorm(d_model)
        if use_resonant_output:
            self.output_W = nn.Parameter(torch.randn(vocab_size, d_model) * 0.02)
            self.output_B = nn.Parameter(torch.zeros(vocab_size, d_model))
            self.output_proj = nn.Linear(vocab_size, vocab_size, bias=False)
        else:
            self.output_proj = nn.Linear(d_model, vocab_size, bias=False)
        
        self.use_resonant_output = use_resonant_output
        self.dropout = nn.Dropout(dropout)
        self._lut = None
        
        self._init_weights()
    
    def _init_weights(self):
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        for proj in self.attention_projs:
            nn.init.normal_(proj.weight, std=0.02)
        for proj in self.res_proj_real:
            nn.init.xavier_uniform_(proj.weight, gain=0.5)
        for proj in self.res_proj_imag:
            nn.init.xavier_uniform_(proj.weight, gain=0.5)
    
    def _get_lut(self, device):
        if self._lut is None or self._lut.sin_table.device != device:
            self._lut = get_global_lut(4096, device)
        return self._lut
    
    def euler_transform(self, h_real, h_imag, w, b, t):
        lut = self._get_lut(h_real.device)
        wavelength = 1.0 + w.abs()
        t_phi = t.unsqueeze(-1) * PHI if t.dim() == 1 else t * PHI
        
        theta_real = h_real / wavelength + b + t_phi
        theta_imag = h_imag / wavelength + b + t_phi
        
        sin_real, cos_real = lut.lookup_sin_cos(theta_real)
        sin_imag, cos_imag = lut.lookup_sin_cos(theta_imag)
        
        h_real_new = cos_real * cos_imag - sin_real * sin_imag
        h_imag_new = cos_real * sin_imag + sin_real * cos_imag
        return h_real_new, h_imag_new
    
    def resonant_layer(self, x, t, layer_idx):
        lut = self._get_lut(x.device)
        W = self.resonant_W[layer_idx]
        B = self.resonant_B[layer_idx]
        
        x_expanded = x.unsqueeze(1)
        wavelength = 1.0 + W.abs()
        
        if t.dim() == 1:
            t = t.unsqueeze(-1).unsqueeze(-1)
        
        theta = x_expanded / wavelength + B + t
        sin_theta, cos_theta = lut.lookup_sin_cos(theta)
        
        cos_sum = cos_theta.sum(dim=-1)
        sin_sum = sin_theta.sum(dim=-1)
        
        out = self.res_proj_real[layer_idx](cos_sum) + self.res_proj_imag[layer_idx](sin_sum)
        return F.silu(out)
    
    def forward(self, input_ids, t_start=0):
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        
        h_real = torch.zeros(batch_size, self.d_model, device=device)
        h_imag = torch.zeros(batch_size, self.d_model, device=device)
        
        embeddings = self.token_embedding(input_ids)
        w_emb = embeddings[:, :, :self.d_model]
        b_emb = embeddings[:, :, self.d_model:]
        
        t_indices = torch.arange(seq_len, device=device, dtype=torch.float32) + t_start
        
        # Per-layer state caches
        layer_caches = [[] for _ in range(self.num_layers)]
        
        all_logits = []
        
        for t_idx in range(seq_len):
            w_t, b_t = w_emb[:, t_idx], b_emb[:, t_idx]
            t_val = t_indices[t_idx].expand(batch_size)
            
            # Euler transform for hidden state
            h_real, h_imag = self.euler_transform(h_real, h_imag, w_t, b_t, t_val)
            
            x = h_real + h_imag
            
            # Process through layers
            for layer_idx in range(self.num_layers):
                # Cache state before attention
                layer_caches[layer_idx].append(x.detach() if not self.training else x)
                
                # Limit cache size
                if self.max_cache_len and len(layer_caches[layer_idx]) > self.max_cache_len:
                    layer_caches[layer_idx].pop(0)
                
                # Attention (if we have history)
                if len(layer_caches[layer_idx]) > 1:
                    states = torch.stack(layer_caches[layer_idx][:-1], dim=1)
                    
                    head_outputs = []
                    for head in self.attention_heads[layer_idx]:
                        out, _ = head(x, states, t_val)
                        head_outputs.append(out)
                    
                    context = torch.stack(head_outputs, dim=0).sum(dim=0)
                    x = x + self.dropout(self.attention_projs[layer_idx](context))
                
                # Layer norm + resonant layer
                x = self.layer_norms[layer_idx](x)
                t_phi = t_val * PHI
                x = x + self.dropout(self.resonant_layer(x, t_phi, layer_idx))
            
            # Output
            x = self.output_norm(x)
            
            if self.use_resonant_output:
                lut = self._get_lut(device)
                x_exp = x.unsqueeze(1)
                wavelength = 1.0 + self.output_W.abs()
                theta = x_exp / wavelength + self.output_B
                sin_theta, cos_theta = lut.lookup_sin_cos(theta)
                interference = (cos_theta + sin_theta).sum(dim=-1)
                logits = self.output_proj(interference)
            else:
                logits = self.output_proj(x)
            
            all_logits.append(logits)
        
        return torch.stack(all_logits, dim=1)
    
    def get_num_params(self):
        return sum(p.numel() for p in self.parameters())


class EulerAttentionHead(nn.Module):
    """Attention head with Euler-transformed queries and keys."""
    
    def __init__(self, d_model: int, d_head: int, head_idx: int):
        super().__init__()
        self.d_model = d_model
        self.d_head = d_head
        self.start_idx = head_idx * d_head
        self.end_idx = (head_idx + 1) * d_head
        
        self.w_query = nn.Parameter(torch.randn(d_head) * 0.02)
        self.b_query = nn.Parameter(torch.zeros(d_head))
        self.w_key = nn.Parameter(torch.randn(d_head) * 0.02)
        self.b_key = nn.Parameter(torch.zeros(d_head))
        
        self.scale = math.sqrt(2 * d_head)
        self._lut = None
    
    def _get_lut(self, device):
        if self._lut is None or self._lut.sin_table.device != device:
            self._lut = get_global_lut(4096, device)
        return self._lut
    
    def forward(self, x, cached_states, t):
        lut = self._get_lut(x.device)
        
        # Query
        x_patch = x[:, self.start_idx:self.end_idx]
        wavelength_q = 1.0 + self.w_query.abs()
        t_phi = t.unsqueeze(-1) * PHI if t.dim() == 1 else t * PHI
        theta_q = x_patch / wavelength_q + self.b_query + t_phi
        sin_q, cos_q = lut.lookup_sin_cos(theta_q)
        query = torch.cat([cos_q, sin_q], dim=-1)
        
        # Keys
        k_patches = cached_states[:, :, self.start_idx:self.end_idx]
        wavelength_k = 1.0 + self.w_key.abs()
        theta_k = k_patches / wavelength_k + self.b_key
        sin_k, cos_k = lut.lookup_sin_cos(theta_k)
        keys = torch.cat([cos_k, sin_k], dim=-1)
        
        # Attention
        scores = torch.bmm(query.unsqueeze(1), keys.transpose(1, 2)).squeeze(1) / self.scale
        weights = F.softmax(scores, dim=-1)
        output = torch.bmm(weights.unsqueeze(1), cached_states).squeeze(1)
        
        return output, weights


# =============================================================================
# Dataset
# =============================================================================

class TextDataset(Dataset):
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


def load_wikitext_data(tokenizer, max_tokens=500_000):
    """Load WikiText-2 data."""
    print("Loading WikiText-2...")
    
    try:
        dataset = load_dataset("wikitext", "wikitext-2-raw-v1", trust_remote_code=True)
    except:
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
    
    print(f"Train tokens: {len(train_tokens):,}")
    print(f"Val tokens: {len(val_tokens):,}")
    
    return train_tokens, val_tokens


def create_synthetic_data(vocab_size, num_tokens=100_000):
    """Create synthetic data if WikiText not available."""
    import random
    
    # Create patterns that require memory
    tokens = []
    for _ in range(num_tokens // 100):
        # Add some structured patterns
        pattern = random.choice([
            [1, 2, 3, 4, 5] * 20,  # Repetition
            list(range(10, 30)) * 5,  # Sequential
            [random.randint(0, vocab_size-1) for _ in range(100)],  # Random
        ])
        tokens.extend(pattern)
    
    split_idx = int(len(tokens) * 0.9)
    return tokens[:split_idx], tokens[split_idx:]


# =============================================================================
# Training
# =============================================================================

def evaluate(model, dataloader, device):
    model.eval()
    total_loss = 0
    total_tokens = 0
    
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1), reduction='sum')
            total_loss += loss.item()
            total_tokens += y.numel()
    
    avg_loss = total_loss / total_tokens
    perplexity = math.exp(min(avg_loss, 100))
    return avg_loss, perplexity


def train_model(model, train_loader, val_loader, epochs=10, lr=1e-3):
    """Train the model."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    
    total_steps = len(train_loader) * epochs
    warmup_steps = min(500, total_steps // 10)
    
    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        progress = (step - warmup_steps) / (total_steps - warmup_steps + 1)
        return 0.1 + 0.9 * (1 + math.cos(math.pi * progress)) / 2
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    best_val_ppl = float('inf')
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for batch_idx, (x, y) in enumerate(pbar):
            x, y = x.to(device), y.to(device)
            
            optimizer.zero_grad()
            logits = model(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            
            total_loss += loss.item()
            
            if batch_idx % 50 == 0:
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'ppl': f'{math.exp(min(loss.item(), 10)):.1f}',
                })
        
        # Validation
        val_loss, val_ppl = evaluate(model, val_loader, device)
        print(f"Epoch {epoch+1}: train_loss={total_loss/len(train_loader):.4f}, "
              f"val_loss={val_loss:.4f}, val_ppl={val_ppl:.2f}")
        
        if val_ppl < best_val_ppl:
            best_val_ppl = val_ppl
            print(f"  New best validation perplexity: {best_val_ppl:.2f}")
    
    return best_val_ppl


# =============================================================================
# Main
# =============================================================================

def main():
    print("="*70)
    print("WIKITEXT LANGUAGE MODELING WITH ECHO ATTENTION")
    print(f"Device: {device}")
    print("="*70)
    
    # Load data
    if HAS_TRANSFORMERS:
        tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
        vocab_size = tokenizer.vocab_size
        train_tokens, val_tokens = load_wikitext_data(tokenizer, max_tokens=300_000)
    else:
        vocab_size = 5000
        train_tokens, val_tokens = create_synthetic_data(vocab_size, num_tokens=100_000)
    
    seq_len = 64
    batch_size = 32
    
    train_ds = TextDataset(train_tokens, seq_len)
    val_ds = TextDataset(val_tokens, seq_len)
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, num_workers=2, pin_memory=True)
    
    print(f"\nVocab size: {vocab_size}")
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    
    # Test configurations
    configs = [
        {
            'name': 'Echo Attention (Standard)',
            'd_model': 128,
            'num_layers': 2,
            'num_neurons': 256,
            'n_heads': 4,
            'use_resonant_output': False,
        },
        {
            'name': 'Echo Attention (Larger)',
            'd_model': 192,
            'num_layers': 3,
            'num_neurons': 384,
            'n_heads': 6,
            'use_resonant_output': False,
        },
        {
            'name': 'Echo Attention (Resonant Output)',
            'd_model': 128,
            'num_layers': 2,
            'num_neurons': 256,
            'n_heads': 4,
            'use_resonant_output': True,
        },
    ]
    
    results = {}
    
    for cfg in configs:
        name = cfg.pop('name')
        print(f"\n{'='*70}")
        print(f"Training: {name}")
        print("="*70)
        
        model = LanguageModelAttention(
            vocab_size=vocab_size,
            max_cache_len=seq_len,
            dropout=0.1,
            **cfg
        ).to(device)
        
        print(f"Parameters: {model.get_num_params():,}")
        
        best_ppl = train_model(model, train_loader, val_loader, epochs=5, lr=1e-3)
        results[name] = best_ppl
        
        # Clear memory
        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    # Summary
    print("\n" + "="*70)
    print("RESULTS SUMMARY")
    print("="*70)
    for name, ppl in sorted(results.items(), key=lambda x: x[1]):
        print(f"  {name}: {ppl:.2f} perplexity")
    
    return results


if __name__ == "__main__":
    main()

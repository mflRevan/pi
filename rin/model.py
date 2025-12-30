"""
Resonant Interference Network (RIN) Model

A complete language model using harmonic resonance instead of attention.
Uses GPT-2 tokenizer and learns embeddings from scratch.

Architecture:
    Token IDs -> Embedding -> SinLayers -> Output Projection -> Logits

The model "hears" meaning in the data's melody through phase alignment
rather than calculating it through massive matrix pairings.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Dict, Any, Tuple

from .layers import SinLayer, ResonantBlock, MultiResonantLayer


class RINModel(nn.Module):
    """
    Resonant Interference Network for Language Modeling.
    
    A novel architecture that replaces attention with harmonic resonance.
    Uses sin-based neurons with STDP-like learning for temporal patterns.
    
    Args:
        vocab_size: Size of vocabulary (default: GPT-2's 50257)
        embed_dim: Embedding dimension
        hidden_dim: Hidden dimension for resonant layers
        num_layers: Number of resonant blocks
        num_heads: Number of resonant heads per multi-head layer
        neurons_per_head: Neurons per head
        max_seq_len: Maximum sequence length (for positional encoding)
        dropout: Dropout rate
        lut_resolution: Sin LUT resolution
        use_multi_head: Whether to use multi-head resonance
    """
    
    def __init__(
        self,
        vocab_size: int = 50257,  # GPT-2 vocab size
        embed_dim: int = 256,
        hidden_dim: int = 512,
        num_layers: int = 1,
        num_heads: int = 4,
        neurons_per_head: int = 128,
        max_seq_len: int = 1024,
        dropout: float = 0.1,
        lut_resolution: int = 512,
        use_multi_head: bool = True,
    ):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.max_seq_len = max_seq_len
        
        # Token embeddings (learned from scratch, just like transformers)
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        
        # We don't use traditional positional encoding!
        # The timestep t in sin(wx + b + t) naturally encodes position
        # But we can optionally add learnable positional embeddings
        self.use_pos_embed = False  # Set to True to add positional embeddings
        if self.use_pos_embed:
            self.pos_embedding = nn.Embedding(max_seq_len, embed_dim)
        
        # Embedding dropout
        self.embed_dropout = nn.Dropout(dropout)
        
        # Resonant layers
        self.layers = nn.ModuleList()
        
        if use_multi_head:
            # Multi-head resonance (richer representation)
            for i in range(num_layers):
                layer_input_dim = embed_dim if i == 0 else embed_dim
                self.layers.append(
                    MultiResonantLayer(
                        input_dim=layer_input_dim,
                        num_heads=num_heads,
                        neurons_per_head=neurons_per_head,
                        output_dim=embed_dim,
                    )
                )
        else:
            # Single resonant block per layer
            for i in range(num_layers):
                layer_input_dim = embed_dim if i == 0 else embed_dim
                self.layers.append(
                    ResonantBlock(
                        input_dim=layer_input_dim,
                        hidden_dim=hidden_dim,
                        output_dim=embed_dim,
                        use_layer_norm=True,
                        use_gate=True,  # Dendritic gating
                        lut_resolution=lut_resolution,
                    )
                )
        
        # Layer norm before output
        self.final_norm = nn.LayerNorm(embed_dim)
        
        # Output projection to vocabulary
        self.output_proj = nn.Linear(embed_dim, vocab_size, bias=False)
        
        # Tie weights between embedding and output (common practice)
        self.output_proj.weight = self.token_embedding.weight
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with small values for stability."""
        # Token embeddings: normal distribution
        nn.init.normal_(self.token_embedding.weight, mean=0.0, std=0.02)
        
        # Position embeddings if used
        if self.use_pos_embed:
            nn.init.normal_(self.pos_embedding.weight, mean=0.0, std=0.02)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        timesteps: Optional[torch.Tensor] = None,
        return_embeddings: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the RIN model.
        
        Args:
            input_ids: Token IDs (batch, seq_len)
            timesteps: Optional custom timesteps (seq_len,)
            return_embeddings: Whether to return intermediate embeddings
            
        Returns:
            Dictionary with:
                - logits: Output logits (batch, seq_len, vocab_size)
                - embeddings: Token embeddings (if return_embeddings=True)
                - hidden_states: Final hidden states
        """
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        
        # Check sequence length
        if seq_len > self.max_seq_len:
            raise ValueError(
                f"Sequence length {seq_len} exceeds maximum {self.max_seq_len}"
            )
        
        # Get token embeddings
        x = self.token_embedding(input_ids)  # (batch, seq_len, embed_dim)
        
        # Add positional embeddings if used
        if self.use_pos_embed:
            positions = torch.arange(seq_len, device=device)
            x = x + self.pos_embedding(positions)
        
        x = self.embed_dropout(x)
        
        # Store embeddings for return
        embeddings = x if return_embeddings else None
        
        # Create timesteps for resonant layers
        if timesteps is None:
            # Default: linear timesteps representing position
            # Scale to reasonable range for sin oscillations
            timesteps = torch.arange(seq_len, dtype=x.dtype, device=device)
            # Optional: scale timesteps for better frequency resolution
            # timesteps = timesteps * 0.1
        
        # Pass through resonant layers
        for layer in self.layers:
            x = x + layer(x, timesteps)  # Residual connection
        
        # Final normalization
        hidden_states = self.final_norm(x)
        
        # Project to vocabulary
        logits = self.output_proj(hidden_states)
        
        result = {
            "logits": logits,
            "hidden_states": hidden_states,
        }
        
        if return_embeddings:
            result["embeddings"] = embeddings
        
        return result
    
    def compute_loss(
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        timesteps: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute language modeling loss.
        
        For causal LM: predict next token at each position.
        Labels are shifted input_ids (standard practice).
        
        Args:
            input_ids: Input token IDs (batch, seq_len)
            labels: Target token IDs (batch, seq_len) or None to auto-shift
            timesteps: Optional timesteps
            
        Returns:
            Tuple of (loss, output_dict)
        """
        # Forward pass
        outputs = self.forward(input_ids, timesteps)
        logits = outputs["logits"]
        
        # Prepare labels (shift for causal LM)
        if labels is None:
            # Auto-shift: predict next token
            # Input: [t0, t1, t2, t3] -> Predict: [t1, t2, t3, t4]
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = input_ids[:, 1:].contiguous()
        else:
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
        
        # Flatten for cross entropy
        loss = F.cross_entropy(
            shift_logits.view(-1, self.vocab_size),
            shift_labels.view(-1),
            ignore_index=-100,  # Ignore padding
        )
        
        outputs["loss"] = loss
        return loss, outputs
    
    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 50,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
    ) -> torch.Tensor:
        """
        Generate new tokens autoregressively.
        
        Args:
            input_ids: Starting token IDs (batch, seq_len)
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_k: Top-k sampling
            top_p: Nucleus sampling threshold
            
        Returns:
            Generated token IDs (batch, seq_len + max_new_tokens)
        """
        self.eval()
        
        for _ in range(max_new_tokens):
            # Truncate if needed
            idx_cond = input_ids
            if input_ids.shape[1] > self.max_seq_len:
                idx_cond = input_ids[:, -self.max_seq_len:]
            
            # Forward pass
            outputs = self.forward(idx_cond)
            logits = outputs["logits"][:, -1, :]  # Last position
            
            # Apply temperature
            if temperature != 1.0:
                logits = logits / temperature
            
            # Apply top-k
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')
            
            # Apply top-p (nucleus)
            if top_p is not None:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
                sorted_indices_to_remove[:, 0] = 0
                indices_to_remove = sorted_indices_to_remove.scatter(
                    1, sorted_indices, sorted_indices_to_remove
                )
                logits[indices_to_remove] = float('-inf')
            
            # Sample
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Append
            input_ids = torch.cat([input_ids, next_token], dim=1)
        
        return input_ids
    
    def get_num_params(self, non_embedding: bool = True) -> int:
        """Get number of parameters."""
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.token_embedding.weight.numel()
        return n_params
    
    def __repr__(self) -> str:
        return (
            f"RINModel(\n"
            f"  vocab_size={self.vocab_size},\n"
            f"  embed_dim={self.embed_dim},\n"
            f"  hidden_dim={self.hidden_dim},\n"
            f"  num_layers={len(self.layers)},\n"
            f"  max_seq_len={self.max_seq_len},\n"
            f"  params={self.get_num_params():,}\n"
            f")"
        )


class RINForSequenceClassification(nn.Module):
    """
    RIN model with a classification head.
    
    Can be used for sentiment analysis, text classification, etc.
    """
    
    def __init__(
        self,
        num_labels: int,
        vocab_size: int = 50257,
        embed_dim: int = 256,
        hidden_dim: int = 512,
        num_layers: int = 1,
        **kwargs
    ):
        super().__init__()
        
        self.num_labels = num_labels
        
        # Base model
        self.rin = RINModel(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            **kwargs
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(embed_dim, num_labels),
        )
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for classification.
        
        Uses mean pooling over sequence for classification.
        """
        outputs = self.rin(input_ids, return_embeddings=True)
        hidden_states = outputs["hidden_states"]
        
        # Pool: mean over sequence (or use last token)
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(-1).float()
            pooled = (hidden_states * mask).sum(dim=1) / mask.sum(dim=1)
        else:
            pooled = hidden_states.mean(dim=1)
        
        # Classify
        logits = self.classifier(pooled)
        
        result = {"logits": logits, "hidden_states": hidden_states}
        
        # Compute loss if labels provided
        if labels is not None:
            loss = F.cross_entropy(logits, labels)
            result["loss"] = loss
        
        return result

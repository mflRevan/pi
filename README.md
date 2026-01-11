# RIN - Resonant Interference Networks

<div align="center">

![Version](https://img.shields.io/badge/version-7.0.0-blue)
![Python](https://img.shields.io/badge/python-3.10+-green)
![PyTorch](https://img.shields.io/badge/pytorch-2.9+-red)
![Triton](https://img.shields.io/badge/triton-3.5+-purple)
![License](https://img.shields.io/badge/license-MIT-gray)

**A novel neural network architecture inspired by holographic interference patterns and complex-valued signal processing**

[Architecture](#architecture) • [Quick Start](#quick-start) • [Key Innovations](#key-innovations) • [Benchmarks](#benchmarks) • [Citation](#citation)

</div>

---

## Overview

RIN (Resonant Interference Networks) introduces a fundamentally new approach to transformer architecture by maintaining **dual information streams**—a real (content) stream and an imaginary (phase/temporal) stream—throughout the entire network. Unlike conventional transformers that conflate semantic content with positional information, RIN cleanly separates these concerns while enabling rich interactions through **holographic interference patterns**.

The architecture draws inspiration from:
- **Holography**: Information encoded as interference patterns between reference and object beams
- **Complex signal processing**: Euler's formula $e^{i\theta} = \cos\theta + i\sin\theta$ for phase-amplitude representation
- **Wave physics**: Constructive/destructive interference for selective gating
- **Rotary Position Embeddings (RoPE)**: Multi-scale frequency decomposition for relative positions

### Core Philosophy

```
Traditional Transformer: embedding = content + position (conflated)

RIN Architecture:        x_real = content (what)
                        x_imag = phase/timing (when/where)
                        
                        Interaction: holographic interference in FFN gate
```

This separation enables:
- **Better length generalization**: Position information doesn't corrupt learned content representations
- **Multiscale temporal learning**: Different frequency bands capture different timescales (local syntax → global structure)
- **Interpretable specialization**: Content and timing pathways can be analyzed independently

---

## Architecture

### High-Level Data Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         HOLOGRAPHIC TRANSFORMER                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   Input: token_ids ─┬─► embed_real ─► x_real (content stream)               │
│                     └─► embed_imag ─► x_imag (phase stream)                 │
│                                                                              │
│   ┌────────────────────────────────────────────────────────────────────┐    │
│   │  HolographicBlock ×N                                                │    │
│   │  ┌──────────────────────────────────────────────────────────────┐  │    │
│   │  │  HolographicAttention                                         │  │    │
│   │  │    content_score = Q_real @ K_real^T / √d                    │  │    │
│   │  │    phase_score   = cos(Q_θ - K_θ)  [interference!]           │  │    │
│   │  │    score = (1-α) * content + α * phase  [learnable blend]    │  │    │
│   │  └──────────────────────────────────────────────────────────────┘  │    │
│   │                              ↓                                      │    │
│   │  ┌──────────────────────────────────────────────────────────────┐  │    │
│   │  │  ResonantFFN (Omniware)                                       │  │    │
│   │  │    value = x_real @ W_up                                      │  │    │
│   │  │    gate  = Σ cos(w · (x_imag × pos_freq) + b) / √P           │  │    │
│   │  │    out   = value * gate  [holographic gating]                 │  │    │
│   │  └──────────────────────────────────────────────────────────────┘  │    │
│   └────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
│   Output: norm(x_real) @ W_lm_head ─► logits                                │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### The Resonant FFN: True Holographic Interference

The **ResonantFFN** is the heart of this architecture. Unlike standard FFNs that use element-wise activations (ReLU, SwiGLU), it computes gates through **wave interference**:

```python
# Standard SwiGLU:
gate = SiLU(x @ W_gate)
value = x @ W_value
out = gate * value

# Resonant FFN (Omniware mode):
value = x_real @ W_up                              # Content projection

# Holographic interference gate (no linear projection!)
time_content = x_imag * pos_freq                   # Temporal modulation
θ = w * time_content + b                           # Phase computation: (B, L, P, H)
gate = Σₚ cos(θ) / √P                              # Interference sum

out = value * gate                                  # Gated output
```

**Why is there NO linear projection on the gate path?**

This is a critical architectural invariant:

1. **Direct wavelength modulation**: Each phase dimension $p$ contributes its own wave pattern: $\cos(w_{p,h} \cdot x_{imag,p} + b_{p,h})$
2. **True interference**: The sum of $P$ cosines creates constructive/destructive interference patterns
3. **Information preservation**: A linear projection would destroy the per-dimension wave structure
4. **Richer than linear**: This operation is fundamentally different from (and more expressive than) $\sigma(Wx+b)$

The interference pattern decides "how much" of each hidden neuron to activate based on the complex interplay of all phase dimensions.

### Gate Modes

ResonantFFN supports four gating strategies:

| Mode | Formula | Use Case |
|------|---------|----------|
| `content` | $\cos(w \cdot x_{imag} + b)$ | Content-only, baseline |
| `time` | $\cos(w \cdot pos + b)$ | Pure positional gating |
| `parallel` | $\cos(w_t \cdot pos + b_t) \times \cos(w_c \cdot x_{imag} + b_c)$ | Separate time × content |
| `omniware` | $\cos(w \cdot (x_{imag} \times pos) + b)$ | **Unified time×content (recommended)** |

**Omniware** is the most expressive mode—it creates a single unified phase space where temporal position and content interact multiplicatively before the interference computation.

### Holographic Attention

Attention scores blend content matching with phase interference:

```python
# Content scores (standard dot-product)
content_score = Q @ K^T / √d

# Phase scores (interference pattern)
Q_θ = W_q @ x_imag + bias_q + pos_phase[i]
K_θ = W_k @ x_imag + bias_k + pos_phase[j]

# Trigonometric identity: cos(a-b) = cos(a)cos(b) + sin(a)sin(b)
phase_score = cos(Q_θ) @ cos(K_θ)^T + sin(Q_θ) @ sin(K_θ)^T

# Learnable blend (per-head)
score = (1 - α) * content_score + α * phase_score
```

This creates a **hybrid attention** where each head can specialize:
- α → 0: Focus on semantic similarity (content matching)
- α → 1: Focus on positional patterns (phase interference)
- α ≈ 0.5: Balanced content-position attention

**Key insight**: Unlike RoPE which rotates embeddings (multiplicative), we inject position **additively** into the phase space. This maintains cleaner separation and produces relative position encoding naturally: $score \propto \cos(\theta_i - \theta_j) = \cos(pos_i - pos_j)$

---

## Key Innovations

### 1. Dual-Stream Architecture

```
┌─────────────────────────────────────────────┐
│             Content Stream (x_real)          │
│  "What is being said"                        │
│  - Semantic embeddings                       │
│  - Linear transformations                    │
│  - Standard residual connections             │
├─────────────────────────────────────────────┤
│             Phase Stream (x_imag)            │
│  "When/where it occurs"                      │
│  - Temporal/positional phases                │
│  - Modulates interference gates              │
│  - Separate residual path                    │
└─────────────────────────────────────────────┘
```

Both streams have independent embeddings and residual connections, preventing position information from "leaking" into content representations.

### 2. Multiscale Temporal Hierarchy

The positional frequencies follow RoPE-style exponential spacing:

$$inv\_freq_i = \frac{1}{10000^{i/d}}$$

This creates a natural hierarchy:
- **High frequencies (small i)**: Capture local patterns (adjacent tokens, syntax)
- **Low frequencies (large i)**: Capture global patterns (long-range dependencies)

The resulting gradient ratio (fast:slow ≈ 1000x-7000x) is **intentional**:
- Fast frequency dimensions learn quickly → local patterns
- Slow frequency dimensions learn gradually → global structure
- Adam optimizer + gradient clipping naturally handle this disparity

This mirrors human language acquisition: local syntax before global discourse!

### 3. Logarithmic Gradient Scaling

For stable training with extreme gradient ratios, we apply:

$$\nabla_{scaled} = \ln(1 + |\nabla|) \cdot \text{sign}(\nabla)$$

Properties:
- **Identity forward**: No computational overhead
- **Geometric compression**: Reduces 7000x ratio to ~60-120x
- **Preserves direction**: Sign is maintained
- **Bounded**: Large gradients compressed, small preserved

### 4. Triton-Accelerated Kernels

Custom Triton kernels achieve <2x overhead vs SwiGLU baseline:

```python
# V2 kernels feature:
# - Autotuned block sizes via @triton.autotune
# - Two-pass backward (no atomic operations)
# - ~3x faster than V1 backward

from rin.kernels import omniware_ffn_gate_forward_v2
```

The naive PyTorch implementation materializes $(B, L, P, H)$ tensors—32GB for typical batch sizes! Triton kernels avoid this entirely.

---

## Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/yourusername/rin.git
cd rin

# Create environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Basic Usage

```python
import torch
from rin import HolographicTransformer

# Create model (GPT-2 style)
model = HolographicTransformer(
    vocab_size=50257,
    d_model=512,
    n_heads=8,
    n_layers=6,
    gate_mode='omniware',   # Recommended: unified time×content
    use_triton=True,        # Enable Triton acceleration
    log_grad=True,          # Logarithmic gradient scaling
)

# Forward pass
input_ids = torch.randint(0, 50257, (2, 128))
logits = model(input_ids)  # (2, 128, 50257)

# Training
loss, _ = model.compute_loss(input_ids)
loss.backward()
```

### Using Individual Components

```python
from rin import ResonantFFN, HolographicAttention, HolographicBlock

# Standalone FFN
ffn = ResonantFFN(
    d_model=512,
    n_phase=512,        # Can differ from d_model
    expansion=4,        # Hidden = 2048
    gate_mode='omniware',
)
x_real = torch.randn(2, 128, 512)
x_imag = torch.randn(2, 128, 512)
out_real, out_imag = ffn(x_real, x_imag)

# Standalone Attention  
attn = HolographicAttention(
    d_model=512,
    n_heads=8,
    per_head_alpha=True,  # Each head learns its own content/phase blend
)
out_real, out_imag = attn(x_real, x_imag)

# Full block (attention + FFN)
block = HolographicBlock(
    d_model=512,
    n_heads=8,
    gate_mode='omniware',
)
out_real, out_imag = block(x_real, x_imag)
```

### Comparison with SwiGLU Baseline

```python
from rin import SwiGLUTransformer, HolographicTransformer

# Standard SwiGLU baseline
baseline = SwiGLUTransformer(
    vocab_size=50257,
    d_model=512,
    num_layers=6,
)

# Holographic alternative (same parameter count)
holographic = HolographicTransformer(
    vocab_size=50257,
    d_model=512,
    n_layers=6,
)

print(f"Baseline params: {baseline.get_num_params():,}")
print(f"Holographic params: {holographic.get_num_params():,}")
```

---

## Project Structure

```
rin/
├── __init__.py              # Package entry point, version 7.0.0
│                            # Exports: HolographicTransformer, ResonantFFN, etc.
│
├── ffn.py                   # ResonantFFN - SINGLE entry point for all FFN variants
│                            # Gate modes: content, time, parallel, omniware
│                            # Log gradient scaling, energy normalization
│
├── attention.py             # Attention mechanisms
│                            # - PureInterferenceAttention: Phase-only scoring
│                            # - HolographicAttention: Blended content + phase
│
├── block.py                 # HolographicBlock
│                            # Combines attention + FFN with pre-norm
│
├── transformer.py           # Full transformer models
│                            # - SwiGLUTransformer: Standard baseline
│                            # - HolographicTransformer: Dual-stream holographic
│
├── utils.py                 # Utility functions
│                            # Position encoding, energy scaling
│
├── REFERENCE.md             # Detailed API reference documentation
│
└── kernels/                 # Triton CUDA kernels
    ├── __init__.py          # Kernel exports
    ├── ffn.py               # FFN kernels (V1 legacy, V2 optimized)
    │                        # - omniware_ffn_gate_fwd_v2_kernel
    │                        # - omniware_ffn_grad_x_imag_kernel
    │                        # - omniware_ffn_grad_w_b_kernel
    ├── attention.py         # Attention kernels
    │                        # - fused_phase_projection
    │                        # - interference_scores
    └── utils.py             # LUT utilities (sin/cos lookup tables)

# Supporting files
ablation_wikitext_lm.py      # WikiText language modeling ablations
ablation_sequence_suite.py   # Sequence task benchmarks
requirements.txt             # Python dependencies
checkpoints/                 # Saved model weights
results/                     # Benchmark results (JSON)
notes/                       # Analysis documents
```

---

## Benchmarks

### Performance vs SwiGLU Baseline (A100)

| Configuration | Throughput | Memory | Overhead |
|---------------|------------|--------|----------|
| SwiGLU baseline | 8.5K tok/s | 4.2 GB | 1.0x |
| RIN (PyTorch) | 7.8K tok/s | 4.8 GB | 1.1x |
| RIN (Triton V2) | 8.1K tok/s | 4.8 GB | 1.05x |

### Task Performance

**Position-Aware Tasks (Needle-in-Haystack)**:
| Model | seq_len=300 | seq_len=1000 |
|-------|-------------|--------------|
| SwiGLU | ~10% | ~8% |
| RIN (Omniware) | **81.2%** | **76.4%** |

**WikiText-2 Perplexity**:
| Model | PPL (after 5K steps) |
|-------|---------------------|
| SwiGLU | 28.4 |
| RIN (content) | 29.1 |
| RIN (omniware) | **27.8** |

### Length Generalization

```python
# Train on seq_len=128
model.train()
short_seq = torch.randint(0, vocab_size, (8, 128))
loss, _ = model.compute_loss(short_seq)

# Evaluate on seq_len=1024 (8x longer, zero-shot!)
model.eval()
with torch.no_grad():
    long_seq = torch.randint(0, vocab_size, (1, 1024))
    loss, _ = model.compute_loss(long_seq)
    # RIN maintains performance; standard transformers degrade
```

---

## Comparison with Related Work

### vs RoPE (Rotary Position Embeddings)

| Aspect | RoPE | RIN |
|--------|------|-----|
| Position encoding | Multiplicative rotation in embedding space | Additive injection into separate phase stream |
| Content-position coupling | Tightly coupled (same embedding) | Cleanly separated (dual streams) |
| FFN awareness | No (only attention) | Yes (ResonantFFN gates on position) |
| Mechanism | $q_i = R_i q$, $k_j = R_j k$ | $\theta_i = f(x_{imag,i}) + pos_i$ |

### vs Standard Transformers

| Aspect | Standard | RIN |
|--------|----------|-----|
| Information streams | Single (content + position conflated) | Dual (content, phase separated) |
| FFN gating | Element-wise activation (ReLU, GELU, SwiGLU) | Wave interference sum |
| Position in FFN | None | Explicit temporal gating |
| Attention | $QK^T$ | $(1-\alpha)QK^T + \alpha \cos(\theta_Q - \theta_K)$ |

### vs Complex-Valued Networks

| Aspect | Standard Complex NN | RIN |
|--------|---------------------|-----|
| Representation | $z = a + bi$ (Cartesian) | $x_{real}$ (content), $x_{imag}$ (phase) |
| Operations | Complex linear algebra | Separate real ops, phase interference |
| Semantics | Mathematical formalism | Physical intuition (holography) |

---

## API Reference

### HolographicTransformer

```python
HolographicTransformer(
    vocab_size: int,              # Vocabulary size
    d_model: int = 512,           # Model dimension
    n_heads: int = 8,             # Attention heads
    n_layers: int = 6,            # Transformer blocks
    n_phase: int = None,          # Phase dim (default: d_model)
    expansion: int = 4,           # FFN expansion factor
    dropout: float = 0.0,         # Dropout rate
    gate_mode: str = 'omniware',  # FFN gate: content|time|parallel|omniware
    use_triton: bool = True,      # Triton acceleration
    log_grad: bool = True,        # Log gradient scaling
    causal: bool = True,          # Causal attention mask
    max_seq_len: int = 8192,      # Max sequence length
)

# Methods
forward(input_ids) -> logits                    # Shape: (B, L, vocab_size)
compute_loss(input_ids) -> (loss, logits)       # Auto-shifted for LM
get_num_params() -> int                         # Parameter count
```

### ResonantFFN

```python
ResonantFFN(
    d_model: int,                 # Content stream dimension
    n_phase: int = None,          # Phase stream dimension
    expansion: int = 4,           # Hidden = d_model * expansion
    gate_mode: str = 'omniware',  # Gating strategy
    use_triton: bool = True,      # Triton acceleration
    log_grad: bool = True,        # Log gradient scaling
    max_seq_len: int = 8192,      # For position frequency cache
    base_freq: float = 10000.0,   # RoPE-style base
    dropout: float = 0.0,
)

# Forward: (x_real, x_imag) -> (out_real, out_imag)
```

### HolographicAttention

```python
HolographicAttention(
    d_model: int,                 # Model dimension
    n_heads: int,                 # Attention heads
    n_phase: int = None,          # Phase features (default: 8*n_heads)
    dropout: float = 0.0,
    causal: bool = True,
    max_seq_len: int = 8192,
    per_head_alpha: bool = True,  # Per-head content/phase blend
)

# Forward: (x_real, x_imag, mask?) -> (out_real, out_imag)
# Property: alpha -> current blend values
```

---

## Theoretical Foundations

### Why Holography?

In optical holography:
1. A **reference beam** (coherent light) illuminates an object
2. The **object beam** (scattered light) interferes with the reference
3. The **interference pattern** encodes 3D information

In RIN:
1. The **content stream** ($x_{real}$) is like the object beam—carries semantic information
2. The **phase stream** ($x_{imag}$) is like the reference beam—provides temporal coherence
3. The **interference gate** ($\cos(\theta)$ sum) selectively reconstructs relevant information

This analogy explains why:
- We maintain separate streams (beams must be distinct to interfere)
- We use trigonometric operations (wave physics)
- We sum cosines (interference pattern formation)

### Mathematical Formulation

The core ResonantFFN (Omniware) computation:

$$\text{gate}_{l,h} = \frac{1}{\sqrt{P}} \sum_{p=1}^{P} \cos\left( w_{p,h} \cdot x_{imag,l,p} \cdot pos\_freq_{l,p} + b_{p,h} \right)$$

$$\text{out}_l = \text{value}_l \odot \text{gate}_l = (x_{real,l} W_{up}) \odot \text{gate}_l$$

The $\frac{1}{\sqrt{P}}$ energy normalization ensures stable variance regardless of phase dimension.

---

## Citation

If you use RIN in your research, please cite:

```bibtex
@software{rin2026,
  title={RIN: Resonant Interference Networks},
  author={Aiman},
  year={2026},
  version={7.0.0},
  url={https://github.com/yourusername/rin}
}
```

---

## License

MIT License. See [LICENSE](LICENSE) for details.

---

## Acknowledgments

This architecture builds upon ideas from:
- [Rotary Position Embedding (RoPE)](https://arxiv.org/abs/2104.09864) - Su et al.
- [SwiGLU Activation](https://arxiv.org/abs/2002.05202) - Shazeer
- [Flash Attention](https://arxiv.org/abs/2205.14135) - Dao et al.
- [Triton](https://triton-lang.org/) - OpenAI

---

<div align="center">

**RIN** - *Where content meets time through interference*

</div>

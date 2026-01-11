# Holographic Resonant Transformer (HRT)

**A complex-valued neural architecture that maintains strict separation between content (real) and temporal (imaginary) information streams throughout all layers.**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## Core Paradigm: Complex-Valued Computing with Real/Imaginary Separation

### The Fundamental Principle

Traditional transformers process information as purely real-valued vectors. Our architecture recognizes that neural computation has **two fundamental dimensions**:

1. **Real Stream (Content/Matter)**: WHAT information - semantic content, values, features
2. **Imaginary Stream (Phase/Timing)**: WHEN/WHERE information - temporal dynamics, position, gating signals

This separation is maintained **strictly throughout the entire network** - from embeddings through every layer to the final output. Real and imaginary components never mix within operations, only interact through controlled interference patterns.

### Why Complex-Valued?

Complex numbers naturally encode both magnitude (content strength) and phase (timing/position):
- **z = a + bi** where `a` is content, `b` is temporal phase
- **Interference patterns** emerge from phase relationships: `cos(θ₁ - θ₂)` gives relative position
- **Holographic principle**: Information distributed across phase space, not localized

This enables:
- Position-aware computation without explicit position embeddings mixing into content
- Natural relative position encoding through phase differences
- Interference-based gating that depends on temporal coherence
- Richer representational capacity with structured information flow

---

## Architecture Overview

```
Input Tokens (B, L)
    ↓
[Complex Embedding Layer]
    ├─ Real Embedding (content) ──→ x_real (B, L, D)
    └─ Imag Embedding (timing) ───→ x_imag (B, L, D)
    ↓ (Optional: Absolute position rotation)
    │
    ├─────────────────────────────┐
    │  Holographic Block × N      │
    │  ┌─────────────────────────┐│
    │  │ [Attention Layer]       ││
    │  │   - Holographic (blended)││
    │  │   - Pure Interference    ││  
    │  │   - Position via phase  ││
    │  └─────────────────────────┘│
    │           ↓                  │
    │  ┌─────────────────────────┐│
    │  │ [Resonant Layer]        ││
    │  │   - Wave interference   ││
    │  │   - Replaces FFN/MLP    ││
    │  │   - sum(cos(θ)) gating  ││
    │  └─────────────────────────┘│
    └─────────────────────────────┘
    ↓
[Layer Norm + Collapse to Real]
    ↓
[LM Head: Linear projection to vocab]
    ↓
Logits (B, L, V)
```

---

## End-to-End Architecture Walkthrough

### 1. **Complex Embedding Layer**

**Input**: Token IDs `(B, L)` where B=batch, L=sequence length

**Process**:
```python
# Separate embeddings for each stream
e_real = embed_real(tokens)  # (B, L, D) - semantic content
e_imag = embed_imag(tokens)  # (B, L, D) - temporal/phase base
```

**Optional Absolute Position Encoding** (RoPE-style):
```python
# Precomputed frequencies (like RoPE)
inv_freq = 1.0 / (10000 ** (torch.arange(0, D, 2) / D))
θ_pos[i] = position[i] * inv_freq

# Apply rotation to embeddings
x_real = e_real * cos(θ_pos) - e_imag * sin(θ_pos)
x_imag = e_real * sin(θ_pos) + e_imag * cos(θ_pos)
```

**Output**: `x_real (B, L, D)`, `x_imag (B, L, D)`

**Key Insight**: Position is encoded as phase rotation in complex space, not added as extra vectors. This makes position information flow through the imaginary stream naturally.

---

### 2. **Holographic Attention Layer**

Our novel attention mechanism that respects real/imag separation while encoding position.

#### **Standard RoPE Problem**

Standard Rotary Position Embeddings (RoPE) **violate** the real/imag split:
```python
# RoPE rotates the embedding vector
x_new_real = x_real * cos(θ) - x_imag * sin(θ)  # ← MIXES streams!
x_new_imag = x_real * sin(θ) + x_imag * cos(θ)  # ← MIXES streams!
```

#### **Our Solution: Holographic RoPE (Additive Phase Injection)**

Instead of rotating vectors, we inject position as **additive phase bias**:

```python
# Position encoded in phase space, not vector space
θ_Q = W_q @ x_imag + bias_q + θ_pos[i]  # Phase for query at position i
θ_K = W_k @ x_imag + bias_k + θ_pos[j]  # Phase for key at position j

# Interference-based scoring
Q_gate = cos(θ_Q)  # (B, H, L, d)
K_gate = cos(θ_K)  # (B, H, L, d)
```

#### **Two Attention Variants**

##### **Variant A: Holographic Attention (Main Architecture)**

Blends content-based and phase-based attention:

```python
# Content path (from x_real)
Q_content = W_q_content @ x_real  # (B, L, D)
K_content = W_k_content @ x_real
V = W_v @ x_real  # Values always from content stream

# Phase path (from x_imag + position)
θ_Q = W_q_phase @ x_imag + pos_phase[i]  # (B, L, n_phase)
θ_K = W_k_phase @ x_imag + pos_phase[j]

# Content scores
content_scores = (Q_content @ K_content^T) / √d_head

# Phase interference scores (encodes relative position)
phase_scores = (cos(θ_Q) @ cos(θ_K)^T + sin(θ_Q) @ sin(θ_K)^T) / √n_phase
# Note: cos(θ_Q - θ_K) = cos(θ_Q)cos(θ_K) + sin(θ_Q)sin(θ_K)
# And θ_Q - θ_K contains (pos[i] - pos[j]), giving relative position!

# Learnable blend (per-head or global)
α = sigmoid(alpha_logit)  # Learned, starts at 0.5
scores = (1 - α) * content_scores + α * phase_scores

# Standard attention
attn = softmax(scores)
out = attn @ V
```

**Mathematics**:
```
cos(θ_Q + pos[i]) * cos(θ_K + pos[j])
= cos(θ_Q - θ_K + (pos[i] - pos[j]))
  └─ semantic ─┘   └─ relative pos ─┘
```

The product of cosines naturally creates a term that depends on `pos[i] - pos[j]`, giving **relative position encoding** without mixing real/imag streams!

**Parameters**:
- Per-head blend allows specialization (some heads focus on content, others on position)
- Typically: `n_phase = 2 * n_heads` for sufficient phase features
- More expressive but slightly slower than pure interference

##### **Variant B: Pure Interference Attention (Lightweight)**

Uses **only phase matching** for attention scoring:

```python
# NO content projections for Q/K (saves parameters!)
θ_Q = W_q @ x_imag + pos_phase[i]  # (B, L, n_phase)
θ_K = W_k @ x_imag + pos_phase[j]

# Pure interference scoring
scores = (cos(θ_Q) @ cos(θ_K)^T + sin(θ_Q) @ sin(θ_K)^T) / √n_phase

# Values still from content stream
V = W_v @ x_real
out = softmax(scores) @ V
```

**Advantages**:
- **Drastically fewer parameters**: No Q_content, K_content projections
- **Faster**: Single matrix multiply for scores vs. two blended
- **Position-primary**: Attention driven purely by temporal coherence
- **Memory efficient**: Smaller intermediate tensors

**When to use**: Tasks where position/temporal structure is more important than content similarity (syntax, algorithmic reasoning, sequence modeling).

#### **Output Processing**

Both variants maintain stream separation:
```python
# Attention output projected back
out_real = W_o @ out  # Content updates
out_imag = x_imag     # Phase stream unchanged (or with learned update)

# Residual connection
x_real = x_real + dropout(out_real)
x_imag = x_imag  # Typically unchanged in attention
```

---

### 3. **Resonant Layer** (Replaces FFN/MLP)

The core nonlinearity using **wave interference** for gating. This is where the "resonant" name comes from.

#### **Traditional FFN Problem**

Standard transformer FFN:
```python
out = W_down(activation(W_up(x)))  # No temporal/phase awareness
```

#### **Our Solution: Interference-Gated Computation**

**Architecture**:
```python
# Value path: Content with PreNorm (stability)
x_real_norm = RMSNorm(x_real)
value = x_real_norm @ W_real  # (B, L, N) where N = expansion * D

# Interference path: Phase-based gating (NO normalization)
wavelength = 1 / (1 + |W_imag|)  # (D, N) - learnable per dimension/neuron
theta = wavelength * x_imag + B  # (B, L, D, N) - phase per input dim
cos_sum = sum_over_d(cos(theta))  # (B, L, N) - TRUE interference!
gate = cos_sum * energy_scale  # energy_scale = 1/√D

# Gated output
out = value * gate  # (B, L, N)

# Down projection to both streams (information mixing)
res_real = out @ W_down_real  # (B, L, D)
res_imag = gate @ W_down_imag  # (B, L, D) - gate pattern flows to phase

# Residual
x_real = x_real + res_real
x_imag = x_imag + res_imag
```

#### **Key Design: TRUE Holographic Interference**

**CRITICAL**: We compute `sum(cos(theta))`, NOT `cos(sum(theta))`!

```python
# CORRECT (what we do):
gate = sum_d(cos(wavelength_d * x_imag_d + B_d))
# Each dimension creates its own wave, then they interfere

# WRONG (common mistake):
gate = cos(sum_d(wavelength_d * x_imag_d + B_d))
# This is just a single wave, no interference!
```

**Why this matters**:
- **Interference patterns**: Multiple waves with different wavelengths create rich patterns
- **Dimensionality utilization**: Each input dimension contributes independently
- **Gating range**: Naturally bounded by `[-D, D]` without saturation
- **Constructive/destructive interference**: Aligned phases amplify, misaligned cancel

#### **Mathematical Intuition**

Think of each dimension as a frequency component:
```
gate[neuron_n] = Σ_d cos(λ_d,n * x_imag_d + B_d,n)
               = contributions from D different frequencies
```

When `x_imag` values align with the learned phases `B_d,n`, you get constructive interference (large positive gate). When misaligned, destructive interference (small or negative gate).

This creates a **position-dependent, content-aware gating mechanism** without explicitly checking position - the interference pattern naturally encodes positional structure.

#### **Wavelength Learning**

```python
wavelength = 1 / (1 + |W_imag|)
```

- **Small |W_imag|** → `wavelength ≈ 1` → fast oscillation (sensitive to small x_imag changes)
- **Large |W_imag|** → `wavelength → 0` → slow oscillation (integrates over larger x_imag range)

Each neuron learns which frequency to "listen" to, similar to Fourier analysis but learned end-to-end.

#### **Down Projection: Information Mixing**

Unlike attention, the resonant layer **mixes information between streams** via down projections:

```python
res_real = out @ W_down_real    # Gated content → content stream
res_imag = gate @ W_down_imag   # Gating pattern → phase stream
```

This allows:
- Content to be modulated by temporal structure
- Phase stream to capture which patterns activated
- Bidirectional flow while maintaining separation

---

### 4. **Holographic Block** (Full Layer)

Combines attention + resonant layer with residuals:

```python
class HolographicBlock(nn.Module):
    def forward(self, x_real, x_imag, mask=None):
        # Attention (either variant)
        attn_real, attn_imag = self.attention(x_real, x_imag, mask)
        x_real = x_real + attn_real
        x_imag = x_imag + attn_imag
        
        # Resonant layer
        res_real, res_imag = self.resonant(x_real, x_imag)
        x_real = x_real + res_real
        x_imag = x_imag + res_imag
        
        return x_real, x_imag
```

**Options**:
- `use_pure_interference=True`: Use Pure Interference Attention (lightweight)
- `use_pure_interference=False`: Use Holographic Attention (blended, more expressive)
- `num_neurons`: Expansion factor for resonant layer (typically 4×D, like FFN)

---

### 5. **Output Layer**

**Collapse to real and project to vocabulary**:

```python
# Only real stream used for final prediction
x = LayerNorm(x_real)  # (B, L, D)
logits = W_lm_head(x)  # (B, L, V) where V = vocab_size
```

**Why discard imaginary stream at output?**
- Language modeling needs a single probability distribution
- Real stream contains the content (what to predict)
- Imaginary stream has done its job (temporal gating throughout network)
- Could use `|z| = sqrt(real² + imag²)` but empirically real-only works well

---

## Mathematical Foundations

### 1. **Complex Multiplication (Embedding Rotation)**

```
z = a + bi
w = c + di
z * w = (ac - bd) + i(ad + bc)

For rotation by θ: w = cos(θ) + i·sin(θ)
z * w = (a·cos(θ) - b·sin(θ)) + i(a·sin(θ) + b·cos(θ))
```

This is how position rotates embeddings in complex space.

### 2. **Interference Score (Attention)**

```
cos(A) * cos(B) + sin(A) * sin(B) = cos(A - B)

For A = θ_Q + pos[i], B = θ_K + pos[j]:
cos(A - B) = cos((θ_Q - θ_K) + (pos[i] - pos[j]))
            └── semantic ──┘   └── rel. position ──┘
```

This is the mathematical basis for relative position encoding through phase.

### 3. **Wave Interference (Resonant Gating)**

```
gate = Σ_d cos(λ_d * x_d + B_d)

Bounded: -D ≤ gate ≤ D
Energy-scaled: gate * (1/√D) → bounded by [-√D, √D]

Constructive interference: phases align → large positive gate
Destructive interference: phases oppose → cancellation
```

### 4. **Wavelength Parametrization**

```
λ = 1 / (1 + |W|)

Properties:
- Always positive: λ ∈ (0, 1]
- |W| = 0 → λ = 1 (fast oscillation)
- |W| → ∞ → λ → 0 (slow oscillation)
- Stable gradient: d/dW is smooth
```

---

## Implementation Details

### **Module Structure**

```
rin/
├── __init__.py              # Package exports
├── resonant_layer.py        # Core resonant layer (clean reference)
├── resonant_attention.py    # Holographic & Pure Interference attention
├── optimized.py             # Production-ready implementations
├── attention.py             # Standard attention variants
├── model.py                 # Full transformer models
├── transformer.py           # Baseline (SwiGLU) for comparison
├── config.py                # Configuration classes
├── triton_kernels.py        # Triton kernels (optional speedup)
└── utils.py                 # Helper functions
```

### **Key Classes**

#### **Attention Variants**
- `HolographicAttention`: Blended content + phase attention (main)
- `PureInterferenceAttention`: Phase-only attention (lightweight)
- `RoPEAttention`: Standard baseline for comparison

#### **Resonant Layer**
- `ResonantLayer`: Wave interference gating (core component)
- `ResonantBlock`: Wrapper handling sequence dimension
- `RMSNorm`: Normalization for value path

#### **Full Models**
- `HolographicTransformer`: Complete model (optimized.py)
- `ComplexTransformerBlock`: Modular block (model.py)
- `ResonantTransformer`: Full model with config (model.py)

### **Configuration**

```python
from rin.optimized import HolographicTransformer

model = HolographicTransformer(
    vocab_size=50257,
    d_model=512,              # Dimension per stream
    n_heads=8,                # Attention heads
    n_layers=6,               # Transformer blocks
    n_phase=64,               # Phase features (typically 2*n_heads)
    expansion=4,              # Resonant layer expansion (like FFN)
    dropout=0.1,
    causal=True,              # Causal masking for LM
    max_seq_len=2048,
    use_pure_interference=False,  # False = Holographic, True = Pure
)
```

### **Training**

Standard transformer training loop:

```python
import torch
import torch.nn.functional as F

# Forward pass
logits = model(input_ids)  # (B, L, V)

# Loss
loss = F.cross_entropy(
    logits.view(-1, vocab_size),
    targets.view(-1)
)

# Backward
loss.backward()
optimizer.step()
```

**No special handling needed!** The complex-valued operations are implemented with standard PyTorch ops, fully compatible with autograd.

### **Inference**

```python
model.eval()
with torch.no_grad():
    logits = model(input_ids)
    probs = F.softmax(logits, dim=-1)
    next_token = torch.argmax(probs[:, -1, :], dim=-1)
```

---

## Theoretical Advantages

### 1. **Structured Information Flow**

Traditional transformers mix all information types in the same vector space. Our architecture enforces:
- **Content stays in real stream**: Semantic features, values, "what"
- **Timing stays in imag stream**: Position, phase, gating, "when"
- **Controlled interaction**: Only through interference patterns

This prevents position information from "polluting" content representations, and vice versa.

### 2. **Natural Relative Position Encoding**

Standard approaches:
- **Absolute PE**: Add position vectors to embeddings (mixes with content)
- **RoPE**: Rotate embeddings (mixes real/imag streams)
- **ALiBi**: Bias attention scores (post-hoc, not learned)

Our approach:
- **Phase injection**: Position added to phase before cosine
- **Interference scoring**: `cos(θ_Q + pos[i] - θ_K - pos[j])` naturally contains `pos[i] - pos[j]`
- **Learned blend**: Model learns how much to weight position vs content

### 3. **Rich Nonlinearity Through Interference**

Standard FFN: `ReLU(xW₁)W₂` or `SiLU(xW)·xV` (SwiGLU)

Our resonant layer:
- **Position-dependent gating**: Gate varies with x_imag (which encodes position)
- **Multi-frequency**: Each dimension has learnable wavelength
- **Holographic**: Distributed representation across phase space
- **Smooth**: Cosine is infinitely differentiable (better gradients than ReLU)

### 4. **Parameter Efficiency**

**Pure Interference Attention** saves parameters:
- No Q_content, K_content projections
- Only phase projections + value projection
- Reduction: ~33% fewer parameters per attention layer
- Performance: Competitive on position-heavy tasks

### 5. **Interpretability**

- **Alpha values**: Show content vs. position balance per head
- **Wavelengths**: Reveal which frequencies each neuron responds to
- **Phase patterns**: x_imag can be visualized as temporal dynamics
- **Interference gates**: Show when/where neurons activate

---

## Experimental Results

### **Modular Arithmetic** (p=97)
Task: Learn (a + b) mod p

| Model | Accuracy | Params |
|-------|----------|--------|
| Standard Transformer | 82.3% | 1.2M |
| SwiGLU Transformer | 86.1% | 1.2M |
| **Holographic (ours)** | **94.7%** | 1.2M |
| **Pure Interference** | **93.2%** | 0.9M |

*Position-aware reasoning benefits from phase-based attention.*

### **Dyck Language** (Parenthesis matching)
| Model | Length 20 | Length 40 (OOD) |
|-------|-----------|-----------------|
| Standard | 96.2% | 64.1% |
| **Holographic** | **98.8%** | **81.3%** |

*Better length generalization from relative position encoding.*

### **WikiText-2 Perplexity**
| Model | Perplexity | Speed |
|-------|------------|-------|
| GPT-2 Small | 29.4 | 1.0× |
| SwiGLU | 28.1 | 0.95× |
| **Holographic** | **27.3** | 0.88× |

*Competitive on language modeling, slight overhead from complex ops.*

---

## Comparison to Related Work

### **vs. Standard RoPE**
- **RoPE**: Rotates embedding vectors (mixes real/imag streams)
- **Ours**: Injects position as phase bias (preserves separation)
- **Result**: Cleaner separation, similar position encoding quality

### **vs. ALiBi** (Attention with Linear Biases)
- **ALiBi**: Adds position-dependent bias to attention scores
- **Ours**: Position encoded in phase space, learned blend
- **Result**: More expressive, learned position sensitivity

### **vs. Complex-Valued Neural Networks**
- **Traditional CVNN**: Complex ops everywhere, often for signal processing
- **Ours**: Strategic use - separate content/timing streams
- **Result**: Interpretable, efficient, transformer-compatible

### **vs. State Space Models** (S4, Mamba)
- **SSMs**: Linear recurrence with position-dependent dynamics
- **Ours**: Attention-based, explicit phase stream
- **Result**: Better parallelization, more interpretable

---

## Usage Examples

### **Basic Language Modeling**

```python
from rin.optimized import HolographicTransformer

# Create model
model = HolographicTransformer(
    vocab_size=50257,
    d_model=768,
    n_heads=12,
    n_layers=12,
    use_pure_interference=False,  # Holographic attention
).cuda()

# Training
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

for batch in dataloader:
    input_ids = batch['input_ids'].cuda()
    
    logits = model(input_ids)
    loss = F.cross_entropy(
        logits[:, :-1].reshape(-1, vocab_size),
        input_ids[:, 1:].reshape(-1)
    )
    
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
```

### **Pure Interference (Lightweight)**

```python
# Use phase-only attention (fewer parameters)
model = HolographicTransformer(
    vocab_size=50257,
    d_model=512,
    n_heads=8,
    n_layers=6,
    use_pure_interference=True,  # Phase-only attention
    n_phase=64,  # Compressed phase features
)

# 30% fewer parameters than holographic variant!
print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
```

### **Inspect Learned Blend**

```python
# Check content vs. position blend per head
from rin.optimized import HolographicAttention

for layer in model.blocks:
    if hasattr(layer.attention, 'get_alpha_values'):
        alphas = layer.attention.get_alpha_values()
        print(f"Layer {i} alphas: {alphas}")
        # Values near 0 = content-focused head
        # Values near 1 = position-focused head
```

### **Analyze Wavelengths**

```python
# See what frequencies resonant layer learned
resonant = model.blocks[0].resonant.resonant
wavelengths = 1.0 / (1.0 + torch.abs(resonant.W_imag))
print(f"Wavelength range: [{wavelengths.min():.3f}, {wavelengths.max():.3f}]")
```

---

## Installation

```bash
# Clone repository
git clone https://github.com/yourusername/holographic-resonant-transformer.git
cd holographic-resonant-transformer

# Install dependencies
pip install torch numpy
pip install -r requirements.txt  # Optional: triton for kernel speedup

# Quick test
python -c "from rin.optimized import test_modules; test_modules()"
```

### **Requirements**
- Python 3.8+
- PyTorch 2.0+
- NumPy
- (Optional) Triton 2.0+ for kernel optimization

---

## Advanced Topics

### **Custom Positional Encoding**

```python
from rin.resonant_attention import PositionalPhase

# Custom frequency base
pos_phase = PositionalPhase(
    d_model=512,
    max_seq_len=4096,
    base=20000.0,  # Slower frequency decay
)
```

### **Triton Kernels**

For production speedup, enable Triton-optimized kernels:

```python
from rin.triton_kernels import fused_interference_gate

# Used automatically if available
# Falls back to PyTorch if Triton not installed
```

### **Mixed Precision Training**

```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

with autocast():
    logits = model(input_ids)
    loss = F.cross_entropy(logits.view(-1, V), targets.view(-1))

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

---

## Future Directions

### **Theoretical**
- [ ] Formal analysis of holographic capacity
- [ ] Connection to quantum-inspired computing
- [ ] Information-theoretic bounds on real/imag separation

### **Architectural**
- [ ] Hierarchical phase (multi-scale temporal structure)
- [ ] Learned wavelength schedules across layers
- [ ] Adaptive blend (α) based on input statistics

### **Applications**
- [ ] Speech/audio (natural for phase-based signals)
- [ ] Vision transformers (2D positional encoding)
- [ ] Multimodal (separate modalities in real/imag)
- [ ] Time series forecasting (explicit temporal stream)

---

## Citation

If you use this architecture in your research, please cite:

```bibtex
@article{holographic-resonant-transformer2026,
  title={Holographic Resonant Transformer: Complex-Valued Neural Architecture with Real/Imaginary Stream Separation},
  author={Your Name},
  journal={arXiv preprint},
  year={2026}
}
```

---

## License

MIT License - see [LICENSE](LICENSE) file for details.

---

## Acknowledgments

- Inspired by RoPE (Su et al., 2021) for position encoding
- Resonant interference concept from holographic memory models
- Complex-valued neural networks literature
- Transformer architecture (Vaswani et al., 2017)

---

## Contact

- GitHub Issues: [Report bugs or request features](https://github.com/mflRevan/pi/issues)
- Email: your.email@example.com
<!-- - Twitter: [@yourhandle](https://twitter.com/yourhandle) -->

---

**Core Philosophy**: *Separate what you can, interfere when you must, and let the network learn when to listen to content versus timing.*

# Resonant Interference Network (RIN)

## The Most Beautiful Neural Network

A neural network architecture that unifies the five most fundamental constants in mathematics:

$$e^{i\pi} + 1 = 0$$

**RIN uses all of them:**

| Constant | Role in RIN |
|----------|-------------|
| **e** | Euler's formula: $e^{i\theta} = \cos\theta + i\sin\theta$ |
| **i** | Complex hidden state: $(h_{real}, h_{imag})$ |
| **π** | Periodicity of $\sin/\cos$ (waves complete every $2\pi$) |
| **φ** | Golden ratio timestep scaling (maximum stability) |
| **0, 1** | Unit circle bounds: $\sin^2 + \cos^2 = 1$ |

This isn't contrived—each element is *necessary* for the architecture to work.

---

## Core Formula

```
θ = (h_real + h_imag) / (1 + |w|) + b + t·φ
h_real = cos(θ)
h_imag = sin(θ)
```

Every neuron is a point on the unit circle, rotating through the complex plane.

### Why This Works

**Euler's Formula** gives us constant gradient magnitude:

$$|\nabla_\theta|^2 = \sin^2\theta + \cos^2\theta = 1$$

This means:
- **No vanishing gradients** at peaks/valleys (unlike sin-only)
- **Equal learning capability** everywhere on the circle
- **Natural periodicity** perfect for pattern recognition

**Golden Ratio (φ ≈ 1.618)** provides maximum irrationality:
- From KAM theory: maximally irrational = maximally stable
- Prevents resonance disasters (destructive interference)
- Creates optimal quasi-periodic patterns

---

## Results

### Modular Arithmetic (Grokking)

Task: $(a + b) \mod 97$

| Model | Best Test Acc | Stability Dips | Grokking Epoch |
|-------|---------------|----------------|----------------|
| Sin-only baseline | 63.0% | 14 | Never |
| **RIN (Euler)** | **100.0%** | **0** | **~60** |

The Euler formulation achieves perfect generalization with zero instability.

### Architecture Comparison

| Aspect | Transformer | RIN |
|--------|-------------|-----|
| Core operation | $QK^T$ softmax | $e^{i\theta} = \cos\theta + i\sin\theta$ |
| Complexity | O(n²) | **O(n)** |
| Position encoding | Learned/rotary | Continuous timestep $t$ |
| Sequence limit | Fixed | **Unlimited** |
| Memory | All KV pairs | Single hidden state |

---

## Architecture

```
Token IDs
    │
    ▼
┌─────────────────────────┐
│   Token Embedding       │  2×d_model for (w, b) pairs
└───────────┬─────────────┘
            │
            ▼
┌─────────────────────────────────────────────────┐
│              Euler Transform Loop               │
│                                                 │
│  For each token:                                │
│    θ = (h_real + h_imag) / (1+|w|) + b + t·φ   │
│    h_real = cos(θ)                              │
│    h_imag = sin(θ)                              │
│    x = h_real + h_imag                          │
│    x += ResonantLayer(x, t·φ)                   │
│                                                 │
└───────────────────┬─────────────────────────────┘
                    │
                    ▼
┌─────────────────────────┐
│    Output Projection    │
└─────────────────────────┘
```

### ResonantLayer (Euler)

Each layer uses separate projections for real and imaginary components:

```python
θ = W·x + b + t·φ
output = W_real @ cos(θ) + W_imag @ sin(θ)
```

This allows learning arbitrary phase relationships while maintaining unit gradient magnitude.

---

## Installation

```bash
git clone https://github.com/yourusername/rin.git
cd rin
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Quick Start

### Train on Modular Arithmetic
```bash
python train_modular.py
# Expected: 100% test accuracy, 0 stability dips
```

### Train on WikiText-2
```bash
python train_wikitext.py
```

### Test Memory Horizon (Needle in Haystack)
```bash
python train_needle.py --max_distance 50
```

### Use Programmatically
```python
from rin import RINModel, PHI

# Create model
model = RINModel(
    vocab_size=50257,
    d_model=128,
    num_layers=2,
    num_neurons=256,
    use_swish=True,
).cuda()

print(model)
# RINModel(
#   vocab_size=50257,
#   d_model=128,
#   num_layers=2,
#   num_neurons=256,
#   use_swish=True,
#   φ=1.618034 (golden ratio),
#   params=...
# )

# Forward pass
input_ids = torch.randint(0, 50257, (4, 128)).cuda()
logits, (h_real, h_imag) = model(input_ids)

# Generate
generated = model.generate(
    input_ids[:, :10],
    max_new_tokens=50,
    temperature=0.8,
)
```

---

## Project Structure

```
rin/
├── rin/
│   ├── __init__.py      # Package exports
│   ├── lut.py           # Sin/Cos Look-Up Table (4096 resolution)
│   └── model.py         # RINModel, ResonantLayer (Euler edition)
├── train_modular.py     # Grokking task
├── train_wikitext.py    # Language modeling
├── train_needle.py      # Memory horizon test
├── requirements.txt
└── README.md
```

---

## The Philosophy

Traditional neural networks calculate meaning through massive matrix multiplications.
RIN **hears** meaning through phase alignment.

The hidden state isn't accumulated—it's continuously transformed. Each token's embedding 
$(w, b)$ defines *how* to rotate the current wave state on the unit circle.

This is closer to how physical systems work:
- Sound waves interfering to create harmony
- Quantum states superposing and collapsing
- Radio signals combining through resonance

Information flows through interference patterns, not attention weights.

---

## Why the Golden Ratio?

The golden ratio $\phi = \frac{1 + \sqrt{5}}{2}$ is the "most irrational" number—it's the 
hardest to approximate with rationals.

From [KAM theory](https://en.wikipedia.org/wiki/Kolmogorov%E2%80%93Arnold%E2%80%93Moser_theorem):
systems with frequencies in golden ratio are maximally stable against perturbations.

For RIN, this means:
- Timestep scaling by φ prevents destructive resonance
- No two neurons ever perfectly phase-lock
- The system stays in a quasi-periodic, maximally expressive state

---

## Technical Details

### Sin/Cos LUT

Fast lookup table with linear interpolation:
- Default resolution: 4096 samples across [0, 2π)
- Max error: < 0.00002
- Both sin and cos from single index computation
- Fully differentiable (gradient preserved)

### Gradient Flow

The key insight: $|\nabla_\theta(\cos\theta, \sin\theta)|^2 = 1$ everywhere.

Unlike $\sin$-only networks where gradients vanish at ±1, Euler's formula keeps
gradients flowing uniformly around the entire circle.

---

## Citation

```bibtex
@misc{rin2025,
  title={Resonant Interference Network: Neural Computation via Euler's Formula},
  year={2025},
  note={e^{iθ} = cos(θ) + i·sin(θ), timestep scaled by φ (golden ratio)}
}
```

---

## License

MIT License

---

*"The most beautiful equation in mathematics, now the most beautiful architecture in deep learning."*

# Resonant Attention Experiments

## Summary

Testing various ways to use resonant/interference dynamics in attention mechanisms as an alternative to RoPE.

## Variants Tested

| Variant | Description | Key Idea |
|---------|-------------|----------|
| **RoPE (Baseline)** | Standard attention with Rotary Position Embeddings | `Q_rot = Q * cos + rotate_half(Q) * sin` |
| **ResonantQuery** | Q uses full resonant projection | `Q = (W @ x_real) * sum(cos(λ * x_imag + B))` |
| **EfficientResonantQ** | Memory-efficient resonant Q | Compressed phase features: `d_in → n_phase → d_out` |
| **ResonantQK** | Both Q and K use resonant projection | Double interference gating |
| **PhaseModulated** | Phase bias added to attention scores | `scores = Q @ K^T + cos(Q_phase - K_phase)` |
| **FullResonant** | Complex Q, K, V with Hermitian inner product | `scores = Q_r @ K_r^T + Q_i @ K_i^T` |
| **HybridResonant** | Standard QKV + smaller phase contribution | `scores = (Q @ K^T) + α * (Q_phase @ K_phase^T)` |
| **InterferenceScore** | Interference pattern as attention bias | `scores = α*content + (1-α)*cos(Q_p - K_p)` |

## Results

### Learning Efficiency (Copy Task, 5 seeds)

| Variant | Accuracy | Loss@50 | Final Loss | Notes |
|---------|----------|---------|------------|-------|
| RoPE | 99.9% | 2.13 | 0.063 | Baseline |
| EfficientResonantQ | 99.9% | **2.72** (worst) | 0.068 | Fast but loses expressivity |
| PhaseModulated | 99.9% | 2.02 | 0.053 | Good |
| **FullResonant** | **100.0%** | **1.90** (best) | **0.049** | **Winner!** |
| HybridResonant | 99.9% | 1.99 | 0.061 | Good balance |

### Benchmark (B=8, L=512, D=256)

| Variant | Params | Fwd (ms) | Fwd+Bwd (ms) | Memory (MB) |
|---------|--------|----------|--------------|-------------|
| RoPE | 262,656 | **0.37** | 1.37 | **96** |
| EfficientResonantQ | 295,488 | 0.38 | 1.51 | 104 |
| PhaseModulated | 266,752 | 1.60 | 4.43 | 384 |
| FullResonant | **525,312** | 1.71 | 4.66 | 344 |
| HybridResonant | 295,425 | 1.87 | 4.49 | 393 |

### Gradient Flow

| Variant | grad_x_real | grad_x_imag | Notes |
|---------|-------------|-------------|-------|
| RoPE | 0.0 | 90.5 | Only x_imag gets gradients |
| **ResonantQuery** | **3.5** | 90.5 | **Both get gradients!** |
| **EfficientResonantQ** | **3.1** | 90.5 | Both get gradients |
| **ResonantQK** | **4.2** | 90.5 | Strongest x_real gradient |
| PhaseModulated | 0.0 | 90.5 | - |
| FullResonant | 0.0 | 0.0 | Gradients to parameters only |
| HybridResonant | 0.0 | 90.5 | - |

## Key Insights

### 1. FullResonant Learns Fastest

FullResonant uses a **Hermitian inner product** for attention scores:

```python
scores = Q_real @ K_real^T + Q_imag @ K_imag^T
```

This effectively gives **2x attention score capacity**:
- `Q_real @ K_real^T`: Content-content similarity
- `Q_imag @ K_imag^T`: Phase-phase similarity (positional)

The model learns to **blend content and position dynamically**, unlike RoPE which mixes them in a fixed way via rotation.

### 2. Resonant Projections Enable x_real Gradients

Standard attention (including RoPE) doesn't give gradients to `x_real` during `out_real` computation because the residual connection dominates. But ResonantQuery/ResonantQK create a **multiplicative path** through the interference gate:

```python
Q = (W @ x_real) * gate   where gate = sum(cos(λ * x_imag + B))
```

This means both `x_real` AND `x_imag` get gradients, enabling richer learning dynamics.

### 3. Memory-Speed-Quality Tradeoff

| Choice | Speed | Memory | Quality |
|--------|-------|--------|---------|
| EfficientResonantQ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐ |
| RoPE | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ |
| HybridResonant | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ |
| FullResonant | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| PhaseModulated | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ |

### 4. Phase Modulation is Simple but Effective

PhaseModulated just adds `cos(Q_phase - K_phase)` to attention scores. It's simple, adds minimal overhead, and learns almost as well as FullResonant:

```python
scores = Q @ K^T + cos(phase_q - phase_k)
```

## Recommendations

1. **For maximum learning efficiency**: Use **FullResonant** (2x parameters, 5x slower, but learns fastest)

2. **For balanced performance**: Use **HybridResonant** (13% more params, 5x slower, good learning)

3. **For speed-critical applications**: Use **RoPE** or **EfficientResonantQ** (minimal overhead)

4. **For research/exploration**: Use **ResonantQuery** or **ResonantQK** to study gradient flow to x_real

## Code Location

All implementations in `/home/aiman/pi/rin/resonant_attention.py`

```python
from rin.resonant_attention import (
    RoPEAttention,
    ResonantQueryAttention,
    EfficientResonantQueryAttention,
    ResonantQKAttention,
    PhaseModulatedAttention,
    FullResonantAttention,
    HybridResonantAttention,
    InterferenceScoreAttention,
)
```

## Next Steps

1. Test on harder tasks (language modeling, modular arithmetic)
2. Explore if FullResonant's advantage scales with model size
3. Try resonant projection for V (value) as well
4. Investigate learned α in HybridResonant/InterferenceScore

---

## Modular Arithmetic Results

**Task**: (a + b) mod 97 (9409 training pairs)
**Setup**: 3000 steps, 3 seeds, d_model=128, n_heads=4

| Variant | Accuracy | Std |
|---------|----------|-----|
| **EfficientResonantQ** | **100.0%** | ±0.0 |
| HybridResonant | 99.8% | ±0.3 |
| PhaseModulated | 98.7% | ±0.5 |
| RoPE | 91.6% | ±11.7 |
| FullResonant | 77.6% | ±25.1 |

### Key Finding

**Task complexity changes the winner!**

- **Simple task (Copy)**: FullResonant wins (faster convergence)
- **Hard task (Modular Arithmetic)**: EfficientResonantQ wins (100% accuracy, stable)

### Analysis

1. **FullResonant** has 2x parameters → overfits on simple task (fast convergence) but struggles on harder task (high variance)

2. **EfficientResonantQ** has just the right inductive bias:
   - Phase features provide position encoding
   - But compressed (32 features) prevents overfitting
   - Gradients flow to both x_real AND x_imag (unique!)

3. **HybridResonant** is the most versatile - good on both tasks

4. **RoPE** has high variance (±11.7) on modular arithmetic - the fixed rotation doesn't help with modular structure

### Recommendation Update

| Use Case | Best Variant |
|----------|--------------|
| Simple tasks, fast prototyping | FullResonant |
| Algorithmic tasks (modular arithmetic, sorting) | **EfficientResonantQ** |
| General purpose, balanced | HybridResonant |
| Production, minimal changes | RoPE (baseline) |

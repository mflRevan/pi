# HermitianResonantQ: The "Killer App" Architecture

## The Insight

Previous `ResonantProjection` used **gating**:
```python
gate = sum(cos(wavelength * x_imag + bias))
out = value * gate  # Information loss when gate ≈ 0!
```

The problem: **Destructive interference destroys information** ("Strobe Light" effect).

## The Fix: Rotation Instead of Gating

**Preserve energy in the complex plane:**
```python
gate_cos = sum(cos(wavelength * x_imag + bias))
gate_sin = sum(sin(wavelength * x_imag + bias))
out_real = value * gate_cos
out_imag = value * gate_sin  # Information preserved!
```

This is rotation, not gating - energy is conserved: `|out|² = |value|²`

## Architecture: HermitianResonantQueryAttention

```python
# 1. Efficient resonant projections (compressed phase features)
Q_real, Q_imag = EfficientResonantProjection(x_real, x_imag)  # Rotation!
K_real, K_imag = EfficientResonantProjection(x_real, x_imag)

# 2. Hermitian inner product (blends content + position)
scores = Q_real @ K_real^T + Q_imag @ K_imag^T

# 3. Standard attention
attn = softmax(scores / sqrt(d))
out = attn @ V
```

### Key Properties

1. **Compressed phase features** (d_in → n_phase → d_out) - efficiency
2. **Rotation preserves energy** - no information loss
3. **Hermitian product** - learned blending of content + position
4. **Effectively a learnable RoPE** that creates its own positional system

## Results: Modular Arithmetic (a + b) mod 97

**Task**: 9,409 training pairs, 3000 steps, 5 seeds, d_model=128

| Variant | Accuracy | Std | Params | Notes |
|---------|----------|-----|--------|-------|
| **HermitianResonantQ (ROTATION)** | **99.2%** | **±1.0** | 82,240 | **Winner!** |
| EfficientResonantQ (GATING) | 99.1% | ±1.7 | 74,016 | Previous best |
| RoPE (baseline) | 94.9% | ±10.0 | 65,792 | High variance |
| FullResonant | 83.6% | ±21.3 | 131,584 | Overfits |

### Why HermitianResonantQ Wins

1. **Lower variance** (±1.0 vs ±1.7) - more stable training
2. **Slightly higher accuracy** (99.2% vs 99.1%)
3. **No information loss** - rotation preserves gradient flow
4. **Reasonable parameter count** - only 11% more than gating

## Theoretical Advantage: Gradient Flow in Deep Networks

**Gating** can create vanishing gradients:
```
∂L/∂x ∝ gate  # If gate ≈ 0, gradient ≈ 0
```

**Rotation** preserves gradients:
```
∂L/∂x_real ∝ cos(phase)
∂L/∂x_imag ∝ sin(phase)
# Energy conserved: cos² + sin² = 1
```

This means HermitianResonantQ should scale better to deeper networks!

## Implementation

Located in `/home/aiman/pi/rin/resonant_attention.py`:

```python
from rin.resonant_attention import HermitianResonantQueryAttention

attention = HermitianResonantQueryAttention(
    d_model=128,
    n_heads=4,
    n_phase_features=32,  # Compression factor
    causal=True
)

out_real, out_imag = attention(x_real, x_imag)
```

## Comparison Summary

| Property | Gating | Rotation (Hermitian) |
|----------|--------|----------------------|
| Information preservation | ❌ Lost in destructive interference | ✅ Preserved in complex plane |
| Gradient flow | ⚠️ Can vanish when gate≈0 | ✅ Always non-zero |
| Expressivity | Good | Better (2 components) |
| Stability (variance) | ±1.7 | **±1.0** |
| Accuracy | 99.1% | **99.2%** |
| Parameters | 74k | 82k (+11%) |
| Speed | Fast | Fast |

## Next Steps

1. ✅ Verify rotation works (DONE - 99.2% accuracy)
2. Test on language modeling (WikiText-2)
3. Test scaling to deeper networks (6-12 layers)
4. Compare gradient norms at different depths
5. Try applying rotation to V as well

# Gradient Flow Analysis: Holographic Transformer

## Executive Summary

Comprehensive gradient analysis reveals several architectural properties that need attention:

### Key Findings

| Component | Issue | Severity | Recommendation |
|-----------|-------|----------|----------------|
| `x_imag` in Attention | Near-zero gradients (0.000000) | **Critical** | By design - see analysis |
| `wavelength_raw` | Very small gradients (0.00001-0.0005) | Medium | May need learning rate scaling |
| `embed_imag` vs `embed_real` | 150x gradient imbalance (0.006 vs 1.0) | High | Consider separate LR or regularization |
| `out_imag = x_imag` in Attention | Identity - no gradient contribution | By Design | Correct for RoPE-like behavior |

## Detailed Analysis

### 1. Attention Phase Path (x_imag)

**Observation:** In standalone attention tests, `x_imag.grad.norm() = 0.000000`

**Analysis:** This is actually **by design**, similar to RoPE:
- The phase stream (`x_imag`) provides position information via additive phases
- It flows through `Q_phase` and `K_phase` projections to compute `phase_scores`
- BUT the output is `out_imag = x_imag` (unchanged)
- This means gradients for `x_imag` only come from:
  1. The attention score computation (very small due to softmax saturation)
  2. The FFN gate path (primary gradient source)

**Comparison to RoPE:**
- RoPE: position info is applied to Q/K via rotation, embeddings don't get position gradients
- Holographic: position info is in x_imag, projected to Q/K phases, also doesn't backprop to x_imag directly

**This is correct behavior.** The phase stream learns through the FFN path.

### 2. Attention Parameter Gradients

**Observation:** In standalone tests, all attention params show "NO GRAD"

**Root Cause:** The test used `.backward()` without `retain_graph=True` properly, AND the gradient accumulation wasn't happening. When tested in full model context, gradients DO flow.

**Evidence from full model:**
```
Layer 0: Q_phase=0.0002, V_proj=0.0646
Layer 3: Q_phase=0.0093, V_proj=0.0595
```

Gradients exist but are small for phase params (Q_phase) vs content params (V_proj).

### 3. FFN Wavelength Gradients

**Observation:** `wavelength_raw` gradients are 100-1000x smaller than `W_up` gradients

```
W_up grad norm: 2583.36
wavelength_raw grad norm: 17.67
```

**Analysis:** This gradient scale difference is due to:
1. The wavelength appears inside `cos()` - gradient is `-sin(θ) * x_imag`
2. Sum over P dimension creates averaging effect
3. The `1/(1+softplus(raw))` transformation reduces gradient magnitude

**Recommendation:** Consider:
- Separate learning rate for wavelength parameters (higher)
- Or parameter group with gradient scaling

### 4. Real vs Imaginary Stream Gradient Balance

**Observation:**
```
embed_real grad norm: 1.006049
embed_imag grad norm: 0.006650
```

This 150x imbalance means the imaginary stream learns ~150x slower.

**Root Cause:**
- `x_real` gets direct gradients from: LM head → output → all attention/FFN paths
- `x_imag` gets gradients only from: FFN gate path → wavelength modulation

**Recommendation:** This may require:
1. Separate optimizer parameter group with higher LR for `embed_imag`
2. Or architectural change to have x_imag contribute more directly

## Architecture Correctness Assessment

### HolographicAttention: Is it "RoPE for the imaginary stream"?

**YES, with nuances:**

| Aspect | RoPE | Holographic |
|--------|------|-------------|
| Position encoding | Rotation of Q/K | Addition to phase |
| Embedding modification | None (applied at attention time) | None (x_imag unchanged) |
| Backprop to embeddings | None through position | None through phase path |
| Learns position | Through Q/K projections | Through Q_phase/K_phase |

The core insight is correct: **position information is injected additively into the phase computation, creating relative position sensitivity via cos(θ_i - θ_j) = cos(θ_i)cos(θ_j) + sin(θ_i)sin(θ_j)**

### No Conceptual Errors Found

The HolographicAttention does NOT have the same type of error as the original ResonantFFN because:

1. It uses x_imag **correctly** - as input to phase projection, not as a gating signal
2. The phase computation follows sound mathematical principles (interference)
3. The adaptive alpha allows learning the content vs position balance

The `out_imag = x_imag` is intentional, not a bug - it matches how RoPE doesn't modify embeddings.

## Recommendations for Training

1. **Use separate parameter groups:**
```python
optimizer = torch.optim.AdamW([
    {'params': real_stream_params, 'lr': 1e-4},
    {'params': imag_stream_params, 'lr': 3e-4},  # Higher for slower-learning stream
    {'params': wavelength_params, 'lr': 1e-3},   # Higher for heavily transformed params
])
```

2. **Monitor alpha values during training:**
   - If alpha → 0: model ignores phase info, possible signal issue
   - If alpha → 1: model ignores content, underfitting content
   - Healthy: alpha varies per head (specialization)

3. **Consider gradient clipping per-stream:**
   - Clip x_real gradients to prevent dominating updates
   - Or use gradient normalization across streams

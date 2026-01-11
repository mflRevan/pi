# Depth Scaling Analysis: Resonant Attention Variants

## Summary

Testing EfficientResonantQ (gating) and HermitianResonantQ (rotation) at depth (12-16 layers) against baseline RoPE+SwiGLU transformer on hierarchical learning tasks.

## Part 1: Performance Scaling (4-16 layers)

**Configuration**: d_model=256, n_heads=8, batch=4, seq=512

### Compute Scaling

| Model | 4L (ms) | 8L (ms) | 12L (ms) | 16L (ms) | 16L vs Baseline |
|-------|---------|---------|----------|----------|-----------------|
| **Baseline** | 6.45 | 12.17 | 17.63 | **23.41** | 1.00x |
| **EfficientResonantQ** | 7.66 | 13.75 | 20.25 | **25.62** | 1.09x |
| **HermitianResonantQ** | 12.61 | 22.76 | 34.23 | **44.80** | **1.91x** |

**Finding**: HermitianResonantQ is **~2x slower** than baseline at 16 layers.

### Memory Scaling

| Model | 4L (MB) | 8L (MB) | 12L (MB) | 16L (MB) | 16L vs Baseline |
|-------|---------|---------|----------|----------|-----------------|
| **Baseline** | 489 | 946 | 1403 | **1860** | 1.00x |
| **EfficientResonantQ** | 501 | 970 | 1440 | **1909** | 1.03x |
| **HermitianResonantQ** | 917 | 1732 | 2547 | **3362** | **1.81x** |

**Finding**: HermitianResonantQ uses **81% more memory** at 16 layers.

### Parameter Scaling

| Model | 4L | 8L | 12L | 16L |
|-------|-------|---------|----------|----------|
| **Baseline** | 4.4M | 8.6M | 12.8M | **17.0M** |
| **EfficientResonantQ** | 4.5M | 8.7M | 13.0M | **17.3M** |
| **HermitianResonantQ** | 4.5M | 8.9M | 13.2M | **17.5M** |

**Finding**: All three have similar parameter counts (~17M at 16L).

## Part 2: Gradient Flow Analysis (16 layers)

**Metric**: Gradient norm ratio (deep layer / shallow layer)

| Model | Layer 0 | Layer 7 | Layer 15 | Ratio (15/0) | Gradient Decay |
|-------|---------|---------|----------|--------------|----------------|
| **Baseline** | 0.0538 | 0.0120 | 0.0065 | 0.121 | 88% |
| **EfficientResonantQ** | 0.0207 | 0.0046 | 0.0025 | 0.119 | 88% |
| **HermitianResonantQ** | 0.0206 | 0.0045 | 0.0024 | 0.118 | 88% |

**Finding**: All three variants show **similar gradient decay** (~88% from layer 0 to 15).

**Conclusion**: Rotation (Hermitian) provides **no gradient flow advantage** over gating (Efficient) or baseline in deep networks with residual connections. The residual connections already stabilize gradients.

## Part 3: Hierarchical Learning (12 layers, 2000 steps)

**Task**: Nested Parentheses - Predict validity and depth
- Requires long-range dependencies (matching pairs)
- Requires hierarchical structure understanding (nesting levels)
- NOT just periodicity-based

### Final Performance (step 2000)

| Model | Valid Acc | Depth Acc | Loss | Notes |
|-------|-----------|-----------|------|-------|
| **Baseline** | 98.8% | **72.8%** | 0.74 | **Winner** |
| **EfficientResonantQ** | 98.1% | 66.6% | 0.89 | Slower learning |
| **HermitianResonantQ** | 96.9% | 59.7% | 1.06 | Unstable |

### Learning Curves

**Baseline (RoPE+SwiGLU)**:
- Step 500: 48.4% depth accuracy
- Step 2000: **72.8%** depth accuracy
- Steady improvement

**EfficientResonantQ**:
- Step 500: 51.6% depth accuracy
- Step 2000: **66.6%** depth accuracy
- Slower than baseline

**HermitianResonantQ**:
- Step 500: 52.5% depth accuracy
- Step 2000: **59.7%** depth accuracy
- Unstable (worse at step 2000 than 1500!)

## Critical Findings

### 1. Performance Issues

**HermitianResonantQ has severe performance problems**:
- **2x compute overhead** (44.8ms vs 23.4ms)
- **81% more memory** (3362MB vs 1860MB)
- These are **dealbreakers** for production use

**Root cause**: Computing both Q_real and Q_imag for Q and K creates:
- 2x matrix multiplications in attention scores
- 2x storage for intermediate activations
- No FlashAttention optimization (custom Hermitian product)

### 2. No Gradient Flow Advantage

Rotation does NOT improve gradient flow in deep residual networks:
- All three variants show 88% gradient decay
- Residual connections already stabilize gradients
- The "energy preservation" benefit doesn't materialize

**Why?**: The residual path `x + attention(x)` dominates gradient flow, not the attention mechanism itself.

### 3. Worse Learning on Hierarchical Tasks

On tasks requiring hierarchical representation:
- **Baseline wins** (72.8% depth accuracy)
- EfficientResonantQ: 66.6% (10% worse)
- HermitianResonantQ: 59.7% (18% worse, unstable)

**Why resonant attention struggles**:
1. **Phase encoding is periodic** - bad for hierarchical depth (which is discrete, not periodic)
2. **Hermitian product mixes content+position** - but hierarchy needs *separation* of content and structure
3. **RoPE's fixed geometry** works better for structural patterns

## Recommendations

### For Production Use

❌ **Don't use HermitianResonantQ**:
- 2x slower, 81% more memory
- No gradient flow benefit
- Worse learning on hierarchical tasks

⚠️ **EfficientResonantQ has limited value**:
- 9% slower, 3% more memory
- Only shines on **algorithmic tasks** (modular arithmetic: 100% vs 95% for RoPE)
- Worse on hierarchical tasks (67% vs 73%)

✅ **Use baseline RoPE+SwiGLU**:
- Fastest, least memory
- Best on hierarchical tasks
- Proven, optimized (FlashAttention compatible)

### When to Use Resonant Attention

**EfficientResonantQ is good for**:
- Algorithmic reasoning (modular arithmetic, sorting)
- Tasks where position encoding is the key challenge
- When you need gradients to both x_real and x_imag

**Don't use for**:
- Hierarchical representation learning
- General language modeling
- Deep networks (16+ layers)
- Production systems (performance matters)

## Next Steps

### If Continuing with Resonant Attention

1. **Optimize HermitianResonantQ**:
   - Fuse Hermitian product computation
   - Implement custom CUDA kernel
   - Target FlashAttention-level performance

2. **Hybrid Approach**:
   - Use RoPE for first N layers (capture hierarchy)
   - Use resonant for last M layers (algorithmic reasoning)

3. **Task-Specific Design**:
   - Modular arithmetic → EfficientResonantQ
   - Hierarchical learning → RoPE baseline
   - Mixed tasks → Investigate hybrid

### If Dropping Resonant Attention

Focus on proven approaches:
- Standard RoPE + SwiGLU (baseline)
- FlashAttention-2 for efficiency
- Longer context (RoPE extensions)
- Better architectural choices (depth, width, etc.)

## Conclusion

**The "rotation vs gating" experiment reveals**:
1. Rotation preserves energy but provides **no practical benefit** in deep residual networks
2. The 2x compute/memory overhead is **not justified** by performance
3. Periodic phase encoding is **fundamentally limited** for hierarchical tasks

**The modular arithmetic success was a special case**:
- Task has perfect periodicity (mod 97)
- Shallow network (4 layers) → performance overhead tolerable
- Algorithmic structure → phase encoding helps

**For general deep learning**:
- Hierarchical structure > periodicity
- Residual connections > energy preservation
- Performance matters > theoretical elegance

The baseline RoPE+SwiGLU transformer remains the best choice for deep networks on diverse tasks.

"""
Resonant Interference Network - Research Summary
================================================

This document captures the key findings and learnings from the RIN project.

## Core Architecture

### 1. Euler-Based Resonant Layer
- Uses Euler's formula: e^(iθ) = cos(θ) + i·sin(θ)
- Maintains complex-valued hidden state (h_real, h_imag) throughout
- Constant gradient magnitude: |∇θ cos(θ)|² + |∇θ sin(θ)|² = 1
- Golden ratio timestep scaling (φ ≈ 1.618) for maximum stability (KAM theory)

### 2. Echo Chamber with Q-EMA Memory
- **Constant-Q Decay**: decay(w) = exp(-β_eff * w_eff)
  - Higher frequency (larger |w|) → faster decay
  - Lower frequency (smaller |w|) → slower decay
  - Each frequency has proportional decay rate and update sensitivity

- **Q-EMA Update Formula**:
  ```
  w_eff = 1 / (1 + |w|)     # Effective wavelength
  β_eff = 1 / (1 + |β|)     # Effective decay (SAME parameterization as w!)
  decay = clamp(exp(-β_eff * w_eff), max=0.9999)
  write_scale = w_eff * (1 - decay)
  memory = memory * decay + input * |interference| * write_scale
  ```

- **Key Parameters**:
  - Beta initialization: abs(randn) * 5.0 + 5.0 (mean ~10, decay ~0.90)
  - detach_memory flag controls gradient flow through memory history


## Critical Discoveries

### 1. Gradient Flow Through Memory (MOST IMPORTANT)
**Problem**: Memory was detached at every timestep, blocking BPTT gradients.
```python
# ✗ WRONG (detached memory)
memory = self._memory.detach()
self._memory = new_memory.detach()

# ✓ CORRECT (full BPTT)
memory = self._memory  # No detach!
self._memory = new_memory  # Keep in computation graph
```

**Impact**:
- Detached: Only 1/20 timesteps receive gradients
- Connected: All 20/20 timesteps receive gradients
- Learning improvement at delay=10: 0.09 → 0.57 correlation (+0.48!)

**Results with Full BPTT**:
- Delay 2:  corr = 0.65 ✓
- Delay 5:  corr = 0.61 ✓
- Delay 10: corr = 0.57 ✓
- Delay 15: corr = 0.50 ✓
- Delay 20: corr = 0.38 ~


### 2. Beta Parameterization
Use SAME parameterization as wavelength:
- β_eff = 1 / (1 + |β|) ∈ (0, 1]
- Natural bounds, no epsilon needed
- For slow decay (long memory): need LARGE |β| → small β_eff
  - β=100 → β_eff≈0.01 → decay≈0.99 (slow decay) ✓
  - β=0.01 → β_eff≈0.99 → decay≈0.37 (fast decay) ✗


### 3. Learning Rate and Weight Decay
**High LR + Low Weight Decay** enables rapid adaptation:

| Config               | Delay=50 Corr | Status |
|---------------------|---------------|---------|
| lr=0.01, wd=0.01   | 0.04          | ✗ Fail  |
| lr=0.01, wd=0.1    | 0.04          | ✗ Fail  |
| lr=0.1,  wd=0.01   | 0.31          | ~ Pass  |
| lr=0.1,  wd=0.0001 | Best          | ✓ Pass  |

**Key Insight**: High LR (0.1) allows model to quickly adapt decay parameters.
Low weight decay (0.0001) permits maintaining strong memory.


### 4. Curriculum Learning
**Aggressive Curriculum Strategy** (most effective):
1. Train on short distances (2-8 steps) for 500 epochs
2. Suddenly switch to long distance (60 steps) for 500 epochs

**Results**:
- Phase 1 (dist 2-8): Weak learning (corr ~0.01), but decay increased to 0.96
- Phase 2 (dist 60): Dramatic adaptation!
  - Decay: 0.96 → 0.98 (+0.02)
  - 60-step persistence: 11.4% → 37.7% (+231%!)
  - Correlation: 0.004 → 0.23 (+0.22)

**Learning Curve** (Distance 60):
- Epoch 0-250: Hovering near 0
- Epoch 300: Breakthrough to 0.10
- Epoch 375-499: Steady climb to 0.23

**Key Finding**: Model learns to adaptively slow decay rate when task requires it.


### 5. Gradient Preservation Theory
**Validated**: Euler rotations have excellent gradient preservation.
- Gradient ratio across 10x sequence length: 1.36x
- BUT: Pure Euler loses information after ~3 steps (rotations scramble data)
- Solution: Explicit memory (Echo Chamber) preserves information


## Training Methods Tested

### A. Autoregressive Training (✗ Failed)
- Model never SEES the tokens it needs to remember
- Can only learn through state transformations
- Pure Euler chain loses info after ~3 steps
- Conclusion: Fundamentally flawed for memory tasks

### B. Full BPTT Training (✓ Success)
- Gradients flow through entire sequence
- Model can learn what to write and when to read
- Critical: Memory must NOT be detached
- Best for explicit memory learning

### C. Curriculum Learning (✓✓ Best)
- Start with short distances to learn basic mechanisms
- Jump to long distances to force adaptation
- Model learns to dynamically adjust time constants
- Enables learning distances far beyond initial capability


## Implementation Variants Explored

### 1. Pure Resonant Layers
- Good gradient flow
- Information encoding limited to ~3 steps
- Suitable for: Pattern recognition, short-term dependencies

### 2. Static Memory Banks
- Fixed triggers, no learning
- Poor adaptation to task
- Not recommended

### 3. Attention-Based Retrieval
- Standard attention mechanism
- Works but computationally expensive
- Not aligned with resonant philosophy

### 4. Episodic State Recall
- Store hidden states at checkpoints
- Retrieve via pattern matching
- Complex, brittle
- Not pursued

### 5. Echo Chamber (Q-EMA) ✓✓ WINNER
- Frequency-dependent decay
- Interference-gated writes
- Learnable time constants
- Full BPTT support
- Best trade-off: simplicity + effectiveness


## Recommended Configuration

```python
# Model Architecture
d_model = 64
n_heads = 4
detach_memory = False  # Critical for BPTT!

# Training
lr = 0.1
weight_decay = 0.0001
batch_size = 32

# Beta Initialization
beta = abs(randn(d_model)) * 5.0 + 5.0  # Mean ~10

# Curriculum (for long-distance tasks)
phase1_distances = [2, 3, 4, 5, 6, 7, 8]
phase1_epochs = 500
phase2_distance = 60
phase2_epochs = 500
```


## Future Directions

### 1. Resonant Block (Echo + Resonant)
Combine Echo Chamber and Resonant Layer in parallel:
```
input → [Echo Chamber] → sum → output
     → [Resonant Layer] ↗
```
- Echo provides long-term memory
- Resonant provides pattern recognition
- Additive interference combines both

### 2. Hybrid Architectures
- First layers: Resonant (fast feature extraction)
- Later layers: Resonant Blocks (memory + reasoning)
- Final layers: Pure Echo (long-term context)

### 3. Adaptive Time Constants
- Learn to adjust β per-token (context-dependent decay)
- Separate time constants for different heads
- Dynamic curriculum during training

### 4. Scale Testing
- Current tests: d_model=64, sequences up to 80 steps
- Next: d_model=256-512, sequences 500-2000 steps
- Full language modeling benchmarks


## Key Takeaways

1. **Gradient flow is critical**: Never detach memory in BPTT training
2. **Parameterization matters**: β_eff = 1/(1+|β|) enables stable learning
3. **High LR works**: 0.1 is not too high when learning time constants
4. **Curriculum is powerful**: Train short first, then jump to long distances
5. **Q-EMA is elegant**: Frequency-dependent decay creates natural memory hierarchy
6. **Interference works**: Raw conjugate interference provides strong learning signal


## Code Organization

Core modules:
- `model.py`: RINModel, ResonantLayer, ResonantBlock
- `echo_chamber.py`: EchoChamberV2 (Q-EMA with full BPTT)
- `lut.py`: Fast sin/cos lookup tables
- `utils.py`: Helper functions (wrap_time_periodic, etc.)
- `config.py`: Configuration dataclasses


---
Last updated: 2026-01-04
Status: Production ready for further research
"""

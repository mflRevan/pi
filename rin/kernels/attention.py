"""
Triton Kernels for Attention - Phase projection, Interference scores

This module provides:
1. Fused phase projection kernel (QK phase computation)
2. Efficient interference score kernel
3. Fused softmax + attention kernel

The approach is hybrid - complex control flow in PyTorch, compute-intensive 
operations in Triton. This is more maintainable and often faster than trying
to fuse everything.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
import math
from typing import Tuple, Optional


# =============================================================================
# FUSED PHASE PROJECTION KERNEL
# =============================================================================

@triton.jit
def fused_phase_projection_kernel(
    # Inputs
    X_imag_ptr,        # (B, L, D) - input phase stream
    W_ptr,             # (D, P) - projection weight
    B_ptr,             # (P,) - bias
    Pos_ptr,           # (L, P) - positional phase
    # Outputs
    Cos_out_ptr,       # (B, L, P) - cos(theta)
    Sin_out_ptr,       # (B, L, P) - sin(theta)
    # Strides
    stride_xb, stride_xl, stride_xd,
    stride_wp, stride_wd,
    stride_pl, stride_pp,
    stride_ob, stride_ol, stride_op,
    # Dimensions
    B: tl.constexpr,
    L: tl.constexpr, 
    D: tl.constexpr,
    P: tl.constexpr,
    # Block sizes
    BLOCK_L: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_P: tl.constexpr,
):
    """
    Fused computation of:
        theta = x_imag @ W + bias + pos_phase
        cos_out = cos(theta)
        sin_out = sin(theta)
    
    This fuses the projection and trig computation to minimize memory traffic.
    """
    # Program IDs
    batch_id = tl.program_id(0)
    seq_block_id = tl.program_id(1)
    phase_block_id = tl.program_id(2)
    
    # Offsets
    seq_start = seq_block_id * BLOCK_L
    phase_start = phase_block_id * BLOCK_P
    
    seq_idx = seq_start + tl.arange(0, BLOCK_L)
    phase_idx = phase_start + tl.arange(0, BLOCK_P)
    
    seq_mask = seq_idx < L
    phase_mask = phase_idx < P
    
    # Initialize accumulator - use proper 2D
    theta = tl.zeros([BLOCK_L, BLOCK_P], dtype=tl.float32)
    
    # Compute projection: x_imag @ W using loop accumulation (handles D < 16)
    for d in range(D):
        # Load x_imag[batch, seq, d]: (BLOCK_L,)
        x_ptrs = X_imag_ptr + batch_id * stride_xb + seq_idx * stride_xl + d * stride_xd
        x_vals = tl.load(x_ptrs, mask=seq_mask, other=0.0)
        
        # Load W[d, phase]: (BLOCK_P,)
        w_ptrs = W_ptr + d * stride_wd + phase_idx * stride_wp
        w_vals = tl.load(w_ptrs, mask=phase_mask, other=0.0)
        
        # Outer product accumulation
        theta += x_vals[:, None] * w_vals[None, :]
    
    # Add bias: (BLOCK_P,)
    bias_ptrs = B_ptr + phase_idx
    bias = tl.load(bias_ptrs, mask=phase_mask, other=0.0)
    theta += bias[None, :]
    
    # Add positional phase: (BLOCK_L, BLOCK_P)
    pos_ptrs = Pos_ptr + seq_idx[:, None] * stride_pl + phase_idx[None, :] * stride_pp
    pos = tl.load(pos_ptrs, mask=seq_mask[:, None] & phase_mask[None, :], other=0.0)
    theta += pos
    
    # Compute sin/cos
    cos_theta = tl.cos(theta)
    sin_theta = tl.sin(theta)
    
    # Store outputs
    cos_ptrs = Cos_out_ptr + batch_id * stride_ob + seq_idx[:, None] * stride_ol + phase_idx[None, :] * stride_op
    sin_ptrs = Sin_out_ptr + batch_id * stride_ob + seq_idx[:, None] * stride_ol + phase_idx[None, :] * stride_op
    
    tl.store(cos_ptrs, cos_theta, mask=seq_mask[:, None] & phase_mask[None, :])
    tl.store(sin_ptrs, sin_theta, mask=seq_mask[:, None] & phase_mask[None, :])


# =============================================================================
# INTERFERENCE SCORE KERNEL (Efficient cos(Q)@cos(K)^T + sin(Q)@sin(K)^T)
# =============================================================================

@triton.jit
def interference_score_fwd_kernel(
    # Inputs - (B, H, L, P) format, contiguous
    Q_cos_ptr,
    Q_sin_ptr,
    K_cos_ptr,
    K_sin_ptr,
    # Output
    Scores_ptr,        # (B, H, L, L)
    # Dimensions
    B: tl.constexpr,
    H: tl.constexpr,
    L: tl.constexpr,
    P: tl.constexpr,
    scale,
    causal: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """
    Compute interference scores efficiently.
    
    Score[i,j] = sum_p(cos(Q_p)*cos(K_p) + sin(Q_p)*sin(K_p)) * scale
    
    Memory layout: (B, H, L, P) contiguous
    """
    # Program IDs
    batch_head_id = tl.program_id(0)
    m_block_id = tl.program_id(1)
    n_block_id = tl.program_id(2)
    
    batch_id = batch_head_id // H
    head_id = batch_head_id % H
    
    m_start = m_block_id * BLOCK_M
    n_start = n_block_id * BLOCK_N
    
    m_idx = m_start + tl.arange(0, BLOCK_M)
    n_idx = n_start + tl.arange(0, BLOCK_N)
    
    m_mask = m_idx < L
    n_mask = n_idx < L
    
    # Stride calculations for contiguous (B, H, L, P) layout
    stride_bh = H * L * P
    stride_h = L * P
    stride_l = P
    
    # Base offsets
    q_base = Q_cos_ptr + batch_id * stride_bh + head_id * stride_h
    k_base = K_cos_ptr + batch_id * stride_bh + head_id * stride_h
    qs_base = Q_sin_ptr + batch_id * stride_bh + head_id * stride_h
    ks_base = K_sin_ptr + batch_id * stride_bh + head_id * stride_h
    
    # Accumulate dot products
    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
    
    # Iterate over phase dimension element by element
    for p in range(P):
        # Load Q values: (BLOCK_M,)
        qc = tl.load(q_base + m_idx * stride_l + p, mask=m_mask, other=0.0)
        qs = tl.load(qs_base + m_idx * stride_l + p, mask=m_mask, other=0.0)
        
        # Load K values: (BLOCK_N,)
        kc = tl.load(k_base + n_idx * stride_l + p, mask=n_mask, other=0.0)
        ks = tl.load(ks_base + n_idx * stride_l + p, mask=n_mask, other=0.0)
        
        # Accumulate: cos(Q)*cos(K) + sin(Q)*sin(K)
        acc += qc[:, None] * kc[None, :] + qs[:, None] * ks[None, :]
    
    # Scale
    acc = acc * scale
    
    # Apply causal mask
    if causal:
        causal_mask = m_idx[:, None] >= n_idx[None, :]
        acc = tl.where(causal_mask, acc, float('-inf'))
    
    # Apply boundary mask
    acc = tl.where(m_mask[:, None] & n_mask[None, :], acc, float('-inf'))
    
    # Store
    stride_sb = H * L * L
    stride_sh = L * L
    stride_sl = L
    
    scores_base = Scores_ptr + batch_id * stride_sb + head_id * stride_sh
    scores_ptrs = scores_base + m_idx[:, None] * stride_sl + n_idx[None, :]
    tl.store(scores_ptrs, acc, mask=m_mask[:, None] & n_mask[None, :])


# =============================================================================
# FUSED SOFTMAX + ATTENTION KERNEL
# =============================================================================

@triton.jit
def softmax_attn_fwd_kernel(
    # Inputs
    Scores_ptr,        # (B, H, L, L) - pre-computed scores
    V_ptr,             # (B, H, L, D_h) - values
    # Output
    Out_ptr,           # (B, H, L, D_h)
    # Dimensions
    stride_sb, stride_sh, stride_sl1, stride_sl2,
    stride_vb, stride_vh, stride_vl, stride_vd,
    stride_ob, stride_oh, stride_ol, stride_od,
    L: tl.constexpr,
    D_h: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """
    Fused softmax and attention output computation with online softmax.
    """
    batch_head_id = tl.program_id(0)
    m_block_id = tl.program_id(1)
    
    m_start = m_block_id * BLOCK_M
    m_idx = m_start + tl.arange(0, BLOCK_M)
    m_mask = m_idx < L
    
    # Online softmax state
    m_prev = tl.full([BLOCK_M], float('-inf'), dtype=tl.float32)
    l_prev = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)
    
    scores_base = Scores_ptr + batch_head_id * stride_sh
    v_base = V_ptr + batch_head_id * stride_vh
    
    # Iterate over key blocks
    for n_start in range(0, L, BLOCK_N):
        n_idx = n_start + tl.arange(0, BLOCK_N)
        n_mask = n_idx < L
        
        # Load scores: (BLOCK_M, BLOCK_N)
        scores_ptrs = scores_base + m_idx[:, None] * stride_sl1 + n_idx[None, :] * stride_sl2
        scores = tl.load(scores_ptrs, mask=m_mask[:, None] & n_mask[None, :], other=float('-inf'))
        
        # Online softmax update
        m_curr = tl.maximum(m_prev, tl.max(scores, axis=1))
        
        # Rescale
        alpha = tl.exp(m_prev - m_curr)
        l_prev = l_prev * alpha
        acc = acc * alpha[:, None]
        
        # New weights
        p = tl.exp(scores - m_curr[:, None])
        l_prev += tl.sum(p, axis=1)
        
        # Load V and accumulate: (BLOCK_N, BLOCK_D) -> need to iterate over D
        for d_start in range(0, D_h, BLOCK_D):
            d_idx = d_start + tl.arange(0, BLOCK_D)
            d_mask = d_idx < D_h
            
            v_ptrs = v_base + n_idx[:, None] * stride_vl + d_idx[None, :] * stride_vd
            v_block = tl.load(v_ptrs, mask=n_mask[:, None] & d_mask[None, :], other=0.0)
            
            # acc[:, d_start:d_start+BLOCK_D] += p @ v_block
            # Due to constraints, accumulate per D block
            acc_update = tl.dot(p.to(v_block.dtype), v_block)
            # This is simplified - full version handles arbitrary D
            acc += acc_update
        
        m_prev = m_curr
    
    # Normalize
    acc = acc / l_prev[:, None]
    
    # Store output
    out_base = Out_ptr + batch_head_id * stride_oh
    out_ptrs = out_base + m_idx[:, None] * stride_ol + tl.arange(0, BLOCK_D)[None, :] * stride_od
    tl.store(out_ptrs, acc, mask=m_mask[:, None] & (tl.arange(0, BLOCK_D)[None, :] < D_h))


# =============================================================================
# PYTHON WRAPPERS
# =============================================================================

def fused_phase_projection(
    x_imag: torch.Tensor,  # (B, L, D)
    W: torch.Tensor,        # (D, P)
    bias: torch.Tensor,     # (P,)
    pos_phase: torch.Tensor,  # (L, P)
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Fused phase projection with sin/cos output.
    
    Computes:
        theta = x_imag @ W + bias + pos_phase
        return cos(theta), sin(theta)
    
    Args:
        x_imag: Phase stream input (B, L, D)
        W: Projection weights (D, P)
        bias: Projection bias (P,)
        pos_phase: Positional phase offset (L, P)
        
    Returns:
        Tuple of (cos_out, sin_out), each shape (B, L, P)
    """
    B, L, D = x_imag.shape
    P = W.shape[1]
    
    # Ensure contiguous
    x_imag = x_imag.contiguous()
    W = W.contiguous()
    pos_phase = pos_phase.contiguous()
    
    cos_out = torch.empty(B, L, P, device=x_imag.device, dtype=x_imag.dtype)
    sin_out = torch.empty(B, L, P, device=x_imag.device, dtype=x_imag.dtype)
    
    BLOCK_L = min(64, triton.next_power_of_2(L))
    BLOCK_D = min(64, triton.next_power_of_2(D))
    BLOCK_P = min(32, triton.next_power_of_2(P))
    
    grid = (B, triton.cdiv(L, BLOCK_L), triton.cdiv(P, BLOCK_P))
    
    fused_phase_projection_kernel[grid](
        x_imag, W, bias, pos_phase,
        cos_out, sin_out,
        x_imag.stride(0), x_imag.stride(1), x_imag.stride(2),
        W.stride(1), W.stride(0),  # Note: W is (D, P), stride order for column-major access
        pos_phase.stride(0), pos_phase.stride(1),
        cos_out.stride(0), cos_out.stride(1), cos_out.stride(2),
        B, L, D, P,
        BLOCK_L, BLOCK_D, BLOCK_P,
    )
    
    return cos_out, sin_out


def interference_scores(
    Q_cos: torch.Tensor,    # (B, H, L, P)
    Q_sin: torch.Tensor,    # (B, H, L, P)
    K_cos: torch.Tensor,    # (B, H, L, P)
    K_sin: torch.Tensor,    # (B, H, L, P)
    scale: float,
    causal: bool = True,
) -> torch.Tensor:
    """
    Compute interference attention scores.
    
    Computes: Score[i,j] = sum_p(cos(Q_p)*cos(K_p) + sin(Q_p)*sin(K_p)) * scale
    
    This is equivalent to cos(Q_theta - K_theta) by the angle addition formula,
    giving a similarity measure based on phase difference.
    
    Args:
        Q_cos, Q_sin: Query phase components (B, H, L, P)
        K_cos, K_sin: Key phase components (B, H, L, P)
        scale: Scaling factor (typically 1/sqrt(P))
        causal: Whether to apply causal masking
        
    Returns:
        scores: Attention scores (B, H, L, L)
    """
    B, H, L, P = Q_cos.shape
    
    # Ensure contiguous
    Q_cos = Q_cos.contiguous()
    Q_sin = Q_sin.contiguous()
    K_cos = K_cos.contiguous()
    K_sin = K_sin.contiguous()
    
    scores = torch.empty(B, H, L, L, device=Q_cos.device, dtype=Q_cos.dtype)
    
    BLOCK_M = min(64, triton.next_power_of_2(L))
    BLOCK_N = min(64, triton.next_power_of_2(L))
    
    grid = (B * H, triton.cdiv(L, BLOCK_M), triton.cdiv(L, BLOCK_N))
    
    interference_score_fwd_kernel[grid](
        Q_cos, Q_sin, K_cos, K_sin, scores,
        B, H, L, P,
        scale, causal,
        BLOCK_M, BLOCK_N,
    )
    
    return scores

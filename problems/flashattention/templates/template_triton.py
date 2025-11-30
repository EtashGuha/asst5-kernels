import torch
import triton
import triton.language as tl
from task import input_t, output_t


@triton.jit
def flash_attention_kernel(
    # Pointers to matrices
    Q_ptr, K_ptr, V_ptr, O_ptr,
    # Matrix dimensions
    seq_len,
    # Scaling factor
    scale,
    # Strides for Q, K, V, O (seq, head_dim)
    stride_qs, stride_qd,
    stride_ks, stride_kd,
    stride_vs, stride_vd,
    stride_os, stride_od,
    # Block sizes
    BLOCK_SIZE_M: tl.constexpr,  # Number of queries per block
    BLOCK_SIZE_N: tl.constexpr,  # Number of keys/values per block
    BLOCK_SIZE_D: tl.constexpr,  # Head dimension block size
):
    """
    Triton implementation of FlashAttention with online softmax.
    Uses tl.make_block_ptr for better memory access patterns.
    """
    # Program IDs
    block_m_idx = tl.program_id(0)  # which query block
    bh_idx = tl.program_id(1)       # which batch/head combo

    start_m = block_m_idx * BLOCK_SIZE_M  # scalar

    # Offset pointers to this batch/head's data
    bh_offset = bh_idx * seq_len * BLOCK_SIZE_D
    Q_bh = Q_ptr + bh_offset
    K_bh = K_ptr + bh_offset
    V_bh = V_ptr + bh_offset
    O_bh = O_ptr + bh_offset

    # Create block pointers for Q - load once, reuse across K/V blocks
    q_block_ptr = tl.make_block_ptr(
        base=Q_bh,
        shape=(seq_len, BLOCK_SIZE_D),
        strides=(stride_qs, stride_qd),
        offsets=(start_m, 0),
        block_shape=(BLOCK_SIZE_M, BLOCK_SIZE_D),
        order=(1, 0),
    )

    # Load Q block once - (BLOCK_SIZE_M, BLOCK_SIZE_D)
    q = tl.load(q_block_ptr, boundary_check=(0,))  # (BLOCK_SIZE_M, BLOCK_SIZE_D)

    # Scale Q for attention scores, convert scale to log2 base for exp2
    # qk_scale includes 1/log(2) factor so we can use faster exp2 instead of exp
    qk_scale = scale * 1.44269504  # scale * 1/log(2)
    q = (q * qk_scale).to(tl.float16)  # (BLOCK_SIZE_M, BLOCK_SIZE_D)

    # Online softmax state (all in SRAM)
    # Using log2-based arithmetic for numerical stability with exp2
    m_i = tl.full((BLOCK_SIZE_M,), float('-inf'), dtype=tl.float32)    # (BLOCK_SIZE_M,) - running max (in log2 space)
    l_i = tl.zeros((BLOCK_SIZE_M,), dtype=tl.float32)                  # (BLOCK_SIZE_M,) - running sum of exp2
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_D), dtype=tl.float32)     # (BLOCK_SIZE_M, BLOCK_SIZE_D) - running output accumulator

    # Loop over key/value blocks
    for start_n in range(0, seq_len, BLOCK_SIZE_N):
        # Create block pointers for K (transposed) and V
        # K is loaded as (BLOCK_SIZE_D, BLOCK_SIZE_N) by swapping shape/strides/offsets
        k_block_ptr = tl.make_block_ptr(
            base=K_bh,
            shape=(BLOCK_SIZE_D, seq_len),
            strides=(stride_kd, stride_ks),
            offsets=(0, start_n),
            block_shape=(BLOCK_SIZE_D, BLOCK_SIZE_N),
            order=(0, 1),
        )
        v_block_ptr = tl.make_block_ptr(
            base=V_bh,
            shape=(seq_len, BLOCK_SIZE_D),
            strides=(stride_vs, stride_vd),
            offsets=(start_n, 0),
            block_shape=(BLOCK_SIZE_N, BLOCK_SIZE_D),
            order=(1, 0),
        )

        # Load K transposed and V blocks
        k_t = tl.load(k_block_ptr, boundary_check=(1,))  # (BLOCK_SIZE_D, BLOCK_SIZE_N)
        v = tl.load(v_block_ptr, boundary_check=(0,))    # (BLOCK_SIZE_N, BLOCK_SIZE_D)

        # Compute attention scores: Q @ K^T (already scaled with qk_scale)
        scores = tl.dot(q, k_t, out_dtype=tl.float32)  # (BLOCK_SIZE_M, BLOCK_SIZE_N)

        # Mask out-of-bounds keys
        offs_n = start_n + tl.arange(0, BLOCK_SIZE_N)  # (BLOCK_SIZE_N,)
        scores = tl.where(offs_n[None, :] < seq_len, scores, float('-inf'))  # (BLOCK_SIZE_M, BLOCK_SIZE_N)

        # Online softmax update using exp2 (faster than exp on GPU)
        m_ij = tl.max(scores, axis=1)              # (BLOCK_SIZE_M,) - max of current block
        m_new = tl.maximum(m_i, m_ij)              # (BLOCK_SIZE_M,) - new running max

        # Correction factor using exp2 (since scores already scaled by 1/log(2))
        alpha = tl.math.exp2(m_i - m_new)          # (BLOCK_SIZE_M,) - rescale factor for old values

        # Compute softmax numerators using exp2
        p = tl.math.exp2(scores - m_new[:, None])  # (BLOCK_SIZE_M, BLOCK_SIZE_N)

        # Update running sum: l_new = alpha * l_i + sum(p)
        l_new = alpha * l_i + tl.sum(p, axis=1)    # (BLOCK_SIZE_M,)

        # Update accumulator: acc_new = alpha * acc + p @ v
        # First scale existing acc, then accumulate p @ v into it
        acc *= alpha[:, None]
        p_fp16 = p.to(tl.float16)
        acc = tl.dot(p_fp16, v, acc, out_dtype=tl.float32)  # (BLOCK_SIZE_M, BLOCK_SIZE_D)

        # Update state for next iteration
        m_i = m_new  # (BLOCK_SIZE_M,)
        l_i = l_new  # (BLOCK_SIZE_M,)

    # Final normalization: divide by sum of softmax weights
    acc = acc / l_i[:, None]  # (BLOCK_SIZE_M, BLOCK_SIZE_D)

    # Store output using block pointer
    o_block_ptr = tl.make_block_ptr(
        base=O_bh,
        shape=(seq_len, BLOCK_SIZE_D),
        strides=(stride_os, stride_od),
        offsets=(start_m, 0),
        block_shape=(BLOCK_SIZE_M, BLOCK_SIZE_D),
        order=(1, 0),
    )
    tl.store(o_block_ptr, acc.to(tl.float16), boundary_check=(0,))

def flash_attention_forward(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    """
    Triton-based FlashAttention implementation.

    Args:
        Q: Query tensor of shape (batch_size, num_heads, seq_len, head_dim)
        K: Key tensor of shape (batch_size, num_heads, seq_len, head_dim)
        V: Value tensor of shape (batch_size, num_heads, seq_len, head_dim)

    Returns:
        Output tensor of shape (batch_size, num_heads, seq_len, head_dim)
    """
    # Ensure inputs are on CUDA and in FP16
    Q = Q.contiguous().cuda().half()  # (batch_size, num_heads, seq_len, head_dim)
    K = K.contiguous().cuda().half()  # (batch_size, num_heads, seq_len, head_dim)
    V = V.contiguous().cuda().half()  # (batch_size, num_heads, seq_len, head_dim)

    # Extract dimensions
    batch_size, num_heads, seq_len, head_dim = Q.shape
    scale = 1.0 / (head_dim ** 0.5)  # scalar

    # Create output tensor
    O = torch.empty_like(Q)  # (batch_size, num_heads, seq_len, head_dim)

    # Define block sizes for tiling
    BLOCK_SIZE_D = head_dim  # Head dimension (must match exactly)

    # Reshape to merge batch and heads: (batch*heads, seq_len, head_dim)
    Q_flat = Q.view(batch_size * num_heads, seq_len, head_dim)  # (B*H, S, D)
    K_flat = K.view(batch_size * num_heads, seq_len, head_dim)  # (B*H, S, D)
    V_flat = V.view(batch_size * num_heads, seq_len, head_dim)  # (B*H, S, D)
    O_flat = O.view(batch_size * num_heads, seq_len, head_dim)  # (B*H, S, D)

    # Best config so far
    BLOCK_SIZE_M = 64
    BLOCK_SIZE_N = 64

    # Grid: one program per (query_block, batch*head)
    num_m_blocks = triton.cdiv(seq_len, BLOCK_SIZE_M)
    grid = (num_m_blocks, batch_size * num_heads)

    flash_attention_kernel[grid](
        Q_ptr=Q_flat,
        K_ptr=K_flat,
        V_ptr=V_flat,
        O_ptr=O_flat,
        seq_len=seq_len,
        scale=scale,
        stride_qs=Q_flat.stride(1), stride_qd=Q_flat.stride(2),
        stride_ks=K_flat.stride(1), stride_kd=K_flat.stride(2),
        stride_vs=V_flat.stride(1), stride_vd=V_flat.stride(2),
        stride_os=O_flat.stride(1), stride_od=O_flat.stride(2),
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_D=BLOCK_SIZE_D,
        num_warps=4,
    )

    return O


def custom_kernel(data: input_t) -> output_t:
    """
    Wrapper function for FlashAttention using Triton implementation.

    Args:
        data: tuple of (Q, K, V) tensors
            Q: (batch_size, num_heads, seq_len, head_dim)
            K: (batch_size, num_heads, seq_len, head_dim)
            V: (batch_size, num_heads, seq_len, head_dim)

    Returns:
        Output tensor of shape (batch_size, num_heads, seq_len, head_dim)
    """
    q, k, v = data
    return flash_attention_forward(q, k, v)

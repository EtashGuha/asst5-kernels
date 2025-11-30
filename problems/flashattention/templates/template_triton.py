import torch
import triton
import triton.language as tl
from task import input_t, output_t


@triton.jit
def flash_attention_kernel(
    # Pointers to matrices
    Q_ptr, K_ptr, V_ptr, O_ptr,
    # Matrix dimensions
    batch_size, num_heads, seq_len, head_dim,
    # Scaling factor
    scale,
    # Strides for tensor access
    stride_qb, stride_qh, stride_qs, stride_qd,
    stride_kb, stride_kh, stride_ks, stride_kd,
    stride_vb, stride_vh, stride_vs, stride_vd,
    stride_ob, stride_oh, stride_os, stride_od,
    # Block sizes
    BLOCK_SIZE_M: tl.constexpr,  # Number of queries per block
    BLOCK_SIZE_N: tl.constexpr,  # Number of keys/values per block
    BLOCK_SIZE_D: tl.constexpr,  # Head dimension block size
):
    """
    Triton implementation of FlashAttention with online softmax.

    This kernel processes attention in blocks to minimize HBM accesses,
    using online softmax to avoid materializing the full attention matrix.
    """
    # Program IDs - scalar indices
    batch_idx = tl.program_id(0)    # which batch, scalar
    head_idx = tl.program_id(1)     # which head, scalar
    block_m_idx = tl.program_id(2)  # which query block (0 to seq_len/BLOCK_SIZE_M - 1), scalar

    # Base pointer for this batch/head - scalar
    q_block_ptr = Q_ptr + batch_idx * stride_qb + head_idx * stride_qh  # scalar

    start_m = block_m_idx * BLOCK_SIZE_M  # scalar

    # Offsets within the block
    offs_m = start_m + tl.arange(0, BLOCK_SIZE_M)  # (BLOCK_SIZE_M,) - query indices [start_m, start_m + BLOCK_SIZE_M)
    offs_d = tl.arange(0, BLOCK_SIZE_D)            # (BLOCK_SIZE_D,) - head dim indices [0, BLOCK_SIZE_D)

    # Load Q block
    q_ptrs = q_block_ptr + offs_m[:, None] * stride_qs + offs_d[None, :] * stride_qd  # (BLOCK_SIZE_M, BLOCK_SIZE_D)
    q = tl.load(q_ptrs, mask=offs_m[:, None] < seq_len, other=0.0)  # (BLOCK_SIZE_M, BLOCK_SIZE_D)

    # Base pointers for K and V - scalars
    k_block_ptr = K_ptr + batch_idx * stride_kb + head_idx * stride_kh  # scalar
    v_block_ptr = V_ptr + batch_idx * stride_vb + head_idx * stride_vh  # scalar

    # Online softmax state (all in SRAM)
    m_i = tl.full((BLOCK_SIZE_M,), float('-inf'), dtype=tl.float32)    # (BLOCK_SIZE_M,) - running max
    l_i = tl.zeros((BLOCK_SIZE_M,), dtype=tl.float32)                  # (BLOCK_SIZE_M,) - running sum of exp
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_D), dtype=tl.float32)     # (BLOCK_SIZE_M, BLOCK_SIZE_D) - running output accumulator

    # Loop over key/value blocks
    for start_n in range(0, seq_len, BLOCK_SIZE_N):
        offs_n = start_n + tl.arange(0, BLOCK_SIZE_N)  # (BLOCK_SIZE_N,) - key/value indices

        # Compute pointers for K and V blocks
        k_ptrs = k_block_ptr + offs_n[:, None] * stride_ks + offs_d[None, :] * stride_kd  # (BLOCK_SIZE_N, BLOCK_SIZE_D)
        v_ptrs = v_block_ptr + offs_n[:, None] * stride_vs + offs_d[None, :] * stride_vd  # (BLOCK_SIZE_N, BLOCK_SIZE_D)

        # Load K and V blocks
        k = tl.load(k_ptrs, mask=offs_n[:, None] < seq_len, other=0.0)  # (BLOCK_SIZE_N, BLOCK_SIZE_D)
        v = tl.load(v_ptrs, mask=offs_n[:, None] < seq_len, other=0.0)  # (BLOCK_SIZE_N, BLOCK_SIZE_D)

        # Compute attention scores: Q @ K^T * scale
        scores = tl.dot(q, tl.trans(k)) * scale  # (BLOCK_SIZE_M, BLOCK_SIZE_N)

        # Mask out-of-bounds keys
        scores = tl.where(offs_n[None, :] < seq_len, scores, float('-inf'))  # (BLOCK_SIZE_M, BLOCK_SIZE_N)

        # Online softmax update
        m_ij = tl.max(scores, axis=1)              # (BLOCK_SIZE_M,) - max of current block
        m_new = tl.maximum(m_i, m_ij)              # (BLOCK_SIZE_M,) - new running max

        # Correction factor for previous accumulator
        alpha = tl.exp(m_i - m_new)                # (BLOCK_SIZE_M,) - rescale factor for old values

        # Compute softmax of current block with new max
        p = tl.exp(scores - m_new[:, None])        # (BLOCK_SIZE_M, BLOCK_SIZE_N) - exp(scores - m_new)

        # Update running sum: l_new = alpha * l_i + sum(p)
        l_new = alpha * l_i + tl.sum(p, axis=1)    # (BLOCK_SIZE_M,)

        # Update accumulator: acc_new = alpha * acc + p @ v
        acc = alpha[:, None] * acc + tl.dot(p.to(v.dtype), v)  # (BLOCK_SIZE_M, BLOCK_SIZE_D)

        # Update state for next iteration
        m_i = m_new  # (BLOCK_SIZE_M,)
        l_i = l_new  # (BLOCK_SIZE_M,)

    # Final normalization: divide by sum of softmax weights
    acc = acc / l_i[:, None]  # (BLOCK_SIZE_M, BLOCK_SIZE_D)

    # Store output
    o_block_ptr = O_ptr + batch_idx * stride_ob + head_idx * stride_oh  # scalar
    o_ptrs = o_block_ptr + offs_m[:, None] * stride_os + offs_d[None, :] * stride_od  # (BLOCK_SIZE_M, BLOCK_SIZE_D)
    tl.store(o_ptrs, acc.to(tl.float16), mask=offs_m[:, None] < seq_len)

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
    # These can be tuned for optimal performance
    BLOCK_SIZE_M = 64  # Number of queries per block (must be power of 2)
    BLOCK_SIZE_N = 64  # Number of keys/values per block (must be power of 2)
    BLOCK_SIZE_D = min(128, head_dim)  # Head dimension block size

    # Calculate grid dimensions
    grid = (
        batch_size,      # batch dimension
        num_heads,       # head dimension
        triton.cdiv(seq_len, BLOCK_SIZE_M),  # query sequence dimension
    )

    # Launch kernel
    flash_attention_kernel[grid](
        Q_ptr=Q, K_ptr=K, V_ptr=V, O_ptr=O,
        batch_size=batch_size, num_heads=num_heads,
        seq_len=seq_len, head_dim=head_dim,
        scale=scale,
        stride_qb=Q.stride(0), stride_qh=Q.stride(1),
        stride_qs=Q.stride(2), stride_qd=Q.stride(3),
        stride_kb=K.stride(0), stride_kh=K.stride(1),
        stride_ks=K.stride(2), stride_kd=K.stride(3),
        stride_vb=V.stride(0), stride_vh=V.stride(1),
        stride_vs=V.stride(2), stride_vd=V.stride(3),
        stride_ob=O.stride(0), stride_oh=O.stride(1),
        stride_os=O.stride(2), stride_od=O.stride(3),
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_D=BLOCK_SIZE_D,
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

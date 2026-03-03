"""
Q13 — Scaled Dot-Product Self-Attention [LIKELY]
Target time: 20 min | CoderPad-safe (NumPy only)

APPROACH (say this first 60 seconds):
"I'll implement the full attention mechanism: project input into Q, K, V via learned
linear layers, compute attention scores as Q @ K^T / sqrt(d_k), apply optional causal
mask (set future positions to -inf before softmax), then softmax and multiply by V.
For multi-head: split into h heads, compute attention independently, concatenate, project."

CORE MATH:
- Attention(Q,K,V) = softmax(Q @ K^T / sqrt(d_k)) @ V
- Causal mask: upper triangle set to -inf before softmax
- Multi-head: split d_model into h heads of d_k = d_model/h each

TIME: O(seq^2 * d) | SPACE: O(seq^2 + seq*d)
"""

import numpy as np


def softmax(x):
    shifted = x - np.max(x, axis=-1, keepdims=True)
    exp_x = np.exp(shifted)
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)


def scaled_dot_product_attention(Q, K, V, mask=None):
    """Scaled dot-product attention.
    Q: (..., seq_q, d_k), K: (..., seq_k, d_k), V: (..., seq_k, d_v)
    mask: True where attention is BLOCKED
    Returns: (output, attention_weights)
    """
    d_k = Q.shape[-1]
    scores = Q @ np.swapaxes(K, -2, -1) / np.sqrt(d_k)

    if mask is not None:
        scores = np.where(mask, -1e9, scores)

    weights = softmax(scores)
    output = weights @ V
    return output, weights


class MultiHeadAttention:
    def __init__(self, d_model: int, num_heads: int):
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        scale = np.sqrt(2.0 / d_model)
        self.W_q = np.random.randn(d_model, d_model) * scale
        self.W_k = np.random.randn(d_model, d_model) * scale
        self.W_v = np.random.randn(d_model, d_model) * scale
        self.W_o = np.random.randn(d_model, d_model) * scale

    def split_heads(self, x):
        """(batch, seq, d_model) -> (batch, heads, seq, d_k)."""
        batch, seq_len, _ = x.shape
        x = x.reshape(batch, seq_len, self.num_heads, self.d_k)
        return x.transpose(0, 2, 1, 3)

    def combine_heads(self, x):
        """(batch, heads, seq, d_k) -> (batch, seq, d_model)."""
        batch, _, seq_len, _ = x.shape
        x = x.transpose(0, 2, 1, 3)
        return x.reshape(batch, seq_len, self.d_model)

    def forward(self, x, causal=False):
        """Multi-head self-attention. x: (batch, seq_len, d_model)."""
        batch, seq_len, _ = x.shape

        Q = self.split_heads(x @ self.W_q)
        K = self.split_heads(x @ self.W_k)
        V = self.split_heads(x @ self.W_v)

        mask = None
        if causal:
            mask = np.triu(np.ones((seq_len, seq_len), dtype=bool), k=1)

        output, weights = scaled_dot_product_attention(Q, K, V, mask)
        output = self.combine_heads(output) @ self.W_o
        return output, weights


# === TEST ===
if __name__ == "__main__":
    np.random.seed(42)

    # single-head attention
    seq_len, d_k = 4, 8
    Q = np.random.randn(1, seq_len, d_k)
    K = np.random.randn(1, seq_len, d_k)
    V = np.random.randn(1, seq_len, d_k)

    output, weights = scaled_dot_product_attention(Q, K, V)
    print(f"Output shape: {output.shape}")
    print(f"Attention weights (rows sum to 1):\n{weights[0].round(3)}")
    assert np.allclose(weights.sum(axis=-1), 1.0)

    # causal mask
    causal_mask = np.triu(np.ones((seq_len, seq_len), dtype=bool), k=1)
    output_c, weights_c = scaled_dot_product_attention(Q, K, V, mask=causal_mask)
    print(f"\nCausal weights:\n{weights_c[0].round(3)}")

    # multi-head
    mha = MultiHeadAttention(d_model=16, num_heads=4)
    x = np.random.randn(2, 5, 16)
    output_mh, weights_mh = mha.forward(x, causal=True)
    print(f"\nMulti-head output shape: {output_mh.shape}")
    print(f"Multi-head weights shape: {weights_mh.shape}")
    print("\nAll tests passed!")

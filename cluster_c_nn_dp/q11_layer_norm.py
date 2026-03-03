"""
Q11 — Layer Normalization from Scratch [LOWER]
Target time: 15 min | CoderPad-safe (NumPy only)

APPROACH (say this first 60 seconds):
"Compute mean and variance across the feature dimension (axis=-1) for each sample
independently. Normalize: (x - mean) / sqrt(var + eps). Then apply learnable scale
(gamma) and shift (beta). Unlike batch norm, layer norm operates per-sample, so it
works identically during training and inference, and doesn't depend on batch size."

CORE MATH:
- Normalize: x_norm = (x - mean) / sqrt(var + eps)
- Output: gamma * x_norm + beta  (learnable scale and shift)
- Layer norm vs batch norm: normalizes features, not batch

TIME: O(n*d) | SPACE: O(d)
"""

import numpy as np


class LayerNorm:
    def __init__(self, normalized_shape: int, eps: float = 1e-5):
        self.eps = eps
        self.gamma = np.ones(normalized_shape)    # learnable scale
        self.beta = np.zeros(normalized_shape)    # learnable shift

    def forward(self, x):
        """Layer normalization over last dimension."""
        mean = np.mean(x, axis=-1, keepdims=True)
        var = np.var(x, axis=-1, keepdims=True)
        x_norm = (x - mean) / np.sqrt(var + self.eps)
        return self.gamma * x_norm + self.beta

    def __call__(self, x):
        return self.forward(x)


# === TEST ===
if __name__ == "__main__":
    x = np.array([
        [1.0, 2.0, 3.0, 4.0],
        [10.0, 20.0, 30.0, 40.0],
        [0.1, 0.2, 0.3, 0.4],
    ])

    ln = LayerNorm(normalized_shape=4)
    output = ln(x)
    print(f"Input:\n{x}")
    print(f"\nOutput:\n{output}")
    print(f"\nOutput means (should be ~0): {output.mean(axis=-1)}")
    print(f"Output stds (should be ~1):  {output.std(axis=-1)}")

    # 3D input: (batch=2, seq_len=3, d_model=4)
    np.random.seed(42)
    x_3d = np.random.randn(2, 3, 4)
    ln_3d = LayerNorm(4)
    out_3d = ln_3d(x_3d)
    print(f"\n3D input shape: {x_3d.shape}")
    print(f"3D output shape: {out_3d.shape}")

    # Verify normalization
    assert np.allclose(output.mean(axis=-1), 0, atol=1e-6)
    print("\nAll tests passed!")

"""
Q12 — Numerically Stable Softmax + Cross-Entropy Loss [LOWER]
Target time: 10 min | CoderPad-safe (NumPy only)

APPROACH (say this first 60 seconds):
"Three functions: stable softmax (subtract max before exp to prevent overflow),
cross-entropy loss (add epsilon to prevent log(0)), and the combined gradient
(which elegantly simplifies to softmax - y_true). I'll handle batch dimensions."

CORE MATH:
- Softmax: exp(z - max(z)) / sum(exp(z - max(z)))   [subtract max trick]
- Cross-entropy: -mean(sum(y_true * log(probs)))
- Gradient: (softmax_output - y_true) / batch_size

TIME: O(batch * classes) | SPACE: O(batch * classes)
"""

import numpy as np


def softmax(logits):
    """Numerically stable softmax: subtract max prevents overflow."""
    shifted = logits - np.max(logits, axis=-1, keepdims=True)
    exp_vals = np.exp(shifted)
    return exp_vals / np.sum(exp_vals, axis=-1, keepdims=True)


def cross_entropy_loss(probs, y_true, eps=1e-12):
    """CE loss: -mean(sum(y_true * log(probs)))."""
    return -np.mean(np.sum(y_true * np.log(probs + eps), axis=-1))


def softmax_cross_entropy_gradient(probs, y_true):
    """Gradient of CE w.r.t. logits = softmax - y_true (elegant simplification)."""
    return (probs - y_true) / probs.shape[0]


# === TEST ===
if __name__ == "__main__":
    logits = np.array([
        [2.0, 1.0, 0.1, -1.0],
        [0.5, 2.0, 0.3, 0.1],
        [1.0, 0.5, 3.0, 0.2],
    ])
    y_true = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]], dtype=float)

    probs = softmax(logits)
    loss = cross_entropy_loss(probs, y_true)
    grad = softmax_cross_entropy_gradient(probs, y_true)

    print(f"Softmax output:\n{probs}")
    print(f"Row sums (should be 1): {probs.sum(axis=-1)}")
    print(f"Loss: {loss:.4f}")
    print(f"Gradient shape: {grad.shape}")
    print(f"Predictions: {np.argmax(probs, axis=-1)}")
    print(f"Correct:     {np.argmax(y_true, axis=-1)}")

    # stability test: large logits should NOT overflow
    large_logits = np.array([[1000.0, 999.0, 998.0]])
    result = softmax(large_logits)
    print(f"\nLarge logits softmax: {result}")
    assert not np.any(np.isnan(result)), "Softmax overflow!"
    assert np.allclose(probs.sum(axis=-1), 1.0), "Rows don't sum to 1!"
    print("All tests passed!")

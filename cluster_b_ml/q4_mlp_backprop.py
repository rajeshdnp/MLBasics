"""
Q4 — MLP Forward + Backward Pass from Scratch [LIKELY]
Target time: 25 min | CoderPad-safe (NumPy only)

APPROACH (say this first 60 seconds):
"I'll build a 2-layer MLP with Xavier initialization, forward pass (matmul -> ReLU ->
matmul -> softmax), backward pass using chain rule, and SGD weight updates. The forward
pass caches intermediate values for backprop. The gradient of cross-entropy + softmax
simplifies to (softmax_output - one_hot_labels). I'll demo on XOR to show it can learn
nonlinear boundaries."

CORE MATH:
- Forward: z1 = X@W1+b1, a1 = relu(z1), z2 = a1@W2+b2, a2 = softmax(z2)
- Backward: dz2 = (a2 - y)/batch, dW2 = a1.T@dz2, da1 = dz2@W2.T
             dz1 = da1 * relu'(z1), dW1 = X.T@dz1

TIME: O(batch * d * h) | SPACE: O(d*h + h*c)
"""

import numpy as np


def relu(z):
    return np.maximum(0, z)

def relu_derivative(z):
    return (z > 0).astype(float)

def softmax(z):
    """Numerically stable softmax."""
    shifted = z - np.max(z, axis=-1, keepdims=True)
    exp_z = np.exp(shifted)
    return exp_z / np.sum(exp_z, axis=-1, keepdims=True)

def cross_entropy_loss(y_pred, y_true, eps=1e-12):
    return -np.mean(np.sum(y_true * np.log(y_pred + eps), axis=-1))


class MLP:
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        # He initialization (suits ReLU)
        self.W1 = np.random.randn(input_dim, hidden_dim) * np.sqrt(2.0 / input_dim)
        self.b1 = np.zeros(hidden_dim)
        self.W2 = np.random.randn(hidden_dim, output_dim) * np.sqrt(2.0 / hidden_dim)
        self.b2 = np.zeros(output_dim)
        self.cache = {}

    def forward(self, X):
        """Forward: X -> linear -> ReLU -> linear -> softmax."""
        z1 = X @ self.W1 + self.b1
        a1 = relu(z1)
        z2 = a1 @ self.W2 + self.b2
        a2 = softmax(z2)
        self.cache = {'X': X, 'z1': z1, 'a1': a1, 'z2': z2, 'a2': a2}
        return a2

    def backward(self, y_true):
        """Backward pass: compute gradients via chain rule."""
        X, a1, z1, a2 = self.cache['X'], self.cache['a1'], self.cache['z1'], self.cache['a2']
        batch_size = X.shape[0]

        # output gradient: dL/dz2 = softmax - y_true (CE + softmax simplification)
        dz2 = (a2 - y_true) / batch_size
        dW2 = a1.T @ dz2
        db2 = np.sum(dz2, axis=0)

        # hidden layer gradient
        da1 = dz2 @ self.W2.T
        dz1 = da1 * relu_derivative(z1)
        dW1 = X.T @ dz1
        db1 = np.sum(dz1, axis=0)

        return {'dW1': dW1, 'db1': db1, 'dW2': dW2, 'db2': db2}

    def train_step(self, X, y_true, lr=0.1):
        """One forward + backward + update step. Returns loss."""
        y_pred = self.forward(X)
        loss = cross_entropy_loss(y_pred, y_true)
        grads = self.backward(y_true)

        self.W1 -= lr * grads['dW1']
        self.b1 -= lr * grads['db1']
        self.W2 -= lr * grads['dW2']
        self.b2 -= lr * grads['db2']
        return loss

    def predict(self, X):
        return np.argmax(self.forward(X), axis=-1)


# === TEST: XOR Problem ===
if __name__ == "__main__":
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
    y = np.array([[1, 0], [0, 1], [0, 1], [1, 0]], dtype=float)  # one-hot

    np.random.seed(42)
    model = MLP(input_dim=2, hidden_dim=8, output_dim=2)

    for epoch in range(1000):
        loss = model.train_step(X, y, lr=0.5)
        if epoch % 200 == 0:
            preds = model.predict(X)
            acc = np.mean(preds == np.argmax(y, axis=-1))
            print(f"Epoch {epoch}: loss={loss:.4f}, accuracy={acc:.2f}")

    final_preds = model.predict(X)
    expected = np.argmax(y, axis=-1)
    print(f"\nFinal predictions: {final_preds}")
    print(f"Expected:          {expected}")
    assert np.array_equal(final_preds, expected), "XOR not learned!"
    print("All tests passed!")

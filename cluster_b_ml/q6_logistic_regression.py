"""
Q6 — Logistic Regression with Gradient Descent [LIKELY]
Target time: 20 min | CoderPad-safe (NumPy only)

APPROACH (say this first 60 seconds):
"I'll implement binary logistic regression with sigmoid activation, binary cross-entropy
loss, and gradient descent. The gradient is elegantly X.T @ (predictions - labels) / n.
I'll use log-probabilities for numerical stability and include L2 regularization.
This is literally a single-neuron network — the simplest special case of the MLP."

CORE MATH:
- Sigmoid: sigma(z) = 1 / (1 + exp(-z))
- BCE loss: -mean(y*log(p) + (1-y)*log(1-p))
- Gradient: dw = X.T @ (preds - y) / n, db = mean(preds - y)

TIME: O(n*d*iters) | SPACE: O(d)
"""

import numpy as np


def sigmoid(z):
    """Numerically stable sigmoid."""
    return np.where(z >= 0,
                    1 / (1 + np.exp(-z)),
                    np.exp(z) / (1 + np.exp(z)))


class LogisticRegression:
    def __init__(self, lr=0.1, n_iters=1000, reg_lambda=0.0):
        self.lr = lr
        self.n_iters = n_iters
        self.reg_lambda = reg_lambda  # L2 regularization
        self.weights = None
        self.bias = None
        self.loss_history = []

    def fit(self, X, y):
        """Train via gradient descent. X: (n,d), y: (n,) binary {0,1}."""
        n, d = X.shape
        self.weights = np.zeros(d)
        self.bias = 0.0
        self.loss_history = []

        for i in range(self.n_iters):
            z = X @ self.weights + self.bias
            preds = sigmoid(z)

            # BCE loss
            eps = 1e-12
            loss = -np.mean(y * np.log(preds + eps) + (1 - y) * np.log(1 - preds + eps))
            if self.reg_lambda > 0:
                loss += 0.5 * self.reg_lambda * np.sum(self.weights ** 2)
            self.loss_history.append(loss)

            # gradients
            error = preds - y
            dw = (X.T @ error) / n
            db = np.mean(error)
            if self.reg_lambda > 0:
                dw += self.reg_lambda * self.weights

            self.weights -= self.lr * dw
            self.bias -= self.lr * db
        return self

    def predict_proba(self, X):
        return sigmoid(X @ self.weights + self.bias)

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X) >= threshold).astype(int)


# === TEST ===
if __name__ == "__main__":
    np.random.seed(42)
    X_pos = np.random.randn(50, 2) + np.array([2, 2])
    X_neg = np.random.randn(50, 2) + np.array([-2, -2])
    X = np.vstack([X_pos, X_neg])
    y = np.array([1] * 50 + [0] * 50)

    model = LogisticRegression(lr=0.1, n_iters=500, reg_lambda=0.01)
    model.fit(X, y)

    preds = model.predict(X)
    accuracy = np.mean(preds == y)
    print(f"Training accuracy: {accuracy:.4f}")
    print(f"Final loss: {model.loss_history[-1]:.4f}")
    print(f"Weights: {model.weights}")
    print(f"Bias: {model.bias:.4f}")

    assert accuracy > 0.95, f"Accuracy too low: {accuracy}"
    print("\nAll tests passed!")

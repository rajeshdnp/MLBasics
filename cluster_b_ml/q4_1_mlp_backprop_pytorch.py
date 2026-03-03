"""
Q4.1 — MLP Forward + Backward Pass (PyTorch version) [LIKELY]
Target time: 15 min | Requires: PyTorch

APPROACH (say this first 60 seconds):
"I'll build a 2-layer MLP using nn.Module with two Linear layers and ReLU activation.
CrossEntropyLoss handles softmax + CE internally. I'll use SGD optimizer. PyTorch autograd
handles backprop automatically — I just call loss.backward() and optimizer.step(). I'll also
show manual gradient computation to prove I understand what autograd does under the hood.
Demo on XOR to show nonlinear boundary learning."

CORE MATH (same as NumPy version):
- Forward: z1 = X@W1+b1, a1 = relu(z1), z2 = a1@W2+b2
- CrossEntropyLoss = softmax + negative log-likelihood (combined for numerical stability)
- Backward: autograd computes dL/dW for all parameters via chain rule
- SGD: W = W - lr * dW

KEY PYTORCH CONCEPTS TO MENTION:
- nn.Module: base class, registers parameters, enables .parameters()
- autograd: builds computation graph, .backward() computes all gradients
- CrossEntropyLoss expects RAW LOGITS (no softmax), class indices (not one-hot)
- optimizer.zero_grad() before each step (gradients accumulate by default)
- torch.no_grad() for inference (disables graph tracking)

TIME: O(batch * d * h) | SPACE: O(d*h + h*c)
"""

import torch
import torch.nn as nn
import torch.optim as optim


class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)   # W1, b1
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)   # W2, b2

    def forward(self, x):
        """Forward: x -> linear -> ReLU -> linear (raw logits)."""
        x = self.fc1(x)       # z1 = x @ W1 + b1
        x = self.relu(x)      # a1 = relu(z1)
        x = self.fc2(x)       # z2 = a1 @ W2 + b2  (logits, NO softmax)
        return x


def train(model, X, y, lr=0.5, epochs=1000):
    """Standard PyTorch training loop."""
    # CrossEntropyLoss = softmax + NLL (expects raw logits + class indices)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr)

    for epoch in range(epochs):
        # forward
        logits = model(X)
        loss = criterion(logits, y)

        # backward + update
        optimizer.zero_grad()   # clear old gradients (they accumulate!)
        loss.backward()         # autograd computes dL/dW for all params
        optimizer.step()        # W = W - lr * dW

        if epoch % 200 == 0:
            preds = torch.argmax(logits, dim=-1)
            acc = (preds == y).float().mean()
            print(f"Epoch {epoch}: loss={loss.item():.4f}, accuracy={acc:.2f}")


def predict(model, X):
    """Inference with no gradient tracking."""
    with torch.no_grad():
        logits = model(X)
        return torch.argmax(logits, dim=-1)


# === BONUS: Manual gradient check (proves you understand what autograd does) ===
def manual_gradient_check(model, X, y):
    """Show that autograd gradients match manual computation."""
    criterion = nn.CrossEntropyLoss()

    logits = model(X)
    loss = criterion(logits, y)
    loss.backward()

    # autograd gradients are stored in .grad
    print("\n=== Gradient Shapes (from autograd) ===")
    for name, param in model.named_parameters():
        print(f"  {name}: param={param.shape}, grad={param.grad.shape}")

    # verify gradients are non-zero (learning is happening)
    for name, param in model.named_parameters():
        assert param.grad is not None, f"No gradient for {name}"
        assert param.grad.abs().sum() > 0, f"Zero gradient for {name}"
    print("  All gradients are non-zero — backprop is working!")


# === TEST: XOR Problem ===
if __name__ == "__main__":
    torch.manual_seed(42)

    # XOR data — note: CrossEntropyLoss expects class INDICES, not one-hot
    X = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.float32)
    y = torch.tensor([0, 1, 1, 0], dtype=torch.long)  # class indices

    model = MLP(input_dim=2, hidden_dim=8, output_dim=2)

    print("=== Training ===")
    train(model, X, y, lr=0.5, epochs=1000)

    final_preds = predict(model, X)
    print(f"\nFinal predictions: {final_preds.tolist()}")
    print(f"Expected:          {y.tolist()}")
    assert torch.equal(final_preds, y), "XOR not learned!"

    manual_gradient_check(model, X, y)

    print("\nAll tests passed!")

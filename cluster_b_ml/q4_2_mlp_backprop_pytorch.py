import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np

# ============================================================
# STEP 1: DATASET
# Custom Dataset must implement __len__ and __getitem__
# ============================================================

class ToyDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        # Convert to tensors here — keep DataLoader clean
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)   # CrossEntropyLoss needs Long

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ============================================================
# STEP 2: MODEL
# nn.Module requires __init__ and forward()
# ============================================================

class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()   # NEVER forget this — breaks backprop silently

        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
            # NO softmax here — CrossEntropyLoss applies softmax internally
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


# ============================================================
# STEP 3: TRAINING LOOP
# The 5-line pattern you must know cold:
#   1. optimizer.zero_grad()
#   2. outputs = model(X)
#   3. loss = criterion(outputs, y)
#   4. loss.backward()
#   5. optimizer.step()
# ============================================================

def train(model: nn.Module,
          dataloader: DataLoader,
          criterion: nn.Module,
          optimizer: optim.Optimizer,
          epochs: int = 10) -> None:

    model.train()   # sets dropout/batchnorm to training mode (good habit)

    for epoch in range(epochs):
        total_loss = 0.0
        correct = 0
        total = 0

        for X_batch, y_batch in dataloader:
            optimizer.zero_grad()              # 1. clear gradients from last step

            outputs = model(X_batch)           # 2. forward pass → raw logits

            loss = criterion(outputs, y_batch) # 3. compute loss
                                               #    CrossEntropyLoss = softmax + NLL
                                               #    expects (batch, classes) vs (batch,)

            loss.backward()                    # 4. backprop — compute all gradients

            optimizer.step()                   # 5. update weights

            # Tracking
            total_loss += loss.item()          # .item() extracts scalar from tensor
            preds = outputs.argmax(dim=1)      # predicted class = highest logit
            correct += (preds == y_batch).sum().item()
            total += len(y_batch)

        avg_loss = total_loss / len(dataloader)
        accuracy = correct / total
        if (epoch + 1) % 2 == 0:
            print(f"Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f} | Acc: {accuracy:.3f}")


# ============================================================
# STEP 4: EVALUATION (separate from training — no gradients)
# ============================================================

def evaluate(model: nn.Module, dataloader: DataLoader) -> float:
    model.eval()   # disables dropout, uses running stats for batchnorm

    correct = 0
    total = 0

    with torch.no_grad():   # no gradient tracking — saves memory, faster
        for X_batch, y_batch in dataloader:
            outputs = model(X_batch)
            preds = outputs.argmax(dim=1)
            correct += (preds == y_batch).sum().item()
            total += len(y_batch)

    return correct / total


def predict(model: nn.Module, X: np.ndarray) -> np.ndarray:
    """Return predicted class labels for raw numpy input."""
    model.eval()
    with torch.no_grad():
        X_tensor = torch.tensor(X, dtype=torch.float32)
        logits = model(X_tensor)
        preds = logits.argmax(dim=-1)
    return preds.numpy()


# ============================================================
# STEP 5: TOY DATA + PUTTING IT ALL TOGETHER
# ============================================================

def make_toy_data(n_samples=200, input_dim=10, n_classes=3, seed=42):
    """Simple linearly separable toy data."""
    np.random.seed(seed)
    X = np.random.randn(n_samples, input_dim).astype(np.float32)
    # Make classes separable: shift each class mean
    y = np.array([i % n_classes for i in range(n_samples)])
    for c in range(n_classes):
        X[y == c] += c * 2.0
    return X, y


if __name__ == "__main__":
    # Config
    INPUT_DIM  = 10
    HIDDEN_DIM = 32
    N_CLASSES  = 3
    BATCH_SIZE = 32
    EPOCHS     = 10
    LR         = 0.01

    # Data
    X, y = make_toy_data(n_samples=200, input_dim=INPUT_DIM, n_classes=N_CLASSES)

    # Train/val split (manual — no sklearn)
    split = int(0.8 * len(X))
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]

    train_dataset = ToyDataset(X_train, y_train)
    val_dataset   = ToyDataset(X_val, y_val)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False)

    # Model, loss, optimizer
    model     = MLP(INPUT_DIM, HIDDEN_DIM, N_CLASSES)
    criterion = nn.CrossEntropyLoss()       # softmax + NLL, expects raw logits
    optimizer = optim.SGD(model.parameters(), lr=LR, momentum=0.9)

    # Train
    print("Training...")
    train(model, train_loader, criterion, optimizer, epochs=EPOCHS)

    # Evaluate
    val_acc = evaluate(model, val_loader)
    print(f"\nValidation Accuracy: {val_acc:.3f}")

    # Predict
    predictions = predict(model, X_val)
    print(f"\nPredicted: {predictions}")
    print(f"Actual:    {y_val}")
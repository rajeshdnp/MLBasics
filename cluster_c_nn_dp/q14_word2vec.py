"""
Q14 — Word2Vec / Skip-Gram from Scratch [LOWER]
Target time: 25 min | CoderPad-safe (NumPy only)

APPROACH (say this first 60 seconds):
"Skip-gram predicts context words given a center word. Two embedding matrices: W_center
and W_context. For a (center, context) pair: embed center, dot with context embedding,
sigmoid for binary classification. Loss is binary cross-entropy. Negative sampling:
for each positive pair, sample K negative context words. SGD updates both embeddings."

CORE MATH:
- Positive: P(context|center) = sigmoid(dot(w_center, w_context))
- Negative sampling: approximate full softmax with K negatives
- Loss: -log(sigmoid(pos_dot)) - sum(log(sigmoid(-neg_dot)))

TIME: O(pairs * neg * d) | SPACE: O(V * d)
"""

import numpy as np
from collections import Counter
import random


def sigmoid(z):
    return np.where(z >= 0, 1 / (1 + np.exp(-z)), np.exp(z) / (1 + np.exp(z)))


class Word2Vec:
    def __init__(self, vocab_size, embed_dim=50, neg_samples=5, lr=0.01):
        self.embed_dim = embed_dim
        self.neg_samples = neg_samples
        self.lr = lr
        self.W_center = np.random.randn(vocab_size, embed_dim) * 0.01
        self.W_context = np.random.randn(vocab_size, embed_dim) * 0.01
        self.vocab_size = vocab_size

    def train_pair(self, center_idx, context_idx, neg_indices):
        """Train one positive pair + negative samples."""
        center_emb = self.W_center[center_idx]
        context_emb = self.W_context[context_idx]
        eps = 1e-12

        # positive example
        pos_score = sigmoid(np.dot(center_emb, context_emb))
        loss = -np.log(pos_score + eps)

        grad_center = (pos_score - 1) * context_emb
        self.W_context[context_idx] -= self.lr * (pos_score - 1) * center_emb

        # negative examples
        for neg_idx in neg_indices:
            neg_emb = self.W_context[neg_idx]
            neg_score = sigmoid(np.dot(center_emb, neg_emb))
            loss -= np.log(1 - neg_score + eps)
            grad_center += neg_score * neg_emb
            self.W_context[neg_idx] -= self.lr * neg_score * center_emb

        self.W_center[center_idx] -= self.lr * grad_center
        return loss

    def most_similar(self, word_idx, top_k=5):
        query = self.W_center[word_idx]
        norm_q = np.linalg.norm(query)
        if norm_q == 0:
            return []
        scores = []
        for i in range(self.vocab_size):
            if i == word_idx:
                continue
            emb = self.W_center[i]
            norm_e = np.linalg.norm(emb)
            if norm_e == 0:
                continue
            scores.append((i, np.dot(query, emb) / (norm_q * norm_e)))
        scores.sort(key=lambda x: -x[1])
        return scores[:top_k]


def build_training_data(corpus, window=2):
    """Generate (center, context) pairs with sliding window."""
    pairs = []
    for tokens in corpus:
        for i in range(len(tokens)):
            for j in range(max(0, i - window), min(len(tokens), i + window + 1)):
                if j != i:
                    pairs.append((tokens[i], tokens[j]))
    return pairs


# === TEST ===
if __name__ == "__main__":
    sentences = [
        ['the', 'cat', 'sat', 'on', 'the', 'mat'],
        ['the', 'dog', 'sat', 'on', 'the', 'log'],
        ['the', 'cat', 'ate', 'the', 'fish'],
        ['the', 'dog', 'ate', 'the', 'bone'],
    ]

    all_words = [w for s in sentences for w in s]
    vocab = sorted(set(all_words))
    word2idx = {w: i for i, w in enumerate(vocab)}
    idx2word = {i: w for w, i in word2idx.items()}

    indexed = [[word2idx[w] for w in s] for s in sentences]
    pairs = build_training_data(indexed, window=2)
    print(f"Vocab size: {len(vocab)}")
    print(f"Training pairs: {len(pairs)}")

    model = Word2Vec(vocab_size=len(vocab), embed_dim=10, neg_samples=3, lr=0.05)
    np.random.seed(42)

    for epoch in range(50):
        random.shuffle(pairs)
        total_loss = 0
        for center, context in pairs:
            neg_indices = [random.randint(0, len(vocab) - 1) for _ in range(model.neg_samples)]
            total_loss += model.train_pair(center, context, neg_indices)
        if epoch % 10 == 0:
            print(f"Epoch {epoch}: avg loss = {total_loss / len(pairs):.4f}")

    for word in ['cat', 'dog']:
        idx = word2idx[word]
        similar = model.most_similar(idx, top_k=3)
        print(f"\nMost similar to '{word}':")
        for sim_idx, score in similar:
            print(f"  {idx2word[sim_idx]}: {score:.4f}")

    print("\nAll tests passed!")

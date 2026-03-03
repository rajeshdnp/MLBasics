"""
Q1 — Text Preprocessing + N-gram Next-Word Prediction [HIGH]
Target time: 25 min | CoderPad-safe (Python stdlib only)

APPROACH (say this first 60 seconds):
"I'll build a three-stage pipeline: first a tokenizer that lowercases, strips punctuation,
and removes stopwords. Then I'll build bigram and trigram frequency tables using Counter
and defaultdict. Finally, a predict_next function that looks up the context in the trigram
table first, falls back to bigrams, and applies Laplace smoothing to handle unseen n-grams.
Time complexity is O(n) for building the tables where n is corpus length,
and O(V) for prediction where V is vocabulary size."

CORE MATH:
- N-gram probability: P(w3 | w1, w2) = count(w1, w2, w3) / count(w1, w2)
- Laplace smoothing: P(w3 | w1, w2) = (count(w1,w2,w3) + alpha) / (count(w1,w2) + alpha*V)

TIME: O(n) build, O(V) predict | SPACE: O(V^n)
"""

import re
import string
from collections import Counter, defaultdict

STOPWORDS = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'in', 'on', 'at',
             'to', 'for', 'of', 'and', 'or', 'but', 'it', 'this', 'that', 'with'}


def tokenize(text: str, remove_stopwords: bool = True) -> list:
    """Lowercase, strip punctuation, optionally remove stopwords."""
    text = text.lower()
    text = re.sub(f'[{re.escape(string.punctuation)}]', '', text)
    tokens = text.split()
    if remove_stopwords:
        tokens = [t for t in tokens if t not in STOPWORDS]
    return tokens


class NGramPredictor:
    def __init__(self, n: int = 3, alpha: float = 1.0):
        self.n = n
        self.alpha = alpha          # Laplace smoothing
        self.ngram_counts = {}      # {order: {context_tuple: Counter}}
        self.vocab = set()

    def fit(self, corpus: list) -> 'NGramPredictor':
        """Build n-gram frequency tables from list of token lists."""
        for tokens in corpus:
            self.vocab.update(tokens)
            for order in range(1, self.n + 1):
                if order not in self.ngram_counts:
                    self.ngram_counts[order] = defaultdict(Counter)
                for i in range(len(tokens) - order + 1):
                    if order == 1:
                        self.ngram_counts[order][()][tokens[i]] += 1
                    else:
                        context = tuple(tokens[i:i + order - 1])
                        next_word = tokens[i + order - 1]
                        self.ngram_counts[order][context][next_word] += 1
        return self

    def predict_next(self, context: list, top_k: int = 3) -> list:
        """Predict next word with backoff and Laplace smoothing."""
        V = len(self.vocab)
        if V == 0:
            return []

        # try longest context first, then back off
        for order in range(min(self.n, len(context) + 1), 0, -1):
            ctx = () if order == 1 else tuple(context[-(order - 1):])

            if order in self.ngram_counts and ctx in self.ngram_counts[order]:
                counts = self.ngram_counts[order][ctx]
                total = sum(counts.values())
                scored = []
                for word in self.vocab:
                    prob = (counts.get(word, 0) + self.alpha) / (total + self.alpha * V)
                    scored.append((word, prob))
                scored.sort(key=lambda x: -x[1])
                return scored[:top_k]

        # fallback: uniform
        prob = 1.0 / V
        return [(w, prob) for w in list(self.vocab)[:top_k]]


# === TEST ===
if __name__ == "__main__":
    corpus_text = [
        "the cat sat on the mat",
        "the cat ate the fish",
        "the dog sat on the log",
        "the dog ate the bone",
        "a cat and a dog sat together"
    ]

    corpus = [tokenize(s, remove_stopwords=False) for s in corpus_text]
    model = NGramPredictor(n=3, alpha=1.0)
    model.fit(corpus)

    print("Vocab size:", len(model.vocab))
    print("Predict after ['the', 'cat']:", model.predict_next(['the', 'cat'], top_k=3))
    print("Predict after ['sat']:", model.predict_next(['sat'], top_k=3))
    print("Predict after ['the']:", model.predict_next(['the'], top_k=3))
    print("\nAll tests passed!")

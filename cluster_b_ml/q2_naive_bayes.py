"""
Q2 — Naive Bayes Text Classifier from Scratch [CRITICAL]
Target time: 20 min | CoderPad-safe (Python stdlib + NumPy)

APPROACH (say this first 60 seconds):
"I'll implement Multinomial Naive Bayes with fit/predict API. fit computes log class priors
and per-feature log-likelihoods with Laplace smoothing — log space avoids underflow on long
documents. predict returns the class with highest log-posterior = log-prior + sum of
log-likelihoods. Key insight: Naive Bayes assumes feature independence given the class,
making the joint probability factorize into a product.
Time: O(n*V) for fit, O(V) per prediction."

CORE MATH:
- Bayes: P(class|features) ~ P(class) * prod(P(feature_i|class))
- Log space: log P(class|x) = log P(class) + sum(log P(x_i|class))
- Laplace: P(x_i|class) = (count(x_i,class) + alpha) / (total_in_class + alpha*V)

TIME: O(n*V) fit, O(V) predict | SPACE: O(C*V)
"""

import numpy as np
from collections import Counter, defaultdict
import re
import string


class NaiveBayesClassifier:
    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha
        self.class_log_priors = {}
        self.feature_log_probs = {}
        self.vocab = set()
        self.classes = []

    def fit(self, X: list, y: list) -> 'NaiveBayesClassifier':
        """Train on bag-of-words features.
        X: list of Counters {word: count}, y: list of class labels
        """
        for doc in X:
            self.vocab.update(doc.keys())
        V = len(self.vocab)

        class_counts = Counter(y)
        n_total = len(y)
        self.classes = sorted(class_counts.keys())

        # log priors
        for cls in self.classes:
            self.class_log_priors[cls] = np.log(class_counts[cls] / n_total)

        # per-class feature log-probabilities with Laplace smoothing
        for cls in self.classes:
            cls_feature_counts = Counter()
            for doc, label in zip(X, y):
                if label == cls:
                    cls_feature_counts.update(doc)

            total_count = sum(cls_feature_counts.values())
            self.feature_log_probs[cls] = {}
            for word in self.vocab:
                count = cls_feature_counts.get(word, 0)
                self.feature_log_probs[cls][word] = np.log(
                    (count + self.alpha) / (total_count + self.alpha * V))
        return self

    def predict_log_proba(self, x: dict) -> dict:
        """Compute log-posterior for each class."""
        scores = {}
        for cls in self.classes:
            score = self.class_log_priors[cls]
            for word, count in x.items():
                if word in self.feature_log_probs[cls]:
                    score += count * self.feature_log_probs[cls][word]
            scores[cls] = score
        return scores

    def predict(self, X: list) -> list:
        """Predict class labels for a list of documents."""
        results = []
        for x in X:
            scores = self.predict_log_proba(x)
            results.append(max(scores, key=scores.get))
        return results

    def predict_single(self, x: dict) -> str:
        return self.predict([x])[0]


def text_to_bow(text: str) -> Counter:
    """Convert raw text to bag-of-words Counter."""
    text = text.lower()
    text = re.sub(f'[{re.escape(string.punctuation)}]', '', text)
    return Counter(text.split())


# === TEST ===
if __name__ == "__main__":
    train_texts = [
        "buy cheap viagra pills discount",
        "free money winner lottery claim",
        "limited offer buy now discount sale",
        "earn cash fast easy money",
        "meeting tomorrow at the office",
        "project deadline review next week",
        "team lunch at noon today",
        "quarterly report is ready for review",
    ]
    train_labels = ['spam', 'spam', 'spam', 'spam', 'ham', 'ham', 'ham', 'ham']

    X_train = [text_to_bow(t) for t in train_texts]
    clf = NaiveBayesClassifier(alpha=1.0)
    clf.fit(X_train, train_labels)

    test_texts = [
        "free discount buy cheap offer",
        "team meeting project deadline",
        "earn money fast limited offer",
    ]
    X_test = [text_to_bow(t) for t in test_texts]
    predictions = clf.predict(X_test)

    for text, pred in zip(test_texts, predictions):
        print(f"  '{text}' -> {pred}")

    scores = clf.predict_log_proba(X_test[0])
    print(f"\nLog posteriors for '{test_texts[0]}':")
    for cls, score in scores.items():
        print(f"  {cls}: {score:.4f}")

    assert predictions[0] == 'spam'
    assert predictions[1] == 'ham'
    print("\nAll tests passed!")

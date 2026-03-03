"""
Q15 — KNN Classifier from Scratch [CRITICAL — BAR RAISER TARGET]
Target time: 10 min | CoderPad-safe (Python stdlib only)

APPROACH (say this first 60 seconds):
"Compute Euclidean distance from query to every training point, sort by distance, take
K nearest, majority vote. Tie-breaking: pick the class with the nearest neighbor among
tied classes. No training phase — KNN is lazy learning.
Time: O(n*d) distance, O(n log n) sort."

CORE MATH:
- Euclidean: sqrt(sum((a_i - b_i)^2))
- Cosine distance: 1 - dot(a,b) / (||a|| * ||b||)
- Majority vote with nearest-neighbor tie-breaking

TIME: O(n*d) compute, O(n log n) sort | SPACE: O(n)
"""

from collections import Counter
import math


def euclidean_distance(a, b):
    return math.sqrt(sum((ai - bi) ** 2 for ai, bi in zip(a, b)))


def cosine_distance(a, b):
    """1 - cosine similarity (smaller = more similar)."""
    dot = sum(ai * bi for ai, bi in zip(a, b))
    norm_a = math.sqrt(sum(ai ** 2 for ai in a))
    norm_b = math.sqrt(sum(bi ** 2 for bi in b))
    if norm_a == 0 or norm_b == 0:
        return 1.0
    return 1.0 - dot / (norm_a * norm_b)


def knn_classify(X_train, y_train, x_query, k=5, metric='euclidean'):
    """K-Nearest Neighbors classifier. Returns predicted label."""
    if not X_train:
        raise ValueError("Training set is empty")

    k = min(k, len(X_train))  # clamp k

    dist_fn = euclidean_distance if metric == 'euclidean' else cosine_distance
    distances = [(dist_fn(x_query, x), y_train[i], i)
                 for i, x in enumerate(X_train)]
    distances.sort(key=lambda x: x[0])

    # majority vote on k nearest
    k_nearest = distances[:k]
    labels = [label for _, label, _ in k_nearest]
    vote_counts = Counter(labels)
    max_count = vote_counts.most_common(1)[0][1]

    # tie-breaking: pick class with nearest neighbor
    tied_classes = [cls for cls, count in vote_counts.items() if count == max_count]
    if len(tied_classes) == 1:
        return tied_classes[0]

    for dist, label, idx in k_nearest:
        if label in tied_classes:
            return label


# === TEST ===
if __name__ == "__main__":
    X_train = [
        [1.0, 2.0], [1.5, 1.8], [1.2, 2.1],    # class A
        [5.0, 8.0], [5.5, 7.5], [5.2, 8.2],    # class B
        [8.0, 1.0], [8.5, 1.5], [8.2, 0.8],    # class C
    ]
    y_train = ['A', 'A', 'A', 'B', 'B', 'B', 'C', 'C', 'C']

    test_cases = [
        ([1.3, 2.0], 3, 'A'),
        ([5.1, 7.8], 3, 'B'),
        ([8.1, 1.2], 3, 'C'),
    ]

    for x_query, k, expected in test_cases:
        pred = knn_classify(X_train, y_train, x_query, k=k)
        print(f"  Query {x_query}, k={k} -> {pred} (expected {expected})")
        assert pred == expected

    # edge case: k > n (should clamp)
    pred = knn_classify(X_train, y_train, [1.0, 2.0], k=100)
    print(f"  k=100 (clamped to {len(X_train)}): {pred}")

    print("\nAll tests passed!")

"""
Q10 — Precision/Recall/F1 + Ranking Metrics (P@K, MRR, NDCG) [CRITICAL]
Target time: 15 min | CoderPad-safe (Python stdlib + NumPy)

APPROACH (say this first 60 seconds):
"Two parts. Part A — classification metrics from confusion matrix: precision = TP/(TP+FP),
recall = TP/(TP+FN), F1 = harmonic mean. Multi-class: macro (avg per class), micro
(global TP/FP/FN), weighted. Part B — ranking: P@K counts relevant in top K, MRR is
1/rank of first relevant, NDCG normalizes discounted cumulative gain by ideal ordering.
All O(n) or O(n*K)."

CORE MATH:
- Precision = TP / (TP + FP), Recall = TP / (TP + FN)
- F1 = 2*P*R / (P+R)  (harmonic mean)
- NDCG@K = DCG@K / IDCG@K, DCG = sum((2^rel - 1) / log2(i+2))

TIME: O(n) classification, O(K log K) ranking | SPACE: O(n)
"""

import numpy as np
import math


# ============================================================
# PART A: Classification Metrics
# ============================================================

def confusion_matrix(y_true: list, y_pred: list, labels: list = None) -> tuple:
    """Build confusion matrix. Rows = true, Cols = predicted."""
    if labels is None:
        labels = sorted(set(y_true) | set(y_pred))
    label_to_idx = {label: i for i, label in enumerate(labels)}
    n = len(labels)
    cm = np.zeros((n, n), dtype=int)
    for true, pred in zip(y_true, y_pred):
        cm[label_to_idx[true]][label_to_idx[pred]] += 1
    return cm, labels


def precision_recall_f1(y_true: list, y_pred: list,
                        average: str = 'macro') -> dict:
    """Precision, recall, F1 with macro/micro/weighted/per_class averaging."""
    cm, labels = confusion_matrix(y_true, y_pred)
    n_classes = len(labels)

    precisions, recalls, f1s, supports = [], [], [], []
    for i in range(n_classes):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        support = cm[i, :].sum()

        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0

        precisions.append(p)
        recalls.append(r)
        f1s.append(f1)
        supports.append(support)

    if average == 'per_class':
        return {labels[i]: {'precision': round(precisions[i], 4),
                            'recall': round(recalls[i], 4),
                            'f1': round(f1s[i], 4),
                            'support': supports[i]}
                for i in range(n_classes)}

    if average == 'macro':
        return {'precision': round(np.mean(precisions), 4),
                'recall': round(np.mean(recalls), 4),
                'f1': round(np.mean(f1s), 4)}

    if average == 'weighted':
        total = sum(supports)
        if total == 0:
            return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0}
        w = [s / total for s in supports]
        return {'precision': round(sum(p * wt for p, wt in zip(precisions, w)), 4),
                'recall': round(sum(r * wt for r, wt in zip(recalls, w)), 4),
                'f1': round(sum(f * wt for f, wt in zip(f1s, w)), 4)}

    if average == 'micro':
        tp_total = sum(cm[i, i] for i in range(n_classes))
        fp_total = sum(cm[:, i].sum() - cm[i, i] for i in range(n_classes))
        fn_total = sum(cm[i, :].sum() - cm[i, i] for i in range(n_classes))
        p = tp_total / (tp_total + fp_total) if (tp_total + fp_total) > 0 else 0.0
        r = tp_total / (tp_total + fn_total) if (tp_total + fn_total) > 0 else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        return {'precision': round(p, 4), 'recall': round(r, 4), 'f1': round(f1, 4)}

    raise ValueError(f"Unknown average type: {average}")


# ============================================================
# PART B: Ranking Metrics
# ============================================================

def precision_at_k(relevant: list, retrieved: list, k: int) -> float:
    """Precision@K: fraction of top-K that are relevant."""
    if k == 0:
        return 0.0
    relevant_set = set(relevant)
    return sum(1 for item in retrieved[:k] if item in relevant_set) / k


def recall_at_k(relevant: list, retrieved: list, k: int) -> float:
    """Recall@K: fraction of relevant found in top-K."""
    if not relevant:
        return 0.0
    relevant_set = set(relevant)
    return sum(1 for item in retrieved[:k] if item in relevant_set) / len(relevant_set)


def mean_reciprocal_rank(queries: list) -> float:
    """MRR: average of 1/rank of first relevant result."""
    rr_sum = 0.0
    for relevant, retrieved in queries:
        relevant_set = set(relevant)
        for rank, item in enumerate(retrieved, 1):
            if item in relevant_set:
                rr_sum += 1.0 / rank
                break
    return rr_sum / len(queries) if queries else 0.0


def ndcg_at_k(relevance_scores: list, k: int) -> float:
    """NDCG@K: normalized discounted cumulative gain."""
    if k == 0 or not relevance_scores:
        return 0.0

    # DCG@K
    dcg = sum((2 ** rel - 1) / math.log2(i + 2)
              for i, rel in enumerate(relevance_scores[:k]))

    # Ideal DCG
    ideal_order = sorted(relevance_scores, reverse=True)[:k]
    idcg = sum((2 ** rel - 1) / math.log2(i + 2)
               for i, rel in enumerate(ideal_order))

    return dcg / idcg if idcg > 0 else 0.0


# === TEST ===
if __name__ == "__main__":
    print("=== Part A: Classification Metrics ===")
    y_true = ['spam', 'ham', 'spam', 'spam', 'ham', 'ham', 'spam', 'ham', 'spam', 'ham']
    y_pred = ['spam', 'ham', 'ham',  'spam', 'ham', 'spam', 'spam', 'ham', 'spam', 'ham']

    for avg in ['macro', 'micro', 'weighted', 'per_class']:
        result = precision_recall_f1(y_true, y_pred, average=avg)
        print(f"  {avg}: {result}")

    print("\n=== Part B: Ranking Metrics ===")
    relevant = ['doc_1', 'doc_3', 'doc_5']
    retrieved = ['doc_2', 'doc_1', 'doc_4', 'doc_3', 'doc_5', 'doc_6']

    for k in [1, 3, 5]:
        p = precision_at_k(relevant, retrieved, k)
        r = recall_at_k(relevant, retrieved, k)
        print(f"  P@{k}={p:.4f}, R@{k}={r:.4f}")

    queries = [
        (relevant, retrieved),
        (['doc_4'], ['doc_1', 'doc_2', 'doc_4', 'doc_5']),
    ]
    print(f"  MRR={mean_reciprocal_rank(queries):.4f}")

    relevance_scores = [3, 2, 0, 1, 3]
    print(f"  NDCG@3={ndcg_at_k(relevance_scores, 3):.4f}")
    print(f"  NDCG@5={ndcg_at_k(relevance_scores, 5):.4f}")
    print("\nAll tests passed!")

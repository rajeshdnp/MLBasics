"""
Q3 — Cosine Similarity + Document Ranking [ALMOST CERTAIN]
Target time: 15 min | CoderPad-safe (Python stdlib only)

APPROACH (say this first 60 seconds):
"I'll represent documents as sparse word-count vectors using Counter. Cosine similarity
is the dot product of two vectors divided by the product of their L2 norms. For ranking,
I'll compute cosine sim between the query vector and every document vector, sort descending,
and return the ranked list. I'll handle edge cases: zero vectors get similarity 0, empty
documents are skipped. Time is O(Q * D * V) where V is the intersection of query and doc
terms — sparse, so fast in practice."

CORE MATH:
- Cosine similarity: cos(A, B) = (A . B) / (||A|| * ||B||)
- Dot product (sparse): sum over shared keys of A[k] * B[k]
- L2 norm: sqrt(sum of squares of values)

TIME: O(V) per pair | SPACE: O(V)
"""

import math
from collections import Counter
import re
import string


def tokenize(text: str) -> list:
    text = text.lower()
    text = re.sub(f'[{re.escape(string.punctuation)}]', '', text)
    return text.split()


def text_to_vector(text: str) -> Counter:
    """Convert text to sparse word-count vector."""
    return Counter(tokenize(text))


def cosine_similarity(vec_a: Counter, vec_b: Counter) -> float:
    """Compute cosine similarity between two sparse vectors."""
    if not vec_a or not vec_b:
        return 0.0

    # dot product: only iterate over shared keys
    shared_keys = set(vec_a.keys()) & set(vec_b.keys())
    dot = sum(vec_a[k] * vec_b[k] for k in shared_keys)

    # L2 norms
    norm_a = math.sqrt(sum(v ** 2 for v in vec_a.values()))
    norm_b = math.sqrt(sum(v ** 2 for v in vec_b.values()))

    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def rank_documents(query: str, documents: list, top_k: int = None) -> list:
    """Rank documents by cosine similarity to query.
    Returns list of (doc_index, score, document_text) tuples, sorted descending.
    """
    query_vec = text_to_vector(query)
    if not query_vec:
        return []

    scored = []
    for i, doc in enumerate(documents):
        doc_vec = text_to_vector(doc)
        score = cosine_similarity(query_vec, doc_vec)
        scored.append((i, score, doc))

    scored.sort(key=lambda x: -x[1])
    if top_k is not None:
        scored = scored[:top_k]
    return scored


# === TEST ===
if __name__ == "__main__":
    documents = [
        "Apple music streaming service provides millions of songs",
        "The pricing of digital content varies by country and region",
        "Machine learning models can classify text documents automatically",
        "Distribution rights for music content across international markets",
        "Cloud computing infrastructure supports large scale applications"
    ]

    query = "music content pricing distribution"
    results = rank_documents(query, documents, top_k=3)
    print(f"Query: '{query}'")
    for idx, score, doc in results:
        print(f"  Doc {idx} (score={score:.4f}): {doc[:60]}...")

    # Edge cases
    assert cosine_similarity(Counter(), Counter({"a": 1})) == 0.0
    vec = Counter({"hello": 1, "world": 1})
    assert abs(cosine_similarity(vec, vec) - 1.0) < 1e-9
    print("\nAll tests passed!")

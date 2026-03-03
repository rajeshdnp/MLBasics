"""
Q8 — Inverted Index + BM25 Retriever [CRITICAL]
Target time: 20 min | CoderPad-safe (Python stdlib only)

APPROACH (say this first 60 seconds):
"I'll build a term-to-posting-list inverted index where each term maps to {doc_id: tf}.
For retrieval, I look up each query term's posting list, take their union for candidates,
and score using BM25 which improves on TF-IDF with document length normalization and term
frequency saturation. BM25 = sum over query terms of IDF * (tf*(k1+1)) / (tf + k1*(1-b+b*dl/avgdl)).
Building the index is O(total_tokens), querying is O(query_terms * posting_list_size)."

CORE MATH:
- IDF: log((N - df + 0.5) / (df + 0.5) + 1)
- BM25: IDF * (f*(k1+1)) / (f + k1*(1-b+b*dl/avgdl))
- k1 controls TF saturation (typically 1.2), b controls length norm (typically 0.75)

TIME: O(n) build, O(q*posting) search | SPACE: O(terms * docs)
"""

import math
import re
import string
from collections import defaultdict, Counter


def tokenize(text: str) -> list:
    text = text.lower()
    text = re.sub(f'[{re.escape(string.punctuation)}]', '', text)
    return text.split()


class InvertedIndex:
    def __init__(self):
        self.index = defaultdict(dict)   # term -> {doc_id: tf}
        self.doc_lengths = {}
        self.documents = {}
        self.num_docs = 0
        self.avg_doc_length = 0.0

    def add_document(self, doc_id: int, text: str) -> None:
        tokens = tokenize(text)
        self.documents[doc_id] = text
        self.doc_lengths[doc_id] = len(tokens)
        self.num_docs += 1
        self.avg_doc_length = sum(self.doc_lengths.values()) / self.num_docs

        tf_counts = Counter(tokens)
        for term, count in tf_counts.items():
            self.index[term][doc_id] = count

    def build_from_corpus(self, corpus: list) -> 'InvertedIndex':
        for i, doc in enumerate(corpus):
            self.add_document(i, doc)
        return self

    def idf(self, term: str) -> float:
        """BM25 IDF: log((N - df + 0.5) / (df + 0.5) + 1)."""
        df = len(self.index.get(term, {}))
        N = self.num_docs
        return math.log((N - df + 0.5) / (df + 0.5) + 1)

    def bm25_score(self, query: str, doc_id: int,
                   k1: float = 1.2, b: float = 0.75) -> float:
        """BM25 score for a single document against query."""
        query_terms = tokenize(query)
        dl = self.doc_lengths.get(doc_id, 0)
        avgdl = self.avg_doc_length if self.avg_doc_length > 0 else 1.0
        score = 0.0

        for term in query_terms:
            if term not in self.index or doc_id not in self.index[term]:
                continue
            tf = self.index[term][doc_id]
            idf_val = self.idf(term)
            numerator = tf * (k1 + 1)
            denominator = tf + k1 * (1 - b + b * dl / avgdl)
            score += idf_val * numerator / denominator
        return score

    def search(self, query: str, top_k: int = 5) -> list:
        """Search and return ranked results as (doc_id, score, text)."""
        query_terms = tokenize(query)
        if not query_terms:
            return []

        # gather candidates: UNION of posting lists
        candidate_docs = set()
        for term in query_terms:
            if term in self.index:
                candidate_docs.update(self.index[term].keys())

        if not candidate_docs:
            return []

        scored = []
        for doc_id in candidate_docs:
            score = self.bm25_score(query, doc_id)
            scored.append((doc_id, score, self.documents[doc_id]))

        scored.sort(key=lambda x: -x[1])
        return scored[:top_k]


# === TEST ===
if __name__ == "__main__":
    corpus = [
        "Apple has distribution rights for music in over 175 countries",
        "The pricing model for streaming services varies by region",
        "Rights management requires tracking content across multiple territories",
        "Machine learning can automate contract review and pricing analysis",
        "International distribution of digital content involves complex licensing",
        "Apple Music pricing tiers include individual family and student plans"
    ]

    idx = InvertedIndex().build_from_corpus(corpus)

    query = "music pricing distribution rights"
    results = idx.search(query, top_k=3)
    print(f"Query: '{query}'")
    for doc_id, score, text in results:
        print(f"  Doc {doc_id} (BM25={score:.4f}): {text[:70]}...")

    print(f"\nIndex stats: {idx.num_docs} docs, {len(idx.index)} unique terms")
    print(f"Avg doc length: {idx.avg_doc_length:.1f} tokens")

    # Verify non-empty results
    assert len(results) > 0
    assert results[0][1] >= results[1][1]  # sorted descending
    print("\nAll tests passed!")

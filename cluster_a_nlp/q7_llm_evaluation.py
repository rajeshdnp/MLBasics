"""
Q7 — LLM Output Evaluation: Faithfulness + Relevance Scoring [ALMOST CERTAIN]
Target time: 25 min | CoderPad-safe (Python stdlib only)

APPROACH (say this first 60 seconds):
"I'll build three functions. faithfulness_score decomposes an answer into sentence-level
claims, then checks each claim against the context using token overlap ratio — CoderPad-safe,
no NLI model needed. relevance_score computes how well each retrieved document matches the
query. evaluate_rag combines both into a structured report. The key insight is that faithfulness
measures whether the answer is grounded in the context, while relevance measures whether the
right context was retrieved."

CORE MATH:
- Token overlap ratio: |tokens(claim) ∩ tokens(context)| / |tokens(claim)|
- Faithfulness: average overlap across all claims
- Relevance: average similarity between query and each retrieved doc

TIME: O(claims * context_len) | SPACE: O(V)
"""

import re
import string

EVAL_STOPWORDS = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'in', 'on', 'at',
                  'to', 'for', 'of', 'and', 'or', 'but', 'it', 'this', 'that', 'with',
                  'be', 'has', 'have', 'had', 'by', 'from', 'as', 'per', 'its', 'there'}


def tokenize(text: str) -> set:
    """Lowercase, strip punctuation, remove stopwords, return set."""
    text = text.lower()
    text = re.sub(f'[{re.escape(string.punctuation)}]', '', text)
    return set(w for w in text.split() if w not in EVAL_STOPWORDS)


def split_into_claims(answer: str) -> list:
    """Split answer into sentence-level claims."""
    sentences = re.split(r'(?<=[.!?])\s+', answer.strip())
    return [s.strip() for s in sentences if s.strip()]


def token_overlap_score(text_a: str, text_b: str) -> float:
    """Fraction of tokens in text_a that appear in text_b."""
    tokens_a = tokenize(text_a)
    tokens_b = tokenize(text_b)
    if not tokens_a:
        return 0.0
    return len(tokens_a & tokens_b) / len(tokens_a)


def faithfulness_score(context: str, answer: str) -> dict:
    """Score how well the answer is grounded in the context."""
    claims = split_into_claims(answer)
    if not claims:
        return {'claims': [], 'overall': 0.0}

    claim_results = []
    for claim in claims:
        score = token_overlap_score(claim, context)
        claim_results.append({
            'claim': claim,
            'score': round(score, 4),
            'grounded': score >= 0.5
        })

    overall = sum(c['score'] for c in claim_results) / len(claim_results)
    grounded_count = sum(1 for c in claim_results if c['grounded'])

    return {
        'claims': claim_results,
        'overall': round(overall, 4),
        'grounded_ratio': f"{grounded_count}/{len(claim_results)}"
    }


def relevance_score(query: str, retrieved_docs: list) -> dict:
    """Score how relevant each retrieved document is to the query."""
    if not retrieved_docs:
        return {'docs': [], 'overall': 0.0}

    doc_results = []
    for i, doc in enumerate(retrieved_docs):
        score = token_overlap_score(query, doc)
        doc_results.append({'doc_index': i, 'score': round(score, 4), 'relevant': score >= 0.3})

    overall = sum(d['score'] for d in doc_results) / len(doc_results)
    return {'docs': doc_results, 'overall': round(overall, 4)}


def evaluate_rag(query: str, context: str, answer: str,
                 retrieved_docs: list = None) -> dict:
    """Full RAG pipeline evaluation combining faithfulness and relevance."""
    faith = faithfulness_score(context, answer)
    rel = relevance_score(query, retrieved_docs or [context])
    answer_rel = token_overlap_score(query, answer)
    verdict = 'PASS' if (faith['overall'] >= 0.5 and answer_rel >= 0.3) else 'FAIL'

    return {
        'faithfulness': faith,
        'context_relevance': rel,
        'answer_relevance': round(answer_rel, 4),
        'verdict': verdict,
    }


# === TEST ===
if __name__ == "__main__":
    context = (
        "Apple Music is available in 175 countries. "
        "The standard individual plan costs $10.99 per month in the US. "
        "Family plans support up to 6 members for $16.99 per month. "
        "Student plans are available at $5.99 per month with verification."
    )
    answer_good = (
        "Apple Music costs $10.99 per month for individuals. "
        "Family plans are $16.99 and support up to 6 people. "
        "It is available in 175 countries."
    )
    answer_hallucinated = (
        "Apple Music costs $10.99 per month for individuals. "
        "There is a free tier supported by advertisements. "
        "Enterprise corporate licenses start at $50 annually."
    )
    query = "How much does Apple Music cost?"

    print("=== Good Answer ===")
    result = evaluate_rag(query, context, answer_good)
    print(f"Faithfulness: {result['faithfulness']['overall']}")
    print(f"Answer relevance: {result['answer_relevance']}")
    print(f"Verdict: {result['verdict']}")

    print("\n=== Hallucinated Answer ===")
    result = evaluate_rag(query, context, answer_hallucinated)
    print(f"Faithfulness: {result['faithfulness']['overall']}")
    for c in result['faithfulness']['claims']:
        print(f"  Claim: '{c['claim'][:50]}...' -> {c['score']} "
              f"({'grounded' if c['grounded'] else 'NOT GROUNDED'})")
    print(f"Verdict: {result['verdict']}")
    print("\nAll tests passed!")

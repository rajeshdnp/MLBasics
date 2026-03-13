# MLBasics

A collection of 17 core ML/NLP algorithms implemented from scratch in Python. Each implementation is self-contained, minimal, and designed for learning and quick reference.

## What's Inside

### Cluster A — NLP Pipeline
| File | Algorithm | Key Concepts |
|------|-----------|-------------|
| `q1_ngram_predictor.py` | N-gram Language Model | Tokenization, Laplace smoothing, backoff |
| `q3_cosine_similarity.py` | Cosine Similarity & Document Ranking | Sparse vectors, L2 norm, ranking |
| `q7_llm_evaluation.py` | LLM Output Evaluation | Faithfulness, relevance, RAG evaluation |
| `q8_bm25_retriever.py` | Inverted Index + BM25 | Posting lists, IDF, TF saturation |
| `q9_chunking_embedding.py` | Document Chunking + Vector Store | Sentence-boundary chunking, embeddings, retrieval |
| `q10_metrics.py` | Classification & Ranking Metrics | P/R/F1, P@K, MRR, NDCG |
| `q17_feature_extraction.py` | Feature Extraction Pipeline | Regex NER, keyword extraction, entity detection |

### Cluster B — ML Algorithms
| File | Algorithm | Key Concepts |
|------|-----------|-------------|
| `q2_naive_bayes.py` | Naive Bayes Text Classifier | Log-probabilities, Laplace smoothing, BoW |
| `q4_mlp_backprop.py` | MLP Forward + Backprop (NumPy) | Chain rule, He init, SGD |
| `q4_1_mlp_backprop_pytorch.py` | MLP Forward + Backprop (PyTorch) | nn.Module, autograd, CrossEntropyLoss |
| `q5_kmeans.py` | K-Means Clustering | Euclidean distance, centroid update, convergence |
| `q6_logistic_regression.py` | Logistic Regression | Sigmoid, BCE loss, gradient descent, L2 reg |
| `q12_softmax_crossentropy.py` | Softmax + Cross-Entropy | Numerical stability, subtract-max trick |
| `q15_knn.py` | K-Nearest Neighbors | Distance metrics, majority vote, tie-breaking |

### Cluster C — Neural Network Building Blocks + DP
| File | Algorithm | Key Concepts |
|------|-----------|-------------|
| `q11_layer_norm.py` | Layer Normalization | Mean/variance normalization, gamma/beta |
| `q13_attention.py` | Scaled Dot-Product Attention | QKV, causal mask, multi-head attention |
| `q14_word2vec.py` | Word2Vec / Skip-Gram | Negative sampling, context windows, embeddings |
| `q16_edit_distance.py` | Edit Distance (Levenshtein) | Dynamic programming, backtracking, space optimization |

## Getting Started

```bash
# Clone
git clone git@github.com:rajeshdnp/MLBasics.git
cd MLBasics

# Run any file directly — each is self-contained with tests
python3 cluster_a_nlp/q3_cosine_similarity.py
python3 cluster_b_ml/q15_knn.py
python3 cluster_c_nn_dp/q16_edit_distance.py
```

## Requirements

- **Python 3.7+**
- **NumPy** (for matrix-based implementations)
- **PyTorch** (only for `q4_1_mlp_backprop_pytorch.py`)

Most files use only Python stdlib + NumPy. No sklearn or other ML libraries.

## Project Structure

```
MLBasics/
├── cluster_a_nlp/          # NLP pipeline algorithms
├── cluster_b_ml/           # Core ML algorithms
├── cluster_c_nn_dp/        # Neural network blocks + dynamic programming
├── CLAUDE.md               # Claude Code project config
└── README.md
```

## Each File Includes

- **Approach statement** — plain-English explanation of the algorithm
- **Core math** — key formulas and equations
- **Time/Space complexity** — Big-O analysis
- **Clean implementation** — minimal, readable code
- **Test cases** — with assertions to verify correctness
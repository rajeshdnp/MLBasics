# MLBasics — Apple ML/NLP Coding Interview Practice

## Project Overview
17 ML/NLP interview coding implementations for Sr. Applied AI/ML Engineer position.
All code is **CoderPad-safe**: Python stdlib + NumPy only (no sklearn, torch, or network access).

## Structure
```
cluster_a_nlp/       # NLP Pipeline questions (Q1, Q3, Q7, Q8, Q9, Q10, Q17)
cluster_b_ml/        # ML Algorithm questions (Q2, Q4, Q5, Q6, Q12, Q15)
cluster_c_nn_dp/     # NN Building Blocks + DP (Q11, Q13, Q14, Q16)
```

## How to Run
Each file is self-contained with tests:
```bash
python3 cluster_a_nlp/q3_cosine_similarity.py
python3 cluster_b_ml/q15_knn.py
```

## Conventions
- Each file starts with the interview APPROACH (first 60 seconds verbal)
- CORE MATH formulas listed in the docstring
- TIME/SPACE complexity noted at top
- Tests with assertions at bottom
- Code is minimal — designed to be memorized and written in interview

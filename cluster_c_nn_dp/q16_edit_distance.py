"""
Q16 — Edit Distance / Levenshtein (DP) [HIGH]
Target time: 20 min | CoderPad-safe (Python stdlib only)

APPROACH (say this first 60 seconds):
"Classic 2D dynamic programming. dp[i][j] = min edits to transform word1[:i] into word2[:j].
Three operations: insert (dp[i][j-1]+1), delete (dp[i-1][j]+1), replace (dp[i-1][j-1]+cost
where cost=0 if chars match, else 1). Fill bottom-up. I'll also implement backtracking for
edit operations and a space-optimized O(n) version."

CORE MATH:
- dp[i][j] = dp[i-1][j-1] if word1[i-1] == word2[j-1]
- dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])  otherwise
- Base: dp[i][0] = i (delete all), dp[0][j] = j (insert all)

TIME: O(m*n) | SPACE: O(m*n) or O(n) optimized
"""


def edit_distance(word1: str, word2: str) -> int:
    """Minimum edit distance (Levenshtein). Insert/delete/replace each cost 1."""
    m, n = len(word1), len(word2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    # base cases
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if word1[i - 1] == word2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(
                    dp[i - 1][j],       # delete
                    dp[i][j - 1],       # insert
                    dp[i - 1][j - 1]    # replace
                )
    return dp[m][n]


def edit_distance_with_ops(word1: str, word2: str) -> tuple:
    """Edit distance + backtracking to reconstruct operations."""
    m, n = len(word1), len(word2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if word1[i - 1] == word2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])

    # backtrack
    ops = []
    i, j = m, n
    while i > 0 or j > 0:
        if i > 0 and j > 0 and word1[i - 1] == word2[j - 1]:
            i -= 1; j -= 1
        elif i > 0 and j > 0 and dp[i][j] == dp[i - 1][j - 1] + 1:
            ops.append(f"replace '{word1[i-1]}' -> '{word2[j-1]}' at pos {i-1}")
            i -= 1; j -= 1
        elif j > 0 and dp[i][j] == dp[i][j - 1] + 1:
            ops.append(f"insert '{word2[j-1]}' at pos {i}")
            j -= 1
        elif i > 0 and dp[i][j] == dp[i - 1][j] + 1:
            ops.append(f"delete '{word1[i-1]}' at pos {i-1}")
            i -= 1
    ops.reverse()
    return dp[m][n], ops


def edit_distance_optimized(word1: str, word2: str) -> int:
    """Space-optimized O(n) version using two rows."""
    m, n = len(word1), len(word2)
    if m < n:
        word1, word2 = word2, word1
        m, n = n, m

    prev = list(range(n + 1))
    curr = [0] * (n + 1)

    for i in range(1, m + 1):
        curr[0] = i
        for j in range(1, n + 1):
            if word1[i - 1] == word2[j - 1]:
                curr[j] = prev[j - 1]
            else:
                curr[j] = 1 + min(prev[j], curr[j - 1], prev[j - 1])
        prev, curr = curr, prev
    return prev[n]


# === TEST ===
if __name__ == "__main__":
    test_cases = [
        ("kitten", "sitting", 3),
        ("intention", "execution", 5),
        ("", "abc", 3),
        ("abc", "abc", 0),
        ("abc", "", 3),
    ]

    for w1, w2, expected in test_cases:
        dist = edit_distance(w1, w2)
        dist_opt = edit_distance_optimized(w1, w2)
        assert dist == dist_opt == expected, f"FAIL: '{w1}'->'{w2}': got {dist}, expected {expected}"
        print(f"  '{w1}' -> '{w2}': distance = {dist}")

    # show operations
    dist, ops = edit_distance_with_ops("kitten", "sitting")
    print(f"\nOperations for 'kitten' -> 'sitting' (distance={dist}):")
    for op in ops:
        print(f"  {op}")

    print("\nAll tests passed!")

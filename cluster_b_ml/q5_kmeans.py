"""
Q5 — K-Means Clustering from Scratch [HIGH]
Target time: 15 min | CoderPad-safe (NumPy only)

APPROACH (say this first 60 seconds):
"Random centroid initialization, iteratively assign each point to nearest centroid by
Euclidean distance, then recompute centroids as cluster means. Converge when assignments
don't change or max iterations reached. I'll use NumPy broadcasting for vectorized distance.
Time: O(n * K * d * iterations)."

CORE MATH:
- Distance: Euclidean = sqrt(sum((x_i - c_j)^2))
- Assignment: argmin over centroids
- Update: centroid_j = mean of points assigned to j

TIME: O(n*K*d*iters) | SPACE: O(n*d + K*d)
"""

import numpy as np


def euclidean_distance(X, centroids):
    """Compute distances: X (n,d) x centroids (K,d) -> (n,K)."""
    diff = X[:, np.newaxis, :] - centroids[np.newaxis, :, :]
    return np.sqrt(np.sum(diff ** 2, axis=2))


def kmeans(X, k, max_iters=100, seed=None):
    """K-Means clustering. Returns (assignments, centroids, n_iterations)."""
    if seed is not None:
        np.random.seed(seed)

    n, d = X.shape
    k = min(k, n)

    # random init: pick k random data points
    indices = np.random.choice(n, size=k, replace=False)
    centroids = X[indices].copy()
    assignments = -np.ones(n, dtype=int)  # -1 so first iteration never falsely converges

    for iteration in range(max_iters):
        # assign each point to nearest centroid
        distances = euclidean_distance(X, centroids)
        new_assignments = np.argmin(distances, axis=1)

        # check convergence
        if np.array_equal(new_assignments, assignments):
            break
        assignments = new_assignments

        # recompute centroids
        for j in range(k):
            mask = assignments == j
            if np.sum(mask) > 0:
                centroids[j] = X[mask].mean(axis=0)

    return assignments, centroids, iteration + 1


def inertia(X, assignments, centroids):
    """Sum of squared distances to assigned centroid (elbow method)."""
    total = 0.0
    for j in range(len(centroids)):
        mask = assignments == j
        if np.sum(mask) > 0:
            total += np.sum((X[mask] - centroids[j]) ** 2)
    return total


# === TEST ===
if __name__ == "__main__":
    np.random.seed(42)
    cluster_0 = np.random.randn(30, 2) + np.array([0, 0])
    cluster_1 = np.random.randn(30, 2) + np.array([5, 5])
    cluster_2 = np.random.randn(30, 2) + np.array([10, 0])
    X = np.vstack([cluster_0, cluster_1, cluster_2])

    assignments, centroids, n_iters = kmeans(X, k=3, seed=42)
    print(f"Converged in {n_iters} iterations")
    print(f"Centroids:\n{centroids}")
    print(f"Inertia: {inertia(X, assignments, centroids):.2f}")
    print(f"Cluster sizes: {[np.sum(assignments == i) for i in range(3)]}")

    # Verify 3 clusters found
    assert len(set(assignments)) == 3
    print("\nAll tests passed!")

import numpy as np
 
from src.attention import softmax  # Week 1 stable softmax, reused
 
 
def selection_matrix(n, S):
    S = sorted(set(S))
    if len(S) == 0:
        raise ValueError("S must be nonempty")
    if S[0] < 0 or S[-1] >= n:
        raise ValueError("S contains an index outside 0..n-1")
    X = np.zeros((n, n))
    X[S, S] = 1.0
    return X, S
 
 
def simple_transformer(Q, K, V, u, S, fs=None):
    """Return (B_S, A_S, T).
 
    Shapes: B_S (n, n), A_S (n, d_v), T (n,).
    fs: None (identity), one callable for all items, or a list of n callables.
    """
    n = Q.shape[0]
    X, S = selection_matrix(n, S)
    L = (X @ Q) @ (X @ K).T
    col_mask = np.full(n, -np.inf)
    col_mask[S] = 0.0
    B = softmax(L + col_mask[None, :])
    A = B @ X @ V
    z = A @ u
    if fs is None:
        T = z.copy()
    elif callable(fs):
        T = np.array([fs(zi) for zi in z])
    else:
        assert len(fs) == n
        T = np.array([f(zi) for f, zi in zip(fs, z)])
    return B, A, T

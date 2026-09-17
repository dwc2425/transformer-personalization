import numpy as np


def softmax(A: np.ndarray) -> np.ndarray:
    """Row-wise softmax. A has shape (n, d); returns shape (n, d)."""
    A_shifted = A - np.max(A, axis=1, keepdims=True)
    exp_A = np.exp(A_shifted)
    return exp_A / np.sum(exp_A, axis=1, keepdims=True)
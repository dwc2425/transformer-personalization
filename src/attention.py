import numpy as np


def softmax(A: np.ndarray) -> np.ndarray:
    """Row-wise softmax. A has shape (n, d); returns shape (n, d)."""
    A = A - A.max(axis=1, keepdims=True)
    E = np.exp(A)
    return E / E.sum(axis=1, keepdims=True)


def attention_weights(X, Q, K) -> np.ndarray:
    """Returns W = softmax((XQ)(XK)^T), of shape (n, n)."""
    return softmax((X @ Q) @ (X @ K).T)


def self_attention(X, Q, K, V) -> np.ndarray:
    """Returns softmax((XQ)(XK)^T) XV, of shape (n, d_v)."""
    return attention_weights(X, Q, K) @ (X @ V)

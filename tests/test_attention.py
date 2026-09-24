import numpy as np
import torch
import torch.nn.functional as F

from src.attention import softmax, attention_weights, self_attention


def test_softmax_rows_sum_to_one_and_nonnegative():
    A = np.random.randn(5, 4)
    W = softmax(A)
    assert np.allclose(W.sum(axis=1), 1.0, atol=1e-10)
    assert np.all(W >= 0)


def test_output_shapes():
    for n, d_kq, d_v in [(3, 2, 2), (6, 4, 3), (10, 1, 5)]:
        X = np.random.randn(n, d_kq)
        Q = np.random.randn(d_kq, d_kq)
        K = np.random.randn(d_kq, d_kq)
        V = np.random.randn(d_kq, d_v)
        out = self_attention(X, Q, K, V)
        assert out.shape == (n, d_v)


def test_agreement_with_pytorch():
    n, d_kq, d_v = 5, 3, 4
    X = np.random.randn(n, d_kq)
    Q = np.random.randn(d_kq, d_kq)
    K = np.random.randn(d_kq, d_kq)
    V = np.random.randn(d_kq, d_v)

    ours = self_attention(X, Q, K, V)

    Xt, Qt, Kt, Vt = (torch.tensor(a, dtype=torch.float64) for a in (X, Q, K, V))
    scores = (Xt @ Qt) @ (Xt @ Kt).T
    W = F.softmax(scores, dim=-1)
    theirs = (W @ (Xt @ Vt)).numpy()

    assert np.allclose(ours, theirs, atol=1e-6)


def test_numerical_stability_no_nan():
    X = np.random.randn(4, 2) * 1000
    Q = np.random.randn(2, 2)
    K = np.random.randn(2, 2)
    W = attention_weights(X, Q, K)
    assert not np.isnan(W).any()


def test_permutation_property():
    n, d_kq, d_v = 5, 2, 3
    X = np.random.randn(n, d_kq)
    Q = np.random.randn(d_kq, d_kq)
    K = np.random.randn(d_kq, d_kq)
    V = np.random.randn(d_kq, d_v)

    perm = np.random.permutation(n)
    P = np.eye(n)[perm]

    out_orig = self_attention(X, Q, K, V)
    out_perm = self_attention(P @ X, Q, K, V)

    assert np.allclose(P @ out_orig, out_perm, atol=1e-8)

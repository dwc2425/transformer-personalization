import numpy as np
import pytest
from src.simple_transformer import simple_transformer
 
 
def rand_instance(rng, n=6, dkq=4, dv=3):
    Q = rng.normal(size=(n, dkq)); K = rng.normal(size=(n, dkq))
    V = rng.normal(size=(n, dv)); u = rng.normal(size=dv)
    S = sorted(rng.choice(n, size=rng.integers(1, n + 1), replace=False).tolist())
    return Q, K, V, u, S
 
 
@pytest.mark.parametrize("seed", range(10))
def test_weights_and_selected_rows(seed):
    rng = np.random.default_rng(seed)
    Q, K, V, u, S = rand_instance(rng)
    n = Q.shape[0]
    B, A, _ = simple_transformer(Q, K, V, u, S)
    unsel = [j for j in range(n) if j not in S]
    assert np.all(B >= 0)
    assert np.allclose(B.sum(axis=1), 1.0)
    assert np.all(B[:, unsel] == 0)
    for i in S:  # independent reference: softmax over selected j of q_i . k_j
        s = np.array([Q[i] @ K[j] for j in S])
        w = np.exp(s - s.max()); w /= w.sum()
        assert np.allclose(B[i, S], w)
        assert np.allclose(A[i], sum(wj * V[j] for wj, j in zip(w, S)))
 
 
@pytest.mark.parametrize("seed", range(10))
def test_unselected_rows_are_mean_of_selected_values(seed):
    rng = np.random.default_rng(100 + seed)
    Q, K, V, u, S = rand_instance(rng)
    n = Q.shape[0]
    _, A, _ = simple_transformer(Q, K, V, u, S)
    for i in range(n):
        if i not in S:
            assert np.allclose(A[i], V[S].mean(axis=0))
 
 
def test_singleton_returns_v_i_and_empty_rejected():
    rng = np.random.default_rng(7)
    Q, K, V, u, _ = rand_instance(rng)
    for i in range(Q.shape[0]):
        _, A, _ = simple_transformer(Q, K, V, u, [i])
        assert np.allclose(A, np.tile(V[i], (Q.shape[0], 1)))
    with pytest.raises(ValueError):
        simple_transformer(Q, K, V, u, [])
 
 
def test_full_map_identity_and_nonlinear_activations():
    rng = np.random.default_rng(11)
    Q, K, V, u, S = rand_instance(rng)
    n = Q.shape[0]
    _, A, T_id = simple_transformer(Q, K, V, u, S)
    assert T_id.shape == (n,) and A.shape == (n, V.shape[1])
    assert np.allclose(T_id, A @ u)
    fs = [lambda t: max(0.0, t - 0.1), np.tanh, lambda t: 2 * t + 1,
          lambda t: np.exp(t), lambda t: float(np.floor(t)), lambda t: t ** 3]
    _, A2, T = simple_transformer(Q, K, V, u, S, fs=fs)
    assert np.allclose(T, [f(z) for f, z in zip(fs, A2 @ u)])
    _, _, T_same = simple_transformer(Q, K, V, u, S, fs=np.tanh)
    assert np.allclose(T_same, np.tanh(A2 @ u))

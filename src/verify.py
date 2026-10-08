"""Task 3: checks that learning is real. Run: python verify.py [--udata path/to/u.data]
 
If --udata is given, the dataset-leakage check also runs on the real data.
To compare with Week 2: set WEEK2_FN below to a function (Q_H, K_H, V_H) -> E_H (numpy) from your
simple_transformer.py; it is checked against the PyTorch layer on random histories.
"""
import argparse
import math
 
import numpy as np
import torch
import torch.nn as nn
 
from data import build_dataset, load_udata
from logistic_regression import LogisticRegression
from simple_transformer_torch import SimpleTransformer
 
import os
import sys
 
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root, for `src.*`
try:
    from src.simple_transformer import simple_transformer as week2_st  # Week 2 NumPy implementation
except Exception as e:  # pragma: no cover
    week2_st = None
    print(f"[warn] Week 2 import skipped: {e}")
 
 
def np_reference(Qh, Kh, Vh):
    """Independent NumPy attention for ONE unpadded history: softmax(Q K^T) V (unscaled)."""
    s = Qh @ Kh.T
    s = s - s.max(1, keepdims=True)
    a = np.exp(s)
    a /= a.sum(1, keepdims=True)
    return a @ Vh
 
 
def check_q2():
    z = torch.tensor([[0.0, math.log(2), 0.0]], requires_grad=True)
    loss = nn.functional.cross_entropy(z, torch.tensor([1]))
    loss.backward()
    assert abs(loss.item() - math.log(2)) < 1e-6
    assert torch.allclose(z.grad, torch.tensor([[0.25, -0.5, 0.25]]), atol=1e-6)
    print(f"Q2 ok: loss={loss.item():.4f} grad={z.grad.tolist()}")
 
 
def check_attention_matches_reference(n=50, L=7):
    torch.manual_seed(0)
    m = SimpleTransformer(n)
    h = torch.randperm(n)[:L][None]
    E = m.attend(h, torch.ones_like(h, dtype=torch.bool))[0].detach().numpy()
    Qh, Kh, Vh = (p.detach().numpy()[h[0].numpy()] for p in (m.Q, m.K, m.V))
    assert np.allclose(E, np_reference(Qh, Kh, Vh), atol=1e-5)
    print("attention matches NumPy reference")
    if week2_st is not None:
        Qf, Kf, Vf = (p.detach().double().numpy() for p in (m.Q, m.K, m.V))
        u = np.random.default_rng(0).standard_normal(Vf.shape[1])
        S = h[0].tolist()
        B, A, T = week2_st(Qf, Kf, Vf, u, S)
        assert np.allclose(E, A[S], atol=1e-5), "PyTorch E_H != Week 2 A_S rows"
        unsel = np.setdiff1d(np.arange(n), S)
        assert np.allclose(A[unsel], A[unsel][0], atol=1e-8), "unselected rows should be identical"
        print("attention matches Week 2 simple_transformer (selected rows; unselected rows identical)")
    else:
        print("[warn] Week 2 comparison skipped")
 
 
def check_padding_invariance(n=50):
    torch.manual_seed(1)
    for model in (SimpleTransformer(n), LogisticRegression(n)):
        short = torch.tensor([[3, 7, 11]])
        padded = torch.tensor([[3, 7, 11, n, n, n, n]])
        a = model(short, short != n)
        b = model(padded, padded != n)
        assert torch.allclose(a, b, atol=1e-5), type(model).__name__
    print("padding invariance ok (both models)")
 
 
def check_lr_features(n=30):
    lr = LogisticRegression(n)
    h = torch.tensor([[1, 4, 9, n, n], [2, n, n, n, n]])
    x = lr.features(h, h != n)
    assert torch.allclose(x.sum(1), torch.ones(2)) and abs(x[0, 4] - 1 / 3) < 1e-6 and abs(x[1, 2] - 1.0) < 1e-6
    print("LR features ok (rows sum to 1, value 1/|H|)")
 
 
def check_learning(n=40, B=64):
    torch.manual_seed(2)
    h = torch.stack([torch.randperm(n)[:5] for _ in range(B)])
    y = (h[:, 0] + 1) % n  # learnable rule
    for model in (LogisticRegression(n), SimpleTransformer(n)):
        opt = torch.optim.Adam(model.parameters(), lr=1e-2)
        losses = []
        for _ in range(200):
            loss = nn.functional.cross_entropy(model(h, h != n), y)
            opt.zero_grad(); loss.backward()
            if not losses:
                if isinstance(model, SimpleTransformer):
                    for name in ("Q", "K", "V"):
                        g = getattr(model, name).grad
                        assert g is not None and g.abs().sum() > 0, f"no gradient to {name}"
                    for p in model.readout.parameters():
                        assert p.grad is not None and p.grad.abs().sum() > 0, "no gradient to readout"
                    gq = model.Q.grad.abs().sum(1)
                    assert gq[torch.unique(h)].sum() > 0 and gq[~torch.isin(torch.arange(n), torch.unique(h))].sum() == 0
            opt.step()
            losses.append(loss.item())
        assert losses[-1] < 0.5 * losses[0], (type(model).__name__, losses[0], losses[-1])
        print(f"{type(model).__name__}: loss {losses[0]:.3f} -> {losses[-1]:.3f}; grads reach "
              f"{'Q,K,V,readout' if isinstance(model, SimpleTransformer) else 'W,b'}")
 
 
def check_data(events, label):
    d = build_dataset(events)
    for s in ("train", "val", "test"):
        x = d[s]
        assert bool((x["hist_max_ts"] < x["target_ts"]).all())
        pad = d["pad"]
        assert bool((x["target"] < d["n"]).all()) and bool((x["hist"] <= pad).all())
        for r in range(min(len(x["target"]), 300)):  # distinct histories, <= 20
            hh = x["hist"][r][x["hist"][r] != pad]
            assert 1 <= len(hh) <= d["max_hist"] and len(set(hh.tolist())) == len(hh)
    tr_ts = d["train"]["target_ts"]
    assert tr_ts.max() <= d["meta"]["cutoff_train_end_ts"] < d["val"]["target_ts"].min()
    assert d["val"]["target_ts"].max() <= d["meta"]["cutoff_val_end_ts"] < d["test"]["target_ts"].min()
    print(f"data ok ({label}): n={d['n']}, counts={d['meta']['counts']}")
 
 
def synthetic_events(U=60, M=120, per_user=40, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for u in range(1, U + 1):
        items = rng.choice(np.arange(1, M + 1), size=per_user, replace=False)
        ts = np.sort(rng.integers(8_800_000_00, 8_900_000_00, size=per_user))
        ts[5] = ts[4]  # same-user timestamp tie
        rows += [(u, int(i), int(t)) for i, t in zip(items, ts)]
    return np.array(rows, dtype=np.int64)
 
 
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--udata", default=None)
    a = ap.parse_args()
    check_q2()
    check_lr_features()
    check_attention_matches_reference()
    check_padding_invariance()
    check_learning()
    check_data(synthetic_events(), "synthetic")
    if a.udata:
        check_data(load_udata(a.udata), "MovieLens 100K")
    print("ALL CHECKS PASSED")

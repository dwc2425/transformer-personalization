"""Week 4 Task 3 checks: shapes, labels, tiny-batch learning, gradient flow. Run: python verify_ratings.py <ml-100k dir>"""
import sys
import torch, torch.nn as nn
import rating_data as rd
from rating_models import LogisticRegression5, GenreTransformer, G, D, C
 
root = sys.argv[1]
r, m = rd.load(root)
user = rd.select_users(r)[0][0]
raw, (tr, va, te), cuts = rd.build(root, user)
x, y = (torch.tensor(a) for a in raw["train"])
 
# labels k = r - 1 and input shape
assert (y.numpy() == tr.rating.values - 1).all() and y.min() >= 0 and y.max() <= 4
assert x.shape[1] == G and set(x.unique().tolist()) <= {0.0, 1.0}
# target rating never in input: inputs are only the 19 genre flags
tm = GenreTransformer()
assert tm.tokens(x[:8]).shape == (8, G, D)
assert tm(x[:8]).shape == (8, C) and LogisticRegression5()(x[:8]).shape == (8, C)
# absent genre token == e_j exactly; present == e_j + v_j
g = torch.zeros(1, G); g[0, 1] = 1
T = tm.tokens(g)[0]
assert torch.allclose(T[0], tm.e[0]) and torch.allclose(T[1], tm.e[1] + tm.v[1])
# attention rows sum to 1
Tt = tm.tokens(x[:4]); A = torch.softmax((Tt @ tm.Wq) @ (Tt @ tm.Wk).transpose(1, 2) / D ** 0.5, -1)
assert torch.allclose(A.sum(-1), torch.ones(4, G), atol=1e-6)
print(f"user {user}: shapes ok (B,19)->(B,19,16)->(B,5); labels r-1 ok; token e/e+v ok; attention rows sum to 1")
 
# tiny-batch learning + gradients (first step)
xb, yb = x[:32], y[:32]
for model in (LogisticRegression5(), GenreTransformer()):
    torch.manual_seed(0)
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    losses = []
    for it in range(400):
        loss = nn.functional.cross_entropy(model(xb), yb)
        opt.zero_grad(); loss.backward()
        if it == 0:
            for n, p in model.named_parameters():
                assert p.grad is not None and p.grad.abs().sum() > 0, f"no grad to {n}"
            print(type(model).__name__, "grads reach:", [n for n, _ in model.named_parameters()])
        opt.step(); losses.append(loss.item())
    print(f"{type(model).__name__}: tiny-batch loss {losses[0]:.3f} -> {losses[-1]:.3f}")
    assert losses[-1] < 0.6 * losses[0]
print("ALL CHECKS PASSED")

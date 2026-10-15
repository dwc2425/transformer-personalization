"""Week 4: train LR and genre-transformer for one user's ratings, evaluate on test once.
 
    python train_ratings.py --root <ml-100k dir> [--user 405] --out runs_w4
Selection rule (counts + timestamps only): most ratings with >= 10 distinct timestamps.
"""
import argparse, json, os, random, time
import numpy as np
import torch
import torch.nn as nn
 
import rating_data as rd
from rating_models import LogisticRegression5, GenreTransformer
 
 
def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
 
 
def n_params(m):
    return sum(p.numel() for p in m.parameters())
 
 
@torch.no_grad()
def evaluate(model, x, y):
    model.eval()
    z = model(x)
    p = torch.softmax(z, 1)
    loss = nn.functional.cross_entropy(z, y).item()
    acc = 100 * ((p.argmax(1) == y).float().mean().item())
    rhat = (p * torch.arange(1, 6).float()).sum(1)
    rmse = ((rhat - (y + 1).float()) ** 2).mean().sqrt().item()
    return {"loss": loss, "acc": acc, "rmse": rmse}
 
 
def run(name, data, a, out_dir):
    set_seed(a.seed)
    model = LogisticRegression5() if name == "lr" else GenreTransformer()
    opt = torch.optim.Adam(model.parameters(), lr=a.lr)
    loss_fn = nn.CrossEntropyLoss()  # RAW logits
    xtr, ytr = data["train"]; N = len(ytr)
    g = torch.Generator().manual_seed(a.seed)
    os.makedirs(out_dir, exist_ok=True)
    ckpt = os.path.join(out_dir, f"{name}_best.pt")
    best, best_ep, bad, log = float("inf"), -1, 0, []
    t0 = time.time()
    for ep in range(a.epochs):
        model.train()
        perm = torch.randperm(N, generator=g)
        run_loss = 0.0
        for i in range(0, N, a.batch_size):
            idx = perm[i:i + a.batch_size]
            loss = loss_fn(model(xtr[idx]), ytr[idx])
            opt.zero_grad(); loss.backward(); opt.step()
            run_loss += loss.item() * len(idx)
        v = evaluate(model, *data["val"])
        log.append({"epoch": ep + 1, "train_loss": run_loss / N, "val_loss": v["loss"]})
        if v["loss"] < best:
            best, best_ep, bad = v["loss"], ep + 1, 0
            torch.save(model.state_dict(), ckpt)
        else:
            bad += 1
            if bad >= a.patience:
                break
    runtime = time.time() - t0
    model.load_state_dict(torch.load(ckpt))
    test = evaluate(model, *data["test"])  # once
    res = {"model": name, "params": n_params(model), "best_epoch": best_ep, "epochs_run": len(log),
           "best_val_loss": best, "test": test, "runtime_sec": runtime,
           "config": {"optimizer": "Adam", "lr": a.lr, "batch_size": a.batch_size, "max_epochs": a.epochs,
                      "patience": a.patience, "seed": a.seed, "d": 16}, "log": log}
    with open(os.path.join(out_dir, f"{name}_results.json"), "w") as f:
        json.dump(res, f, indent=2)
    print(f"[{name}] params {res['params']} best_ep {best_ep}/{len(log)} val {best:.4f} | TEST loss "
          f"{test['loss']:.4f} acc {test['acc']:.2f}% rmse {test['rmse']:.4f} | {runtime:.1f}s")
    return res
 
 
def baselines(data):
    """Reference points only: majority class and train-mean rating."""
    ytr = data["train"][1]; yte = data["test"][1]
    maj = torch.bincount(ytr, minlength=5).argmax().item()
    mean_r = (ytr + 1).float().mean().item()
    return {"majority_class_rating": maj + 1, "majority_test_acc": 100 * (yte == maj).float().mean().item(),
            "train_mean_rating": mean_r, "mean_test_rmse": ((mean_r - (yte + 1).float()) ** 2).mean().sqrt().item()}
 
 
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--user", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--patience", type=int, default=25)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="runs_w4")
    a = ap.parse_args()
    r, _ = rd.load(a.root)
    users, _ = rd.select_users(r)
    user = a.user or users[0]
    raw, (tr, va, te), cuts = rd.build(a.root, user)
    data = {k: (torch.tensor(x), torch.tensor(y)) for k, (x, y) in raw.items()}
    out_dir = os.path.join(a.out, f"user{user}")
    os.makedirs(out_dir, exist_ok=True)
    info = {"user": user, "counts": [len(tr), len(va), len(te)], "cuts": cuts, "baselines": baselines(data),
            "dataset": "MovieLens 100K (ml-100k)"}
    json.dump(info, open(os.path.join(out_dir, "split.json"), "w"), indent=2)
    print("user", user, info)
    for name in ("lr", "attn"):
        run(name, data, a, out_dir)

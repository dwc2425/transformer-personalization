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
        tr_m = evaluate(model, xtr, ytr)  # train-set metrics at end of epoch (no test data touched)
        log.append({"epoch": ep + 1, "train_loss": run_loss / N, "train_acc": tr_m["acc"],
                    "val_loss": v["loss"], "val_acc": v["acc"]})
        if v["loss"] < best:
            best, best_ep, bad = v["loss"], ep + 1, 0
            torch.save(model.state_dict(), ckpt)
        else:
            bad += 1
            # patience <= 0 disables early stopping: the full a.epochs always run
            if a.patience > 0 and bad >= a.patience:
                break
    runtime = time.time() - t0
    # Test is evaluated exactly twice, with the selection rule fixed in advance (never used to pick an epoch):
    #   last  = weights after the final epoch (what "train for N epochs" means)
    #   best  = weights at the epoch with the lowest validation loss
    test_last = evaluate(model, *data["test"])
    model.load_state_dict(torch.load(ckpt))
    test = evaluate(model, *data["test"])
    res = {"model": name, "params": n_params(model), "best_epoch": best_ep, "epochs_run": len(log),
           "best_val_loss": best, "test": test, "test_last_epoch": test_last, "runtime_sec": runtime,
           "config": {"optimizer": "Adam", "lr": a.lr, "batch_size": a.batch_size, "max_epochs": a.epochs,
                      "patience": a.patience, "seed": a.seed, "d": 16}, "log": log}
    with open(os.path.join(out_dir, f"{name}_results.json"), "w") as f:
        json.dump(res, f, indent=2)
    print(f"[{name}] params {res['params']} epochs {len(log)} best_ep {best_ep} val {best:.4f} | "
          f"TEST@best acc {test['acc']:.2f}% loss {test['loss']:.4f} rmse {test['rmse']:.4f} | "
          f"TEST@last acc {test_last['acc']:.2f}% loss {test_last['loss']:.4f} | {runtime:.1f}s")
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
    ap.add_argument("--user", type=int, default=None, help="single user; default = ALL eligible users")
    ap.add_argument("--min_ratings", type=int, default=100, help="keep users with MORE than this many ratings")
    ap.add_argument("--max_users", type=int, default=None, help="debug: cap number of users")
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--patience", type=int, default=0, help="early-stop patience; 0 = disabled, run all epochs")
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="runs_w4")
    a = ap.parse_args()
    r, _ = rd.load(a.root)
    n_all = r.user.nunique()
    eligible, _ = rd.eligible_users(r, min_ratings=a.min_ratings)
    print(f"users: {n_all} total -> {len(eligible)} eligible (> {a.min_ratings} ratings), "
          f"{n_all - len(eligible)} dropped")
    if a.user is not None:
        if a.user not in eligible:
            raise SystemExit(f"user {a.user} fails the filter (> {a.min_ratings} ratings, >= 10 timestamps)")
        users = [a.user]
    else:
        users = eligible[:a.max_users] if a.max_users else eligible
    os.makedirs(a.out, exist_ok=True)
    json.dump({"min_ratings": a.min_ratings, "n_total": n_all, "n_eligible": len(eligible), "users_run": users},
              open(os.path.join(a.out, "filter.json"), "w"), indent=2)
    for user in users:
        try:
            raw, (tr, va, te), cuts = rd.build(a.root, user)
        except AssertionError as e:  # e.g. no valid chronological cut points
            print(f"user {user}: skipped ({e})")
            continue
        data = {k: (torch.tensor(x), torch.tensor(y)) for k, (x, y) in raw.items()}
        out_dir = os.path.join(a.out, f"user{user}")
        os.makedirs(out_dir, exist_ok=True)
        info = {"user": user, "counts": [len(tr), len(va), len(te)], "cuts": cuts, "baselines": baselines(data),
                "dataset": "MovieLens 100K (ml-100k)"}
        json.dump(info, open(os.path.join(out_dir, "split.json"), "w"), indent=2)
        print("user", user, info["counts"], f"majority_test_acc {info['baselines']['majority_test_acc']:.1f}")
        for name in ("lr", "attn"):
            run(name, data, a, out_dir)

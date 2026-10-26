"""Week 4: train LR and genre-transformer for one user's ratings, evaluate on test once.
 
    python train_ratings.py --root <ml-100k dir> [--user 405] --out runs_w4
Selection rule (counts + timestamps only): most ratings with >= 10 distinct timestamps.
"""
import argparse, json, os, random, time
import numpy as np
import torch
import torch.nn as nn

torch.set_num_threads(1)  # tiny models: 1 thread each is far faster and lets runs share cores
 
import rating_data as rd
from rating_models import LogisticRegression5, GenreTransformer, LogisticRegressionSyn, GenreSynopsisTransformer
 
 
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
    model = {"lr": lambda: LogisticRegression5(), "attn": lambda: GenreTransformer(),
             "lr_syn": lambda: LogisticRegressionSyn(a.k_syn),
             "attn_syn": lambda: GenreSynopsisTransformer(a.k_syn)}[name]()
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
    ap.add_argument("--synopsis", default=None, help="CSV with title[,year],plot -> also runs lr_syn and attn_syn (E4)")
    ap.add_argument("--k_syn", type=int, default=16, help="synopsis feature dimension (TF-IDF -> SVD)")
    ap.add_argument("--max_users", type=int, default=None, help="debug: cap number of users")
    ap.add_argument("--cohort", choices=["eligible", "dropped"], default="eligible",
                    help="eligible = > min_ratings (default); dropped = the users the filter removes (for E3)")
    ap.add_argument("--shard", default="0/1", help="i/n: run every n-th user starting at i (parallel runs)")
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
    if a.cohort == "dropped":
        eligible, _ = rd.dropped_users(r, min_ratings=a.min_ratings)
        print(f"cohort=dropped: {len(eligible)} users with <= {a.min_ratings} ratings and >= 10 timestamps")
    if a.user is not None:
        if a.cohort == "eligible" and a.user not in eligible:
            raise SystemExit(f"user {a.user} fails the filter (> {a.min_ratings} ratings, >= 10 timestamps)")
        users = [a.user]
    else:
        users = eligible[:a.max_users] if a.max_users else eligible
        si, sn = (int(x) for x in a.shard.split("/"))
        users = users[si::sn]
    os.makedirs(a.out, exist_ok=True)
    syn = None
    if a.synopsis:
        _, movies = rd.load(a.root)
        syn, cov = rd.synopsis_features(movies, a.synopsis, k=a.k_syn, seed=a.seed)
        print(f"synopsis: matched {cov['matched']}/{cov['total']} movies ({100 * cov['fraction']:.1f}%)")
        json.dump(cov, open(os.path.join(a.out, "synopsis_coverage.json"), "w"), indent=2)
    json.dump({"min_ratings": a.min_ratings, "cohort": a.cohort, "n_total": n_all, "n_eligible": len(eligible),
               "shard": a.shard, "users_run": users},
              open(os.path.join(a.out, f"filter_{a.shard.replace('/', 'of')}.json"), "w"), indent=2)
    names = ("lr", "attn") + (("lr_syn", "attn_syn") if syn is not None else ())
    for user in users:
        out_dir = os.path.join(a.out, f"user{user}")
        if all(os.path.exists(os.path.join(out_dir, f"{n}_results.json")) for n in names):
            print(f"user {user}: already done, skipping (resume)")
            continue
        try:
            raw, (tr, va, te), cuts = rd.build(a.root, user, synopsis=syn)
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
        base = {k: (x[:, :19], y) for k, (x, y) in data.items()}  # genre-only view for lr / attn
        for name in names:
            run(name, data if name.endswith("_syn") else base, a, out_dir)

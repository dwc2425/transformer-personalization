"""Shared training/eval script for both models.
 
    python train.py --data data_processed.pt --model lr
    python train.py --data data_processed.pt --model attn
    python train.py --data data_processed.pt --model both
"""
import argparse
import json
import os
import random
import time
 
import numpy as np
import torch
import torch.nn as nn
 
from data import load_processed
from logistic_regression import LogisticRegression
from simple_transformer_torch import SimpleTransformer
 
 
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
 
 
def make_model(name, n, a):
    return LogisticRegression(n) if name == "lr" else SimpleTransformer(n, a.d_kq, a.d_v)
 
 
def n_params(model):
    return sum(p.numel() for p in model.parameters())
 
 
@torch.no_grad()
def evaluate(model, split, pad, bs=1024):
    """Mean CE, top-1 %, Hit@10 % over all n candidate movies. Parameters frozen."""
    model.eval()
    loss_fn = nn.CrossEntropyLoss(reduction="sum")
    N = len(split["target"])
    tot, c1, c10 = 0.0, 0, 0
    for i in range(0, N, bs):
        h, y = split["hist"][i : i + bs], split["target"][i : i + bs]
        z = model(h, h != pad)
        tot += loss_fn(z, y).item()
        top = z.topk(10, dim=1).indices
        c1 += (top[:, 0] == y).sum().item()
        c10 += (top == y[:, None]).any(1).sum().item()
    return {"loss": tot / N, "acc": 100 * c1 / N, "hit10": 100 * c10 / N}
 
 
def run(name, data, a):
    set_seed(a.seed)
    n, pad = data["n"], data["pad"]
    model = make_model(name, n, a)
    opt = torch.optim.Adam(model.parameters(), lr=a.lr)
    loss_fn = nn.CrossEntropyLoss()  # takes RAW logits
    tr = data["train"]
    N = len(tr["target"])
    g = torch.Generator().manual_seed(a.seed)
    out_dir = os.path.join(a.out, name)
    os.makedirs(out_dir, exist_ok=True)
    ckpt = os.path.join(out_dir, "best.pt")
 
    best, best_ep, bad, log = float("inf"), -1, 0, []
    t0 = time.time()
    for ep in range(a.epochs):
        model.train()
        perm = torch.randperm(N, generator=g)
        run_loss = 0.0
        for i in range(0, N, a.batch_size):
            idx = perm[i : i + a.batch_size]
            h, y = tr["hist"][idx], tr["target"][idx]
            loss = loss_fn(model(h, h != pad), y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            run_loss += loss.item() * len(idx)
        v = evaluate(model, data["val"], pad)
        log.append({"epoch": ep + 1, "train_loss": run_loss / N, "val_loss": v["loss"], "val_hit10": v["hit10"]})
        print(f"[{name}] ep {ep+1:3d} train {run_loss/N:.4f} val {v['loss']:.4f} hit10 {v['hit10']:.2f}")
        if v["loss"] < best:
            best, best_ep, bad = v["loss"], ep + 1, 0
            torch.save(model.state_dict(), ckpt)
        else:
            bad += 1
            if bad >= a.patience:
                print(f"[{name}] early stop (patience {a.patience})")
                break
    runtime = time.time() - t0
 
    model.load_state_dict(torch.load(ckpt))  # best-by-validation checkpoint
    test = evaluate(model, data["test"], pad)  # test evaluated exactly once
    res = {
        "model": name, "params": n_params(model), "n": n, "best_epoch": best_ep, "best_val_loss": best,
        "test": test, "runtime_sec": runtime,
        "config": {"optimizer": "Adam", "lr": a.lr, "batch_size": a.batch_size, "max_epochs": a.epochs,
                   "patience": a.patience, "seed": a.seed, "d_kq": a.d_kq, "d_v": a.d_v},
        "log": log,
    }
    if name == "attn":
        torch.save({"Q": model.Q.data, "K": model.K.data, "V": model.V.data}, os.path.join(out_dir, "QKV.pt"))
    with open(os.path.join(out_dir, "results.json"), "w") as f:
        json.dump(res, f, indent=2)
    print(f"[{name}] TEST loss {test['loss']:.4f} acc {test['acc']:.2f}% hit@10 {test['hit10']:.2f}% "
          f"params {res['params']:,} runtime {runtime:.1f}s")
    return res
 
 
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data_processed.pt")
    ap.add_argument("--model", choices=["lr", "attn", "both"], default="both")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--d_kq", type=int, default=16)
    ap.add_argument("--d_v", type=int, default=16)
    ap.add_argument("--out", default="runs")
    a = ap.parse_args()
    data = load_processed(a.data)
    for name in (["lr", "attn"] if a.model == "both" else [a.model]):
        run(name, data, a)

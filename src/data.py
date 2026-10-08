"""Task 1: shared dataset for both models (MovieLens 100K, time-ordered next-movie prediction).
 
Usage:
    python data.py --udata path/to/ml-100k/u.data --out data_processed.pt
 
Design:
  * Events = every (user, movie, timestamp) row; star values ignored.
  * Global time split ~70/15/15. Cutoffs fall on timestamp boundaries, so ties are never split.
  * Vocabulary = movies appearing in TRAIN events only (n = len(vocab)).
  * For each event (the target y), H = up to MAX_HIST most recent distinct in-vocab movies
    that the same user rated at STRICTLY earlier timestamps (any split; parameters are frozen
    at eval, so earlier val/test events may appear in later histories).
  * Excluded: unknown target (not in vocab) first, then empty history. Counts recorded per split.
  * hist is padded with index n (PAD); mask = hist != n.
"""
import argparse
import json
from collections import defaultdict
from datetime import datetime, timezone
 
import numpy as np
import torch
 
MAX_HIST = 20
SPLITS = ("train", "val", "test")
 
 
def load_udata(path):
    arr = np.loadtxt(path, dtype=np.int64, usecols=(0, 1, 3))  # user, item, timestamp (ignore rating)
    return arr
 
 
def time_cutoffs(ts_sorted, frac_train=0.70, frac_val=0.15):
    """Return (t_train_end, t_val_end): train = ts <= t_train_end, val = (t_train_end, t_val_end], test = rest.
    Cutoffs are timestamp values, so ties always stay together."""
    N = len(ts_sorted)
    t1 = ts_sorted[int(np.ceil(frac_train * N)) - 1]
    t2 = ts_sorted[int(np.ceil((frac_train + frac_val) * N)) - 1]
    return int(t1), int(t2)
 
 
def fmt(ts):
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
 
 
def build_dataset(events, max_hist=MAX_HIST):
    """events: int array [N,3] of (user, item, timestamp). Returns dict of tensors + metadata."""
    order = np.lexsort((events[:, 1], events[:, 0], events[:, 2]))  # by ts, then user, item (deterministic)
    ev = events[order]
    ts = ev[:, 2]
    t1, t2 = time_cutoffs(ts)
    split_id = np.where(ts <= t1, 0, np.where(ts <= t2, 1, 2))
 
    vocab = np.unique(ev[split_id == 0, 1])  # train-only vocabulary, sorted original movie IDs
    item2idx = {int(m): i for i, m in enumerate(vocab)}
    n = len(vocab)
    PAD = n
 
    # per-user chronological event lists
    by_user = defaultdict(list)
    for r in range(len(ev)):
        by_user[int(ev[r, 0])].append(r)
 
    hist = np.full((len(ev), max_hist), PAD, dtype=np.int64)
    keep = np.zeros(len(ev), dtype=bool)
    hist_max_ts = np.zeros(len(ev), dtype=np.int64)
    excluded = {s: {"total": 0, "unknown_target": 0, "empty_history": 0, "kept": 0} for s in SPLITS}
 
    for u, rows in by_user.items():  # rows already ts-ascending (global sort)
        rows_ts = ts[rows]
        for k, r in enumerate(rows):
            s = SPLITS[split_id[r]]
            excluded[s]["total"] += 1
            tgt = int(ev[r, 1])
            if tgt not in item2idx:
                excluded[s]["unknown_target"] += 1
                continue
            # first position with timestamp == this one; everything before is strictly earlier
            g = int(np.searchsorted(rows_ts, rows_ts[k], side="left"))
            h, seen, hmax = [], set(), 0
            for j in range(g - 1, -1, -1):  # most recent first
                m = int(ev[rows[j], 1])
                if m in item2idx and m not in seen:
                    seen.add(m)
                    h.append(item2idx[m])
                    hmax = max(hmax, int(rows_ts[j]))
                    if len(h) == max_hist:
                        break
            if not h:
                excluded[s]["empty_history"] += 1
                continue
            hist[r, : len(h)] = h
            hist_max_ts[r] = hmax
            keep[r] = True
            excluded[s]["kept"] += 1
 
    target = np.array([item2idx.get(int(m), -1) for m in ev[:, 1]], dtype=np.int64)
    out = {"n": n, "pad": PAD, "vocab": torch.tensor(vocab), "max_hist": max_hist}
    for sid, s in enumerate(SPLITS):
        sel = keep & (split_id == sid)
        out[s] = {
            "hist": torch.from_numpy(hist[sel]),
            "target": torch.from_numpy(target[sel]),
            "user": torch.from_numpy(ev[sel, 0]),
            "target_ts": torch.from_numpy(ts[sel]),
            "hist_max_ts": torch.from_numpy(hist_max_ts[sel]),
        }
        # leakage guard: every history strictly earlier than its target
        assert bool((out[s]["hist_max_ts"] < out[s]["target_ts"]).all()), f"future info in {s} histories"
        excluded[s]["kept"] = int(sel.sum())
    out["meta"] = {
        "n_events": int(len(ev)),
        "cutoff_train_end_ts": t1, "cutoff_val_end_ts": t2,
        "cutoff_train_end": fmt(t1), "cutoff_val_end": fmt(t2),
        "first_event": fmt(int(ts[0])), "last_event": fmt(int(ts[-1])),
        "vocab_size_n": n, "max_hist": max_hist, "counts": excluded,
    }
    return out
 
 
def example(data, split="train", i=0):
    """Human-readable history/target pair (original MovieLens IDs) for Q3."""
    d, vocab = data[split], data["vocab"]
    h = d["hist"][i]
    h = h[h != data["pad"]]
    return {
        "user": int(d["user"][i]),
        "history_movie_ids": [int(vocab[j]) for j in h],
        "target_movie_id": int(vocab[d["target"][i]]),
        "target_time": fmt(int(d["target_ts"][i])),
        "latest_history_time": fmt(int(d["hist_max_ts"][i])),
    }
 
 
def load_processed(path):
    return torch.load(path, weights_only=False)
 
 
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--udata", required=True, help="path to ml-100k/u.data")
    ap.add_argument("--out", default="data_processed.pt")
    a = ap.parse_args()
    data = build_dataset(load_udata(a.udata))
    torch.save(data, a.out)
    with open(a.out.replace(".pt", "_meta.json"), "w") as f:
        json.dump(data["meta"], f, indent=2)
    print(json.dumps(data["meta"], indent=2))
    print("example:", json.dumps(example(data, "train", 0), indent=2))

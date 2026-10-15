"""Week 4 data build: user selection, genre join, leak-free chronological split."""
import sys
import numpy as np
import pandas as pd
 
GENRES = ["unknown", "Action", "Adventure", "Animation", "Children's", "Comedy",
          "Crime", "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror",
          "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western"]
 
 
def load(root):
    r = pd.read_csv(f"{root}/u.data", sep="\t", names=["user", "item", "rating", "ts"])
    cols = ["item", "title", "date", "vdate", "url"] + GENRES
    m = pd.read_csv(f"{root}/u.item", sep="|", names=cols, encoding="latin-1")
    return r, m
 
 
def select_users(r, k=3, min_groups=10):
    """Rule fixed before fitting: most ratings, >= min_groups distinct timestamps.
    Uses only rating counts and timestamps, never model performance."""
    g = r.groupby("user").agg(n=("rating", "size"), groups=("ts", "nunique"))
    g = g[g.groups >= min_groups].sort_values(["n", "groups"], ascending=False)
    return list(g.index[:k]), g
 
 
def chrono_split(df, fr=(0.70, 0.15, 0.15)):
    """Sort by timestamp; cut only at timestamp boundaries (ties stay together)."""
    df = df.sort_values(["ts", "item"], kind="stable").reset_index(drop=True)
    n = len(df)
    ts = df.ts.values
    bounds = np.flatnonzero(np.diff(ts)) + 1  # valid cut positions
    def cut(target):
        return int(bounds[np.argmin(np.abs(bounds - target))])
    c1, c2 = cut(fr[0] * n), cut((fr[0] + fr[1]) * n)
    assert 0 < c1 < c2 < n
    tr, va, te = df.iloc[:c1], df.iloc[c1:c2], df.iloc[c2:]
    # leak check: no timestamp shared across splits
    assert tr.ts.max() < va.ts.min() and va.ts.max() < te.ts.min()
    cuts = dict(train_end_ts=int(tr.ts.max()), val_end_ts=int(va.ts.max()))
    return tr, va, te, cuts
 
 
def build(root, user):
    r, m = load(root)
    d = r[r.user == user].merge(m[["item", "title"] + GENRES], on="item", how="inner")
    assert len(d) == (r.user == user).sum(), "join dropped ratings"
    tr, va, te, cuts = chrono_split(d)
    def xy(x):
        return x[GENRES].values.astype(np.float32), (x.rating.values - 1).astype(np.int64)
    return {k: xy(v) for k, v in dict(train=tr, val=va, test=te).items()}, (tr, va, te), cuts
 
 
if __name__ == "__main__":
    root = sys.argv[1]
    r, _ = load(root)
    users, g = select_users(r)
    print("top users by rule:\n", g.loc[users])
    for u in users:
        _, (tr, va, te), cuts = build(root, u)
        print(u, "counts", len(tr), len(va), len(te), cuts)

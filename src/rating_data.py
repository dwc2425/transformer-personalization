"""Week 4 data build: user selection, genre join, leak-free chronological split."""
import re
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
 
 
def norm_title(t):
    """'Postino, Il (Postman, The) (1994)' / 'Matrix, The (1999)' -> (normalized title, year)."""
    t = t.strip()
    m = re.search(r"\((\d{4})\)\s*$", t)
    year = int(m.group(1)) if m else None
    if m:
        t = t[:m.start()].strip()
    t = re.sub(r"\([^)]*\)", "", t).strip()          # drop alternate titles
    m2 = re.match(r"^(.*),\s*(The|A|An|Il|La|Le|Les|Das|Der|Die|El|Los)$", t)
    if m2:
        t = f"{m2.group(2)} {m2.group(1)}"
    return re.sub(r"[^a-z0-9]", "", t.lower()), year


def synopsis_features(m, csv_path, k=16, seed=0):
    """Synopsis context vectors for ml-100k movies (unsupervised: text only, no ratings or labels).

    csv needs a title column and a plot/synopsis/overview column; a year column is optional.
    TF-IDF -> TruncatedSVD(k) -> L2 norm. Movies with no match get a zero vector.
    Returns (feats [max_item+1, k] float32, coverage dict)."""
    from sklearn.decomposition import TruncatedSVD
    from sklearn.feature_extraction.text import TfidfVectorizer
    df = pd.read_csv(csv_path)
    df.columns = [c.strip().lower() for c in df.columns]
    tcol = next(c for c in ("title", "name") if c in df.columns)
    pcol = next(c for c in ("plot", "synopsis", "overview", "summary") if c in df.columns)
    ycol = next((c for c in ("year", "release year", "release_year") if c in df.columns), None)
    by_ty, by_t = {}, {}
    for i, row in df.iterrows():
        if not isinstance(row[pcol], str) or not row[pcol].strip():
            continue
        yr = re.search(r"(\d{4})", str(row[ycol])) if ycol else None   # tolerate messy year values
        key, y = norm_title(str(row[tcol]) + (f" ({yr.group(1)})" if yr else ""))
        by_ty.setdefault((key, y), row[pcol])
        by_t.setdefault(key, []).append(row[pcol])
    texts = {}
    for _, row in m.iterrows():
        key, y = norm_title(row["title"])
        if (key, y) in by_ty:
            texts[int(row["item"])] = by_ty[(key, y)]
        elif len(by_t.get(key, [])) == 1:        # unambiguous title-only match
            texts[int(row["item"])] = by_t[key][0]
    feats = np.zeros((int(m["item"].max()) + 1, k), dtype=np.float32)
    if len(texts) >= k + 1:
        items = sorted(texts)
        X = TfidfVectorizer(stop_words="english", min_df=2, max_features=5000, sublinear_tf=True).fit_transform(
            [texts[i] for i in items])
        Z = TruncatedSVD(k, random_state=seed).fit_transform(X)
        Z /= np.maximum(np.linalg.norm(Z, axis=1, keepdims=True), 1e-8)
        feats[items] = Z.astype(np.float32)
    cov = {"matched": len(texts), "total": int(len(m)), "fraction": len(texts) / len(m)}
    return feats, cov


def eligible_users(r, min_ratings=100, min_groups=10):
    """Data-quality filter: keep users with MORE than min_ratings ratings (100 or fewer are dropped).
    Also requires >= min_groups distinct timestamps so a chronological 70/15/15 split is possible.
    Uses only counts and timestamps, never model performance."""
    g = r.groupby("user").agg(n=("rating", "size"), groups=("ts", "nunique"))
    g = g[(g.n > min_ratings) & (g.groups >= min_groups)].sort_values(["n", "groups"], ascending=False)
    return [int(u) for u in g.index], g


def dropped_users(r, min_ratings=100, min_groups=10):
    """Complement cohort for the filtered-vs-unfiltered comparison: users the filter removes
    (<= min_ratings ratings) that still have enough timestamps for a chronological split."""
    g = r.groupby("user").agg(n=("rating", "size"), groups=("ts", "nunique"))
    g = g[(g.n <= min_ratings) & (g.groups >= min_groups)].sort_values(["n", "groups"], ascending=False)
    return [int(u) for u in g.index], g


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
 
 
def build(root, user, synopsis=None):
    """synopsis: optional [max_item+1, k] array; its row for each movie is appended after the 19 genre
    columns, so x has shape [N, 19 + k]."""
    r, m = load(root)
    d = r[r.user == user].merge(m[["item", "title"] + GENRES], on="item", how="inner")
    assert len(d) == (r.user == user).sum(), "join dropped ratings"
    tr, va, te, cuts = chrono_split(d)
    def xy(x):
        g = x[GENRES].values.astype(np.float32)
        if synopsis is not None:
            g = np.hstack([g, synopsis[x["item"].values]])
        return g, (x.rating.values - 1).astype(np.int64)
    return {k: xy(v) for k, v in dict(train=tr, val=va, test=te).items()}, (tr, va, te), cuts
 
 
if __name__ == "__main__":
    root = sys.argv[1]
    r, _ = load(root)
    users, g = select_users(r)
    print("top users by rule:\n", g.loc[users])
    for u in users:
        _, (tr, va, te), cuts = build(root, u)
        print(u, "counts", len(tr), len(va), len(te), cuts)

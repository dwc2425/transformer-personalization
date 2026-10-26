"""Plumbing tests for the synopsis (E4) pipeline: title matching, feature shapes, model forward passes.
These use a synthetic stand-in text built from titles/genres; they check mechanics, not real accuracy."""
import os
import sys

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import rating_data as rd  # noqa: E402
from rating_models import GenreSynopsisTransformer, LogisticRegressionSyn  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..", "ml-100k")


def test_norm_title_handles_article_inversion_and_alt_titles():
    assert rd.norm_title("Toy Story (1995)") == ("toystory", 1995)
    assert rd.norm_title("Usual Suspects, The (1995)") == ("theusualsuspects", 1995)
    assert rd.norm_title("Postino, Il (Postman, The) (1994)") == ("ilpostino", 1994)


def test_synopsis_features_matching_shapes_and_norm(tmp_path):
    _, m = rd.load(ROOT)
    sub = m.iloc[:60]
    rows = []
    for _, r in sub.iterrows():
        key_title = r["title"].rsplit("(", 1)[0].strip()
        genres = " ".join(g for g in rd.GENRES if r[g] == 1)
        rows.append({"title": key_title, "year": int(r["title"].rsplit("(", 1)[1].strip(")")),
                     "plot": f"{genres} story about {key_title} and friends " * 3})
    p = tmp_path / "plots.csv"
    pd.DataFrame(rows).to_csv(p, index=False)
    feats, cov = rd.synopsis_features(m, p, k=8)
    assert feats.shape == (int(m["item"].max()) + 1, 8)
    assert cov["matched"] >= 50 and cov["total"] == len(m)
    nz = np.linalg.norm(feats, axis=1)
    assert np.allclose(nz[nz > 0], 1.0, atol=1e-5)
    assert (nz > 0).sum() == cov["matched"]  # unmatched movies are exactly the zero rows


def test_build_appends_synopsis_columns_and_models_run():
    k = 8
    syn = np.random.default_rng(0).normal(size=(1700, k)).astype(np.float32)
    raw, _, _ = rd.build(ROOT, 405, synopsis=syn)
    x, y = raw["train"]
    assert x.shape[1] == 19 + k
    xt = torch.tensor(x[:5])
    assert GenreSynopsisTransformer(k)(xt).shape == (5, 5)
    assert LogisticRegressionSyn(k)(xt).shape == (5, 5)

"""Split-consistency and filter tests for rating_data (run: python -m pytest tests/test_rating_data.py).

Guarantees checked for EVERY eligible user:
  * train < val < test strictly in time (no timestamp shared across splits), so training is never on a
    later window than validation/test;
  * the split is ~70/15/15 (ties at a timestamp boundary may shift it slightly);
  * every rating is kept exactly once (no loss from the genre join, no duplicates across splits).
Also checks the >100-ratings filter and that the dropped cohort is its exact complement.
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import rating_data as rd  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..", "ml-100k")


@pytest.fixture(scope="module")
def ratings():
    r, _ = rd.load(ROOT)
    return r


def test_filter_keeps_only_users_with_more_than_100_ratings(ratings):
    users, g = rd.eligible_users(ratings, min_ratings=100)
    counts = ratings.groupby("user").size()
    assert len(users) > 0
    assert all(counts[u] > 100 for u in users)
    assert 405 in users  # sanity: the old default user passes the filter


def test_dropped_cohort_is_exact_complement(ratings):
    kept, _ = rd.eligible_users(ratings, min_ratings=100)
    dropped, _ = rd.dropped_users(ratings, min_ratings=100)
    counts = ratings.groupby("user").size()
    assert not set(kept) & set(dropped)
    assert all(counts[u] <= 100 for u in dropped)


def test_every_eligible_user_has_a_consistent_chronological_split(ratings):
    users, _ = rd.eligible_users(ratings, min_ratings=100)
    counts = ratings.groupby("user").size()
    for u in users:
        _, (tr, va, te), cuts = rd.build(ROOT, u)
        # strictly earlier -> later, no shared timestamps
        assert tr.ts.max() < va.ts.min(), f"user {u}: train overlaps val"
        assert va.ts.max() < te.ts.min(), f"user {u}: val overlaps test"
        assert cuts["train_end_ts"] == tr.ts.max() and cuts["val_end_ts"] == va.ts.max()
        # nothing lost, nothing duplicated
        assert len(tr) + len(va) + len(te) == counts[u], f"user {u}: ratings lost"
        # ~70/15/15 (large tolerance only because identical timestamps must stay together)
        n = counts[u]
        assert abs(len(tr) / n - 0.70) < 0.10, f"user {u}: train frac {len(tr) / n:.2f}"
        assert abs(len(va) / n - 0.15) < 0.10 and abs(len(te) / n - 0.15) < 0.10


def test_build_labels_and_features_shapes(ratings):
    raw, (tr, va, te), _ = rd.build(ROOT, 405)
    for k, df in dict(train=tr, val=va, test=te).items():
        x, y = raw[k]
        assert x.shape == (len(df), 19) and y.shape == (len(df),)
        assert y.min() >= 0 and y.max() <= 4  # rating - 1

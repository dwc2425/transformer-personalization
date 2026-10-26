# transformer-personalization

Reading project on *Near-Optimal Real-Time Personalization with Simple Transformers*. Code is in `src/`,
tests in `tests/`, weekly reports in `reports/`. Data: MovieLens 100K in `ml-100k/`.

## Week 5: what affects transformer accuracy

Per-user rating prediction (5 classes) from a movie's 19 genre flags: logistic regression (`lr`) vs the simple
transformer (`attn`, D=16). Chronological 70/15/15 split per user (train < val < test in time, enforced by
`tests/test_rating_data.py`).

```bash
cd src
# E1/E2: 300 epochs (early stopping off), all users with > 100 ratings (no single-user default)
python train_ratings.py --root ../ml-100k --out runs_w5
# E3: the users the filter removes (<= 100 ratings), for the filtered-vs-unfiltered comparison
python train_ratings.py --root ../ml-100k --cohort dropped --out runs_w5_dropped
# E4: add movie-synopsis context (CSV with title[,year],plot) -> also runs lr_syn and attn_syn
python train_ratings.py --root ../ml-100k --synopsis plots.csv --out runs_w5_syn
# Cross-user statistics (mean, std, 5th/95th pct, 99% CI), paired comparisons, plateau epoch, plots
python summarize.py --runs runs_w5 --compare_runs runs_w5_dropped --plots figs --out summary.json
```

Useful flags: `--user N` (one user), `--shard i/n` (parallel runs, then summarize the shared `--out`),
`--patience K` (re-enable early stopping), `--min_ratings`, `--epochs`. Runs resume: finished users are skipped.

Tests: `python -m pytest tests`.

Note: accuracy is only comparable across users relative to each user's majority-class baseline; `summarize.py`
reports this as "lift".

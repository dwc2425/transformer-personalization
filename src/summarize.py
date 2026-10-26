"""Aggregate per-user results from train_ratings.py into cross-user accuracy statistics.

    python summarize.py --runs runs_w4 [--ci 0.99] [--out summary.json] [--plots figs/]
                        [--compare_runs runs_dropped]

Per model (lr, attn, and lr_syn / attn_syn when run with --synopsis), at the best-val epoch and the
final epoch: mean, std, 5th / 95th percentile of per-user test accuracy, bootstrap CI of the mean, and the
same for LIFT = accuracy - the user's majority-class baseline.
Also: paired model comparisons (E4: synopsis vs genre-only), the epoch where mean validation accuracy
plateaus (E2), and an unpaired cohort comparison against another run directory (E3: kept vs dropped users).
"""
import argparse, glob, json, os
import numpy as np

MODELS = ("lr", "attn", "lr_syn", "attn_syn")
PAIRS = (("attn_syn", "attn"), ("lr_syn", "lr"), ("attn", "lr"))  # (a, b): a minus b


def load(runs):
    rows = []
    for d in sorted(glob.glob(os.path.join(runs, "user*"))):
        try:
            split = json.load(open(os.path.join(d, "split.json")))
        except FileNotFoundError:
            continue
        res = {}
        for m in MODELS:
            f = os.path.join(d, f"{m}_results.json")
            if os.path.exists(f):
                res[m] = json.load(open(f))
        if "lr" not in res or "attn" not in res:
            continue  # user skipped or run incomplete
        rows.append({"user": split["user"], "n_test": split["counts"][2],
                     "majority": split["baselines"]["majority_test_acc"], "res": res})
    return rows


def boot_ci(x, level, n=10000, seed=0):
    rng = np.random.default_rng(seed)
    means = rng.choice(x, size=(n, len(x)), replace=True).mean(1)
    return [float(v) for v in np.percentile(means, [100 * (1 - level) / 2, 100 * (1 + level) / 2])]


def stats(x, level):
    x = np.asarray(x, dtype=float)
    lo, hi = boot_ci(x, level)
    return {"mean": float(x.mean()), "std": float(x.std(ddof=1)), "p5": float(np.percentile(x, 5)),
            "p50": float(np.percentile(x, 50)), "p95": float(np.percentile(x, 95)),
            "ci_lo": lo, "ci_hi": hi, "ci_level": level, "n_users": len(x)}


def paired(a, b, level):
    d = np.asarray(a, float) - np.asarray(b, float)
    s = stats(d, level)
    s.update({"frac_better": float((d > 0).mean()), "frac_worse": float((d < 0).mean()),
              "significant_at_ci": bool(s["ci_lo"] > 0 or s["ci_hi"] < 0)})
    return s


def unpaired_diff(x, y, level, n=10000, seed=0):
    rng = np.random.default_rng(seed)
    x, y = np.asarray(x, float), np.asarray(y, float)
    d = rng.choice(x, (n, len(x))).mean(1) - rng.choice(y, (n, len(y))).mean(1)
    lo, hi = np.percentile(d, [100 * (1 - level) / 2, 100 * (1 + level) / 2])
    return {"mean_diff": float(x.mean() - y.mean()), "ci_lo": float(lo), "ci_hi": float(hi), "ci_level": level,
            "significant_at_ci": bool(lo > 0 or hi < 0), "n_a": len(x), "n_b": len(y)}


def curves_for(rows, m):
    L = min(len(r["res"][m]["log"]) for r in rows)
    return np.array([[e["val_acc"] for e in r["res"][m]["log"][:L]] for r in rows])


def plateau_epoch(curves, tol=0.5, window=10):
    """optimal_epoch = argmax of the cross-user mean val-acc curve; plateau_epoch = first epoch from which the
    mean curve stays within `tol` points of its max for `window` consecutive epochs (None if never)."""
    m = curves.mean(0)
    k = int(np.argmax(m))
    for e in range(len(m) - window + 1):
        if np.all(np.abs(m[e:e + window] - m.max()) <= tol):
            return {"optimal_epoch": k + 1, "plateau_epoch": e + 1, "max_mean_val_acc": float(m.max())}
    return {"optimal_epoch": k + 1, "plateau_epoch": None, "max_mean_val_acc": float(m.max())}


def model_block(rows, m, level):
    maj = np.array([r["majority"] for r in rows])
    best = np.array([r["res"][m]["test"]["acc"] for r in rows])
    last = np.array([r["res"][m]["test_last_epoch"]["acc"] for r in rows])
    be = [r["res"][m]["best_epoch"] for r in rows]
    return {"test_acc_best_val": stats(best, level), "test_acc_last_epoch": stats(last, level),
            "lift_over_majority_best_val": stats(best - maj, level),
            "lift_over_majority_last_epoch": stats(last - maj, level),
            "best_epoch": {"mean": float(np.mean(be)), "median": float(np.median(be))},
            "val_acc_training_time": plateau_epoch(curves_for(rows, m))}


def make_plots(rows, models, outdir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs(outdir, exist_ok=True)
    maj = np.array([r["majority"] for r in rows])
    # 1) spread of per-user accuracy
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
    for m in models:
        best = np.array([r["res"][m]["test"]["acc"] for r in rows])
        ax[0].hist(best, bins=25, alpha=0.5, label=m)
        ax[1].hist(best - maj, bins=25, alpha=0.5, label=m)
    ax[0].set(title="Per-user test accuracy (best-val epoch)", xlabel="accuracy (%)", ylabel="users")
    ax[1].set(title="Lift over each user's majority baseline", xlabel="accuracy - majority (points)")
    ax[1].axvline(0, color="k", lw=0.8)
    ax[0].legend(); fig.tight_layout(); fig.savefig(os.path.join(outdir, "spread.png"), dpi=150); plt.close(fig)
    # 2) validation accuracy vs training time, with 5-95 percentile band across users
    fig, ax = plt.subplots(figsize=(6.5, 4))
    for m in models:
        c = curves_for(rows, m); x = np.arange(1, c.shape[1] + 1)
        ax.plot(x, c.mean(0), label=f"{m} mean")
        ax.fill_between(x, np.percentile(c, 5, 0), np.percentile(c, 95, 0), alpha=0.15)
    ax.set(title="Validation accuracy vs epochs (band = 5th-95th pct of users)", xlabel="epoch",
           ylabel="val accuracy (%)")
    ax.legend(); fig.tight_layout(); fig.savefig(os.path.join(outdir, "val_acc_vs_epochs.png"), dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True)
    ap.add_argument("--ci", type=float, default=0.99, help="CI level (wider than 0.95)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--plots", default=None, help="directory for spread.png and val_acc_vs_epochs.png")
    ap.add_argument("--compare_runs", default=None, help="second run dir (e.g. dropped users) for E3")
    a = ap.parse_args()
    rows = load(a.runs)
    if not rows:
        raise SystemExit("no complete user results found")
    models = [m for m in MODELS if all(m in r["res"] for r in rows)]
    maj = [r["majority"] for r in rows]
    out = {"n_users": len(rows), "ci_level": a.ci, "majority_baseline": stats(maj, a.ci),
           "models": {m: model_block(rows, m, a.ci) for m in models}, "paired": {}}
    for x, y in PAIRS:
        if x in models and y in models:
            out["paired"][f"{x}_minus_{y}"] = {
                k: paired([r["res"][x][key]["acc"] for r in rows], [r["res"][y][key]["acc"] for r in rows], a.ci)
                for k, key in (("best_val", "test"), ("last_epoch", "test_last_epoch"))}
    if a.compare_runs:
        other = load(a.compare_runs)
        if other:
            om = [m for m in models if all(m in r["res"] for r in other)]
            out["cohort_comparison"] = {"n_kept": len(rows), "n_other": len(other),
                "majority_baseline_other": stats([r["majority"] for r in other], a.ci),
                "models_other": {m: model_block(other, m, a.ci) for m in om},
                "lift_diff_kept_minus_other": {m: unpaired_diff(
                    [r["res"][m]["test"]["acc"] - r["majority"] for r in rows],
                    [r["res"][m]["test"]["acc"] - r["majority"] for r in other], a.ci) for m in om},
                "acc_std_kept_vs_other": {m: [float(np.std([r["res"][m]["test"]["acc"] for r in rows], ddof=1)),
                    float(np.std([r["res"][m]["test"]["acc"] for r in other], ddof=1))] for m in om}}
    if a.out:
        open(a.out, "w").write(json.dumps(out, indent=2))
    if a.plots:
        make_plots(rows, models, a.plots)

    print(f"users: {out['n_users']}   CI level: {a.ci}")
    b = out["majority_baseline"]
    print(f"majority baseline: mean {b['mean']:.1f}  std {b['std']:.1f}  p5 {b['p5']:.1f}  p95 {b['p95']:.1f}")
    for m, d in out["models"].items():
        for k in ("test_acc_best_val", "test_acc_last_epoch", "lift_over_majority_best_val"):
            s = d[k]
            print(f"[{m:8s}] {k:28s} mean {s['mean']:6.2f}  std {s['std']:5.2f}  p5 {s['p5']:6.2f}  "
                  f"p95 {s['p95']:6.2f}  CI[{s['ci_lo']:.2f}, {s['ci_hi']:.2f}]")
        print(f"[{m:8s}] best_epoch mean {d['best_epoch']['mean']:.1f} median {d['best_epoch']['median']:.0f}  "
              f"val-acc training time: {d['val_acc_training_time']}")
    for name, d in out["paired"].items():
        for k, s in d.items():
            print(f"[paired] {name:18s} {k:10s} mean diff {s['mean']:+6.2f}  CI[{s['ci_lo']:+.2f}, {s['ci_hi']:+.2f}]  "
                  f"better {100 * s['frac_better']:.0f}% worse {100 * s['frac_worse']:.0f}%  "
                  f"{'SIGNIFICANT' if s['significant_at_ci'] else 'n.s.'}")
    if "cohort_comparison" in out:
        c = out["cohort_comparison"]
        print(f"[cohort] kept n={c['n_kept']} vs other n={c['n_other']}")
        for m, s in c["lift_diff_kept_minus_other"].items():
            print(f"[cohort] {m:8s} lift diff (kept-other) {s['mean_diff']:+6.2f}  CI[{s['ci_lo']:+.2f}, {s['ci_hi']:+.2f}]  "
                  f"acc std kept/other {c['acc_std_kept_vs_other'][m][0]:.1f}/{c['acc_std_kept_vs_other'][m][1]:.1f}  "
                  f"{'SIGNIFICANT' if s['significant_at_ci'] else 'n.s.'}")

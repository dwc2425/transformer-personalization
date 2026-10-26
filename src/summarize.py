"""Aggregate per-user results from train_ratings.py into cross-user accuracy statistics.

    python summarize.py --runs runs_w4 [--ci 0.99] [--out summary.json]

Reports, for LR and the transformer ("attn"), at the best-val epoch and at the final epoch:
  mean, std, 5th / 95th percentile of per-user test accuracy, bootstrap CI of the mean,
  and the same for accuracy minus the user's majority-class baseline (lift).
Also: the epoch where mean validation accuracy plateaus (training-time analysis).
"""
import argparse, glob, json, os
import numpy as np


def load(runs):
    rows = []
    for d in sorted(glob.glob(os.path.join(runs, "user*"))):
        try:
            split = json.load(open(os.path.join(d, "split.json")))
            res = {m: json.load(open(os.path.join(d, f"{m}_results.json"))) for m in ("lr", "attn")}
        except FileNotFoundError:
            continue  # user skipped or run incomplete
        rows.append({"user": split["user"], "n_test": split["counts"][2],
                     "majority": split["baselines"]["majority_test_acc"], "res": res})
    return rows


def boot_ci(x, level, n=10000, seed=0):
    rng = np.random.default_rng(seed)
    means = rng.choice(x, size=(n, len(x)), replace=True).mean(1)
    lo, hi = np.percentile(means, [100 * (1 - level) / 2, 100 * (1 + level) / 2])
    return float(lo), float(hi)


def stats(x, level):
    x = np.asarray(x, dtype=float)
    lo, hi = boot_ci(x, level)
    return {"mean": float(x.mean()), "std": float(x.std(ddof=1)), "p5": float(np.percentile(x, 5)),
            "p50": float(np.percentile(x, 50)), "p95": float(np.percentile(x, 95)),
            "ci_lo": lo, "ci_hi": hi, "ci_level": level, "n_users": len(x)}


def plateau_epoch(curves, tol=0.5, window=10):
    """First epoch where the cross-user mean val-acc curve is within `tol` points of its own
    maximum AND stays so for `window` epochs; and the epoch of the maximum (optimal training time)."""
    m = np.mean(curves, axis=0)
    k = int(np.argmax(m))
    for e in range(len(m) - window + 1):
        if np.all(np.abs(m[e:e + window] - m.max()) <= tol):
            return {"optimal_epoch": k + 1, "plateau_epoch": e + 1, "max_mean_val_acc": float(m.max())}
    return {"optimal_epoch": k + 1, "plateau_epoch": None, "max_mean_val_acc": float(m.max())}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True)
    ap.add_argument("--ci", type=float, default=0.99, help="CI level for the mean (wider than 0.95)")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    rows = load(a.runs)
    if not rows:
        raise SystemExit("no complete user results found")
    out = {"n_users": len(rows), "ci_level": a.ci, "models": {}}
    maj = [r["majority"] for r in rows]
    out["majority_baseline"] = stats(maj, a.ci)
    for m in ("lr", "attn"):
        best = [r["res"][m]["test"]["acc"] for r in rows]
        last = [r["res"][m]["test_last_epoch"]["acc"] for r in rows]
        out["models"][m] = {
            "test_acc_best_val": stats(best, a.ci),
            "test_acc_last_epoch": stats(last, a.ci),
            "lift_over_majority_best_val": stats(np.array(best) - np.array(maj), a.ci),
            "lift_over_majority_last_epoch": stats(np.array(last) - np.array(maj), a.ci),
            "best_epoch": {"mean": float(np.mean([r["res"][m]["best_epoch"] for r in rows])),
                           "median": float(np.median([r["res"][m]["best_epoch"] for r in rows]))},
        }
        L = min(len(r["res"][m]["log"]) for r in rows)
        curves = [[e["val_acc"] for e in r["res"][m]["log"][:L]] for r in rows]
        out["models"][m]["val_acc_training_time"] = plateau_epoch(curves)
    txt = json.dumps(out, indent=2)
    if a.out:
        open(a.out, "w").write(txt)
    print(f"users: {out['n_users']}   CI level: {a.ci}")
    print(f"majority baseline: mean {out['majority_baseline']['mean']:.1f}  "
          f"p5 {out['majority_baseline']['p5']:.1f}  p95 {out['majority_baseline']['p95']:.1f}")
    for m, d in out["models"].items():
        for k in ("test_acc_best_val", "test_acc_last_epoch", "lift_over_majority_best_val"):
            s = d[k]
            print(f"[{m}] {k:30s} mean {s['mean']:6.2f}  std {s['std']:5.2f}  p5 {s['p5']:6.2f}  "
                  f"p95 {s['p95']:6.2f}  CI[{s['ci_lo']:.2f}, {s['ci_hi']:.2f}]")
        print(f"[{m}] best_epoch mean {d['best_epoch']['mean']:.1f} median {d['best_epoch']['median']:.0f}  "
              f"val-acc training time: {d['val_acc_training_time']}")

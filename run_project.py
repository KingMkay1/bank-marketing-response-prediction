"""Bank Marketing Response Prediction.

Dummy vs linear vs tree baselines; threshold chosen on validation only;
final numbers reported once on a held-out test set.

Usage:
    python run_project.py --data data/bank-additional-full.csv
    python run_project.py --synthetic        # smoke test only, NOT real results
"""
import argparse, json, time, platform
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, confusion_matrix,
                             precision_recall_curve, precision_score,
                             recall_score, f1_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

SEED = 42
RECALL_TARGET = 0.60  # threshold rule: highest precision with val recall >= this
OUT = Path("outputs")


def load(path):
    df = pd.read_csv(path, sep=";")
    df["y"] = (df["y"] == "yes").astype(int)
    # 'duration' is only known after the call ends -> target leakage; drop it.
    df = df.drop(columns=["duration"])
    # 999 means "not previously contacted"; make that explicit.
    df["previously_contacted"] = (df["pdays"] != 999).astype(int)
    df["pdays"] = df["pdays"].replace(999, 0)
    return df


def synthetic(n=8000):
    """Schema-compatible fake data, only to check the code runs."""
    rng = np.random.default_rng(SEED)
    df = pd.DataFrame({
        "age": rng.integers(18, 90, n),
        "job": rng.choice(["admin.", "technician", "retired", "student"], n),
        "marital": rng.choice(["married", "single", "unknown"], n),
        "education": rng.choice(["basic.9y", "university.degree", "unknown"], n),
        "default": rng.choice(["no", "unknown"], n),
        "housing": rng.choice(["yes", "no"], n),
        "loan": rng.choice(["yes", "no"], n),
        "contact": rng.choice(["cellular", "telephone"], n),
        "month": rng.choice(["may", "jul", "aug", "mar"], n),
        "day_of_week": rng.choice(["mon", "tue", "wed"], n),
        "campaign": rng.integers(1, 10, n),
        "pdays": rng.choice([999, 3, 6], n, p=[.9, .05, .05]),
        "previous": rng.integers(0, 3, n),
        "poutcome": rng.choice(["nonexistent", "failure", "success"], n),
        "emp.var.rate": rng.normal(0, 1.5, n),
        "euribor3m": rng.normal(3, 1.7, n),
    })
    logit = -2.4 + 1.5 * (df.poutcome == "success") - 0.3 * df["euribor3m"] + 0.5 * (df.job == "student")
    df["y"] = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
    df["previously_contacted"] = (df["pdays"] != 999).astype(int)
    df["pdays"] = df["pdays"].replace(999, 0)
    return df


def make_pre(X):
    cat = X.select_dtypes(exclude="number").columns.tolist()
    num = [c for c in X.columns if c not in cat]
    return ColumnTransformer([
        ("num", StandardScaler(), num),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat)])


def pick_threshold(y_val, p_val):
    """Highest precision subject to recall >= RECALL_TARGET, validation only."""
    prec, rec, thr = precision_recall_curve(y_val, p_val)
    prec, rec = prec[:-1], rec[:-1]
    ok = rec >= RECALL_TARGET
    if not ok.any():
        return float(thr[np.argmax(rec)])
    idx = np.where(ok)[0][np.argmax(prec[ok])]
    return float(thr[idx])


def metrics(y, p, t):
    pred = (p >= t).astype(int)
    return {"pr_auc": average_precision_score(y, p),
            "recall": recall_score(y, pred),
            "precision": precision_score(y, pred, zero_division=0),
            "f1": f1_score(y, pred), "threshold": t}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/bank-additional-full.csv")
    ap.add_argument("--synthetic", action="store_true")
    a = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    t0 = time.time()

    df = synthetic() if a.synthetic else load(a.data)
    X, y = df.drop(columns="y"), df["y"]
    print(f"Rows {len(df)}, positive rate {y.mean():.4f}")

    # 60 / 20 / 20 stratified split. (Dataset has no customer id, so no group split possible.)
    X_tmp, X_te, y_tmp, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)
    X_tr, X_va, y_tr, y_va = train_test_split(X_tmp, y_tmp, test_size=0.25, stratify=y_tmp, random_state=SEED)
    split_info = {k: {"n": int(len(v)), "pos_rate": float(w.mean())}
                  for k, v, w in [("train", X_tr, y_tr), ("val", X_va, y_va), ("test", X_te, y_te)]}

    # Candidate models; tuned on validation PR AUC only.
    grids = {
        "Dummy (prior)": [DummyClassifier(strategy="prior")],
        "Logistic regression": [LogisticRegression(C=c, class_weight="balanced", max_iter=2000)
                                for c in [0.01, 0.1, 1, 10]],
        "Decision tree": [DecisionTreeClassifier(max_depth=d, min_samples_leaf=l, class_weight="balanced",
                                                 random_state=SEED)
                          for d in [3, 5, 8] for l in [20, 50, 100]],
        "Random forest (extra)": [RandomForestClassifier(n_estimators=300, max_depth=d, min_samples_leaf=20,
                                                         class_weight="balanced_subsample", n_jobs=-1,
                                                         random_state=SEED) for d in [8, 12]],
    }

    rows, fitted, test_probs, val_probs = [], {}, {}, {}
    for name, cands in grids.items():
        best, best_s = None, -1
        for est in cands:
            pipe = Pipeline([("pre", make_pre(X)), ("clf", est)]).fit(X_tr, y_tr)
            s = average_precision_score(y_va, pipe.predict_proba(X_va)[:, 1])
            if s > best_s:
                best, best_s = pipe, s
        pv, pt = best.predict_proba(X_va)[:, 1], best.predict_proba(X_te)[:, 1]
        thr = 0.5 if name.startswith("Dummy") else pick_threshold(y_va, pv)
        mv, mt = metrics(y_va, pv, thr), metrics(y_te, pt, thr)
        rows.append({"model": name, "best_params": str({k: v for k, v in best["clf"].get_params().items()
                                                        if k in ("C", "max_depth", "min_samples_leaf")}),
                     **{f"val_{k}": v for k, v in mv.items() if k != "threshold"},
                     "threshold": thr, **{f"test_{k}": v for k, v in mt.items() if k != "threshold"}})
        fitted[name], val_probs[name], test_probs[name] = best, pv, pt
        print(f"{name}: val PR-AUC {mv['pr_auc']:.3f} | test PR-AUC {mt['pr_auc']:.3f} "
              f"recall {mt['recall']:.3f} precision {mt['precision']:.3f} @ t={thr:.3f}")

    res = pd.DataFrame(rows)
    res.to_csv(OUT / "results.csv", index=False)

    # Figures
    fig, ax = plt.subplots(figsize=(5, 3))
    y.value_counts().sort_index().plot.bar(ax=ax, color=["#888", "#2a7"])
    ax.set_xticklabels(["no", "yes"], rotation=0); ax.set_title("Class balance"); fig.tight_layout()
    fig.savefig(OUT / "class_balance.png", dpi=200); plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.5, 4))
    for n, p in test_probs.items():
        pr, rc, _ = precision_recall_curve(y_te, p)
        ax.plot(rc, pr, label=f"{n} (AP={average_precision_score(y_te, p):.3f})")
    ax.axhline(y_te.mean(), ls=":", c="k", lw=.8)
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision"); ax.set_title("Test PR curves"); ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(OUT / "pr_curves_test.png", dpi=200); plt.close(fig)

    sel = res.loc[res.model != "Dummy (prior)"].sort_values("val_pr_auc").iloc[-1]["model"]
    pv = val_probs[sel]
    ts = np.linspace(0.05, 0.95, 91)
    fig, ax = plt.subplots(figsize=(5.5, 4))
    ax.plot(ts, [recall_score(y_va, pv >= t) for t in ts], label="Recall")
    ax.plot(ts, [precision_score(y_va, pv >= t, zero_division=0) for t in ts], label="Precision")
    ax.plot(ts, [f1_score(y_va, pv >= t) for t in ts], label="F1")
    ax.axvline(float(res.loc[res.model == sel, "threshold"].iloc[0]), ls="--", c="k", label="chosen (validation)")
    ax.set_xlabel("Threshold"); ax.set_title(f"Threshold trade-off on validation: {sel}"); ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(OUT / "threshold_tradeoff_val.png", dpi=200); plt.close(fig)

    t = float(res.loc[res.model == sel, "threshold"].iloc[0])
    cm = confusion_matrix(y_te, (test_probs[sel] >= t).astype(int))
    pd.DataFrame(cm, index=["true no", "true yes"], columns=["pred no", "pred yes"]).to_csv(OUT / "confusion_test.csv")

    json.dump({"split": split_info, "seed": SEED, "recall_target": RECALL_TARGET,
               "selected_model": sel, "runtime_s": round(time.time() - t0, 1),
               "python": platform.python_version(), "machine": platform.processor() or platform.machine(),
               "synthetic": a.synthetic}, open(OUT / "run_info.json", "w"), indent=2)
    print(res.round(3).to_string(index=False))
    print("Confusion matrix (test):\n", cm)


if __name__ == "__main__":
    main()

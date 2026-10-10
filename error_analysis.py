"""Group-level error analysis for the Bank Marketing Response Prediction project.

Recreates the project's stratified 60/20/20 split and model-selection procedure,
checks the selected random-forest threshold against outputs/results.csv, and
writes outputs/error_by_group.csv.

Run from the repository root:
    python error_analysis.py --data data/bank-additional-full.csv
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, precision_recall_curve
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

SEED = 42
RECALL_TARGET = 0.60
OUT = Path("outputs")


def load_data(path):
    df = pd.read_csv(path, sep=";")
    df["y"] = (df["y"] == "yes").astype(int)
    # Match run_project.py exactly.
    df = df.drop(columns=["duration"])
    df["previously_contacted"] = (df["pdays"] != 999).astype(int)
    df["pdays"] = df["pdays"].replace(999, 0)
    return df


def make_preprocessor(X):
    cat = X.select_dtypes(exclude="number").columns.tolist()
    num = [c for c in X.columns if c not in cat]
    return ColumnTransformer([
        ("num", StandardScaler(), num),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat),
    ])


def choose_threshold(y_val, probabilities):
    precision, recall, thresholds = precision_recall_curve(y_val, probabilities)
    precision, recall = precision[:-1], recall[:-1]
    eligible = recall >= RECALL_TARGET
    if not eligible.any():
        return float(thresholds[np.argmax(recall)])
    idx = np.where(eligible)[0][np.argmax(precision[eligible])]
    return float(thresholds[idx])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/bank-additional-full.csv")
    parser.add_argument("--small-group-min", type=int, default=100,
                        help="Groups with fewer test rows than this are marked small_group=True.")
    args = parser.parse_args()

    OUT.mkdir(exist_ok=True)
    df = load_data(args.data)
    X = df.drop(columns="y")
    y = df["y"]

    # Match the exact split sequence and seed in run_project.py.
    X_tmp, X_test, y_tmp, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=SEED
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_tmp, y_tmp, test_size=0.25, stratify=y_tmp, random_state=SEED
    )

    # Recreate the random-forest candidate search in run_project.py.
    best_pipe = None
    best_val_ap = -1.0
    best_params = None
    for depth in [8, 12]:
        estimator = RandomForestClassifier(
            n_estimators=300,
            max_depth=depth,
            min_samples_leaf=20,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=SEED,
        )
        pipe = Pipeline([
            ("pre", make_preprocessor(X)),
            ("clf", estimator),
        ])
        pipe.fit(X_train, y_train)
        p_val = pipe.predict_proba(X_val)[:, 1]
        val_ap = average_precision_score(y_val, p_val)
        if val_ap > best_val_ap:
            best_pipe = pipe
            best_val_ap = val_ap
            best_params = {"max_depth": depth, "min_samples_leaf": 20}

    p_val = best_pipe.predict_proba(X_val)[:, 1]
    selected_threshold = choose_threshold(y_val.to_numpy(), p_val)

    # Compare the recreated threshold with the recorded result.
    results_path = OUT / "results.csv"
    if not results_path.exists():
        raise FileNotFoundError(
            "outputs/results.csv was not found. Run run_project.py first."
        )
    results = pd.read_csv(results_path)
    rf_rows = results[results["model"] == "Random forest (additional)"]
    if rf_rows.empty:
        raise ValueError("Random forest row not found in outputs/results.csv.")
    recorded_threshold = float(rf_rows.iloc[0]["threshold"])
    difference = abs(selected_threshold - recorded_threshold)
    matches = difference <= 1e-8

    print(f"Selected random forest parameters: {best_params}")
    print(f"Validation PR AUC: {best_val_ap:.12f}")
    print(f"Recreated threshold: {selected_threshold:.12f}")
    print(f"results.csv threshold: {recorded_threshold:.12f}")
    print(f"Threshold check: {'(match)' if matches else 'MISMATCH'}")
    if not matches:
        print("WARNING: Do not use these group findings in the report until the mismatch is resolved.")

    # Score the held-out test set using the validation-selected threshold.
    p_test = best_pipe.predict_proba(X_test)[:, 1]
    y_true = y_test.to_numpy()
    y_pred = (p_test >= selected_threshold).astype(int)
    test = X_test.copy()
    test["actual"] = y_true
    test["predicted"] = y_pred

    # Human-readable age groups for an additional useful segment comparison.
    if "age" in test.columns:
        test["age_group"] = pd.cut(
            test["age"],
            bins=[-np.inf, 29, 39, 49, 59, np.inf],
            labels=["18-29", "30-39", "40-49", "50-59", "60+"],
        ).astype("object")

    group_columns = [
        c for c in [
            "age_group", "job", "education", "contact", "month",
            "poutcome", "previously_contacted", "housing", "loan",
        ] if c in test.columns
    ]

    rows = []
    for column in group_columns:
        for group_value, part in test.groupby(column, dropna=False, observed=True):
            actual = part["actual"].to_numpy()
            predicted = part["predicted"].to_numpy()
            tp = int(((actual == 1) & (predicted == 1)).sum())
            fn = int(((actual == 1) & (predicted == 0)).sum())
            fp = int(((actual == 0) & (predicted == 1)).sum())
            tn = int(((actual == 0) & (predicted == 0)).sum())
            actual_yes = int((actual == 1).sum())
            predicted_yes = int((predicted == 1).sum())
            n = len(part)
            rows.append({
                "feature": column,
                "group": str(group_value),
                "n": int(n),
                "actual_yes": actual_yes,
                "predicted_yes": predicted_yes,
                "tp": tp,
                "fn": fn,
                "fp": fp,
                "tn": tn,
                "recall": tp / actual_yes if actual_yes else np.nan,
                "precision": tp / predicted_yes if predicted_yes else np.nan,
                "small_group": bool(n < args.small_group_min),
            })

    out_path = OUT / "error_by_group.csv"
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"Test observations: {len(test)}")
    print(f"Test false negatives: {int(((y_true == 1) & (y_pred == 0)).sum())}")
    print(f"Test false positives: {int(((y_true == 0) & (y_pred == 1)).sum())}")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()

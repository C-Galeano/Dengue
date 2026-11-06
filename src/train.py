"""Phase 2: trains and compares models to predict dengue (confirmed vs. discarded) based on
age, sex, and reportable symptoms. Saves the best model to models/model.pkl.

Usage:
    python -m src.train
"""
import json
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import (StratifiedKFold, cross_val_predict, cross_validate,
                                     train_test_split)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from src.preprocessing import (AGE_GROUPS, BINARY, CATEGORICAL, FEATURES, MODEL_PATH,
                               SYMPTOM_LABELS, prepare_data)

SEED = 42
TARGET_RECALL = 0.80    # Minimum recall for the dengue class when selecting the threshold
AUC_TOLERANCE = 0.005  # If two models differ by less than this in AUC, the simpler one is preferred
METRICS = ["accuracy", "precision", "recall", "f1", "roc_auc"]
CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)


def build_preprocessor():
    """Imputes missing values (most frequent value) and encodes age/sex using one-hot encoding.

    It goes INSIDE the pipeline: this way, imputation is learned only from the training data of
    each fold (without leakage), and the web app reuses exactly the same transformation.
    """
    categorical = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer([
        ("cat", categorical, CATEGORICAL),
        ("bin", SimpleImputer(strategy="most_frequent"), BINARY),
    ])


def build_models():
    """The three candidates. class_weight='balanced' compensates for the 68/32 class imbalance
    by giving more weight to each error in the minority class, without adding or removing rows
    (unlike SMOTE or undersampling)."""
    candidates = {
        "Logistic Regression": LogisticRegression(class_weight="balanced", max_iter=1000),
        "Random Forest": RandomForestClassifier(
            n_estimators=300, min_samples_leaf=5, class_weight="balanced",
            n_jobs=-1, random_state=SEED),
        "Gradient Boosting": HistGradientBoostingClassifier(
            learning_rate=0.05, max_iter=200, class_weight="balanced", random_state=SEED),
    }
    return {name: Pipeline([("prep", build_preprocessor()), ("clf", clf)])
            for name, clf in candidates.items()}


def cross_validation(models, X, y):
    """5-fold stratified cross-validation (threshold 0.5). Returns the mean and standard deviation."""
    rows = []
    for name, model in models.items():
        res = cross_validate(model, X, y, cv=CV, scoring=METRICS, n_jobs=-1)
        row = {"model": name}
        for m in METRICS:
            row[m] = res[f"test_{m}"].mean()
            row[f"{m}_std"] = res[f"test_{m}"].std()
        rows.append(row)
    return pd.DataFrame(rows).set_index("model")


def choose_best(cv_results):
    """The model with the highest ROC-AUC; in a technical tie (< AUC_TOLERANCE), the first
    model in the list wins because it is the simplest and most interpretable."""
    best_auc = cv_results["roc_auc"].max()
    return cv_results.index[cv_results["roc_auc"] >= best_auc - AUC_TOLERANCE][0]


def choose_threshold(model, X, y, target_recall=TARGET_RECALL):
    """Selects the HIGHEST threshold whose dengue recall (on out-of-fold predictions) is
    >= target_recall.

    We prioritize recall because failing to detect dengue is more serious than a false alarm;
    however, we do not lower the threshold more than necessary, because with a threshold near 0,
    the model would classify everyone as having "dengue".
    """
    proba = cross_val_predict(model, X, y, cv=CV, method="predict_proba", n_jobs=-1)[:, 1]
    for threshold in np.arange(0.95, 0.04, -0.01):
        if recall_score(y, proba >= threshold) >= target_recall:
            return round(float(threshold), 2)
    return 0.05


def compute_metrics(y_true, proba, threshold):
    pred = (proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred).ravel()
    return {
        "accuracy": accuracy_score(y_true, pred),
        "precision": precision_score(y_true, pred),
        "recall": recall_score(y_true, pred),
        "specificity": tn / (tn + fp),
        "f1": f1_score(y_true, pred),
        "roc_auc": roc_auc_score(y_true, proba),
        "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
    }


def feature_importance(model, X, y):
    """Permutation importance: how much ROC-AUC decreases when a variable is shuffled.
    It works equally well for all three models and uses the original variables (not the one-hot encoded ones)."""
    r = permutation_importance(model, X, y, scoring="roc_auc", n_repeats=10,
                               random_state=SEED, n_jobs=-1)
    labels = {**SYMPTOM_LABELS, "age_group": "Age group", "sex": "Sex"}
    return (pd.DataFrame({"feature": X.columns, "label": [labels[c] for c in X.columns],
                          "importance": r.importances_mean, "std": r.importances_std})
            .sort_values("importance", ascending=False).reset_index(drop=True))


def temporal_validation(model, X, y, dates):
    """Additional robustness test: does the model work in a period it did not see during training?

    - 2024 -> 2023: train on the epidemic year and evaluate on the previous year.
    - 2023-2024 -> 2025-2026: train on past data and evaluate on the post-epidemic period.
    """
    year = dates.dt.year
    scenarios = {"Train 2024, evaluate 2023": (year == 2024, year == 2023),
                 "Train 2023-2024, evaluate 2025-2026": (year <= 2024, year >= 2025)}
    results = []
    for name, (train, test) in scenarios.items():
        m = sklearn.base.clone(model).fit(X[train], y[train])
        results.append({
            "scenario": name, "n_train": int(train.sum()), "n_test": int(test.sum()),
            "test_prevalence": round(float(y[test].mean()), 4),
            "roc_auc": round(float(roc_auc_score(y[test], m.predict_proba(X[test])[:, 1])), 4)})
    return results


def run_training(save=True):
    X, y, dates = prepare_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=SEED)
    print(f"Data: {len(X):,} cases ({y.mean():.1%} dengue) | train {len(X_train):,} | test {len(X_test):,}")

    models = build_models()
    cv_results = cross_validation(models, X_train, y_train)
    print("\nCross-validation (5 folds, threshold 0.5):")
    print(cv_results[METRICS].round(3).to_string())

    name = choose_best(cv_results)
    best = models[name]
    threshold = choose_threshold(best, X_train, y_train)
    print(f"\nBest model: {name} | selected threshold: {threshold}")

    best.fit(X_train, y_train)
    proba_test = best.predict_proba(X_test)[:, 1]
    test = compute_metrics(y_test, proba_test, threshold)
    test_05 = compute_metrics(y_test, proba_test, 0.5)
    print("\nTest (20% reserved):")
    for k in ["accuracy", "precision", "recall", "specificity", "f1", "roc_auc"]:
        print(f"  {k:<14} threshold {threshold}: {test[k]:.3f}   | threshold 0.5: {test_05[k]:.3f}")
    print(f"  Confusion matrix [[TN, FP], [FN, TP]]: {test['confusion_matrix']}")

    imp = feature_importance(best, X_test, y_test)
    print("\nImportance (ROC-AUC decrease after permutation):")
    print(imp[["label", "importance"]].round(4).to_string(index=False))

    temporal = temporal_validation(best, X, y, dates)
    print("\nTemporal validation:")
    for t in temporal:
        print(f"  {t['scenario']:<38} ROC-AUC = {t['roc_auc']:.3f}  (n_test={t['n_test']:,})")

    # Final model: retrained with ALL the data. The reported metrics are those from the test set,
    # which the model had not seen; retraining with more data can only improve the estimate.
    final = sklearn.base.clone(best).fit(X, y)

    artifact = {
        "model": final,
        "model_name": name,
        "features": FEATURES,
        "threshold": threshold,
        "age_groups": AGE_GROUPS,
        "symptom_labels": SYMPTOM_LABELS,
        "cv_metrics": cv_results.round(4).reset_index().to_dict(orient="records"),
        "test_metrics": {k: (round(v, 4) if isinstance(v, float) else v) for k, v in test.items()},
        "test_metrics_threshold_05": {k: (round(v, 4) if isinstance(v, float) else v) for k, v in test_05.items()},
        "importance": imp.round(4).to_dict(orient="records"),
        "temporal_validation": temporal,
        "data": {"n_total": int(len(X)), "dengue_prevalence": round(float(y.mean()), 4),
                 "n_train": int(len(X_train)), "n_test": int(len(X_test)),
                 "source": "SIVIGILA Tuluá - datos.gov.co (dq2k-gub9)"},
        "target_recall": TARGET_RECALL,
        "sklearn_version": sklearn.__version__,
        "trained_at": datetime.now().isoformat(timespec="seconds"),
    }
    if save:
        MODEL_PATH.parent.mkdir(exist_ok=True)
        joblib.dump(artifact, MODEL_PATH)
        summary = {k: v for k, v in artifact.items() if k != "model"}
        MODEL_PATH.with_name("metrics.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2, default=float), encoding="utf-8")
        print(f"\nSaved: {MODEL_PATH} and metrics.json")
    return artifact, {"models": models, "X_train": X_train, "X_test": X_test,
                      "y_train": y_train, "y_test": y_test, "dates": dates}


if __name__ == "__main__":
    run_training()

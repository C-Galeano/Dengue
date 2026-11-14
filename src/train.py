"""CRISP-ML(Q) Phases 3 (Model Engineering) and 4 (Quality Assurance): arbovirus differential diagnosis.

Trains and compares classifiers that predict, for a febrile patient, whether the illness is DENGUE,
CHIKUNGUNYA or ANOTHER FEBRILE ILLNESS, from age, sex, days since onset, symptoms, basic clinical and
laboratory findings and comorbidities. Saves the best model to models/model.pkl and the results to
models/metrics.json.

Explained step by step in notebooks/03_model_engineering.ipynb and notebooks/04_quality_assurance.ipynb.

Usage (needs the processed datasets; see src/sinan.py):
    python -m src.train
"""
import json
from datetime import datetime

import joblib
import lightgbm
import numpy as np
import pandas as pd
import sklearn
import xgboost
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score,
                             log_loss, precision_recall_fscore_support, recall_score, roc_auc_score)
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from src.explain import global_importance
from src.preprocessing import CLASS_LABELS, CLASSES, FEATURE_LABELS, FEATURES, MODEL_PATH
from src.sinan import TRAIN_YEAR, VALIDATION_YEAR, load_dataset

SEED = 42
DENGUE = CLASSES.index("dengue")
TARGET_DENGUE_RECALL = 0.80  # The dengue alert must catch at least 80 % of dengue cases
AUC_TOLERANCE = 0.005        # Differences smaller than this are a technical tie
METRICS = {"accuracy": "accuracy", "balanced_accuracy": "balanced_accuracy", "f1_macro": "f1_macro",
           "roc_auc": "roc_auc_ovr", "log_loss": "neg_log_loss"}
CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
TREE_MODELS = ["Random Forest", "XGBoost", "LightGBM"]


def build_models():
    """Logistic Regression is the linear BASELINE; the three tree ensembles are the candidates.

    Tree models need no scaling or encoding (all features are numeric or 0/1) and handle missing values
    natively, so they receive the features as they are. The classes are balanced by sampling (src/sinan.py),
    so no class weights are needed.
    """
    return {
        "Logistic Regression": Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(max_iter=2000)),
        ]),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, max_depth=14, min_samples_leaf=30, max_features="sqrt",
            n_jobs=-1, random_state=SEED),
        "XGBoost": XGBClassifier(
            n_estimators=400, learning_rate=0.05, max_depth=5, subsample=0.8, colsample_bytree=0.8,
            min_child_weight=5, objective="multi:softprob", eval_metric="mlogloss", tree_method="hist",
            n_jobs=-1, random_state=SEED),
        "LightGBM": LGBMClassifier(
            n_estimators=400, learning_rate=0.05, num_leaves=31, min_child_samples=40, subsample=0.8,
            subsample_freq=1, colsample_bytree=0.8, n_jobs=-1, random_state=SEED, verbose=-1),
    }


def cross_validation(models, X, y):
    """5-fold stratified cross-validation. Returns mean and standard deviation of each metric."""
    rows = []
    for name, model in models.items():
        res = cross_validate(model, X, y, cv=CV, scoring=METRICS, n_jobs=1)
        row = {"model": name}
        for m in METRICS:
            values = -res[f"test_{m}"] if m == "log_loss" else res[f"test_{m}"]
            row[m], row[f"{m}_std"] = values.mean(), values.std()
        rows.append(row)
        print(f"  {name:<20} ROC-AUC {row['roc_auc']:.3f}  balanced acc {row['balanced_accuracy']:.3f}")
    return pd.DataFrame(rows).set_index("model")


def choose_best(cv_results):
    """The tree model with the highest macro ROC-AUC (one-vs-rest). Within AUC_TOLERANCE, the lower
    log-loss wins (better calibrated probabilities)."""
    trees = cv_results.loc[TREE_MODELS]
    tied = trees[trees["roc_auc"] >= trees["roc_auc"].max() - AUC_TOLERANCE]
    return tied["log_loss"].idxmin()


def decide(proba, dengue_threshold):
    """Decision rule: DENGUE if P(dengue) >= threshold (safety alert); otherwise the most likely class.

    Missing dengue is the most dangerous error (it can progress to severe dengue), so dengue gets a
    lower bar than the plain "highest probability" rule.
    """
    pred = proba.argmax(axis=1)
    return np.where(proba[:, DENGUE] >= dengue_threshold, DENGUE, pred)


def choose_dengue_threshold(model, X, y, target_recall=TARGET_DENGUE_RECALL):
    """The HIGHEST threshold whose dengue recall, on out-of-fold predictions, is >= target_recall.
    Returns (threshold, out-of-fold probabilities)."""
    proba = cross_val_predict(model, X, y, cv=CV, method="predict_proba", n_jobs=1)
    for threshold in np.arange(0.90, 0.09, -0.01):
        if recall_score(y == DENGUE, proba[:, DENGUE] >= threshold) >= target_recall:
            return round(float(threshold), 2), proba
    return 0.10, proba


def compute_metrics(y_true, proba, dengue_threshold):
    """Multiclass metrics of the decision rule plus per-class detail."""
    y_true = np.asarray(y_true)
    pred = decide(proba, dengue_threshold)
    precision, recall, f1, support = precision_recall_fscore_support(y_true, pred, labels=range(len(CLASSES)),
                                                                     zero_division=0)
    return {
        "accuracy": accuracy_score(y_true, pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, pred),
        "f1_macro": f1_score(y_true, pred, average="macro"),
        "roc_auc": roc_auc_score(y_true, proba, multi_class="ovr", average="macro"),
        "log_loss": log_loss(y_true, proba, labels=range(len(CLASSES))),
        "per_class": [{"class": c, "label": CLASS_LABELS[c], "precision": float(precision[i]),
                       "recall": float(recall[i]), "f1": float(f1[i]), "support": int(support[i]),
                       "roc_auc": float(roc_auc_score(y_true == i, proba[:, i]))}
                      for i, c in enumerate(CLASSES)],
        "confusion_matrix": confusion_matrix(y_true, pred, labels=range(len(CLASSES))).tolist(),
    }


def temporal_validation(model, dengue_threshold):
    """Robustness: the model trained on 2023 is evaluated on 2025, a year it never saw."""
    X, y, _ = load_dataset(VALIDATION_YEAR)
    m = compute_metrics(y, model.predict_proba(X), dengue_threshold)
    return {"scenario": f"Train {TRAIN_YEAR}, evaluate {VALIDATION_YEAR}", "n_test": int(len(y)),
            **{k: m[k] for k in ("accuracy", "balanced_accuracy", "f1_macro", "roc_auc")},
            "per_class": m["per_class"]}


def external_validation(model, dengue_threshold):
    """Colombia: does P(dengue) separate confirmed from discarded dengue in Tuluá (SIVIGILA)?"""
    from src.tulua import SOURCE, load_tulua
    try:
        X, y = load_tulua()
    except FileNotFoundError:
        return None
    p = model.predict_proba(X)[:, DENGUE]
    alert = p >= dengue_threshold
    return {"source": SOURCE, "n": int(len(y)), "dengue_share": round(float(y.mean()), 4),
            "roc_auc_dengue": round(float(roc_auc_score(y, p)), 4),
            "dengue_recall": round(float(alert[y == 1].mean()), 4),
            "specificity": round(float((~alert[y == 0]).mean()), 4)}


def rounded(d):
    return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in d.items()}


def run_training(save=True):
    X, y, _ = load_dataset(TRAIN_YEAR)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)
    print(f"Data {TRAIN_YEAR}: {len(X):,} lab-confirmed cases | train {len(X_train):,} | test {len(X_test):,}")

    models = build_models()
    print("\nCross-validation (5 folds):")
    cv_results = cross_validation(models, X_train, y_train)

    name = choose_best(cv_results)
    best = models[name]
    threshold, _ = choose_dengue_threshold(best, X_train, y_train)
    print(f"\nBest model: {name} | dengue alert threshold: {threshold}")

    best.fit(X_train, y_train)
    proba_test = best.predict_proba(X_test)
    test = compute_metrics(y_test, proba_test, threshold)
    test_argmax = compute_metrics(y_test, proba_test, 1.01)  # plain "most likely class" rule
    print(f"\nTest: macro ROC-AUC {test['roc_auc']:.3f} | balanced accuracy {test['balanced_accuracy']:.3f}")
    for c in test["per_class"]:
        print(f"  {c['label']:<22} recall {c['recall']:.3f}  precision {c['precision']:.3f}  AUC {c['roc_auc']:.3f}")

    sample = X_test.sample(min(3000, len(X_test)), random_state=SEED)
    importance = global_importance(best, sample, FEATURE_LABELS, CLASSES)
    print("\nTop SHAP features:", ", ".join(r["label"] for r in importance[:6]))

    temporal = temporal_validation(best, threshold)
    external = external_validation(best, threshold)
    print(f"Temporal ({temporal['scenario']}): macro ROC-AUC {temporal['roc_auc']:.3f}")
    if external:
        print(f"External (Colombia, Tuluá): dengue ROC-AUC {external['roc_auc_dengue']:.3f}")

    # Final model: retrained with ALL the 2023 data. Reported metrics come from data it had not seen.
    final = sklearn.base.clone(best).fit(X, y)

    artifact = {
        "model": final,
        "model_name": name,
        "features": FEATURES,
        "feature_labels": FEATURE_LABELS,
        "classes": CLASSES,
        "class_labels": CLASS_LABELS,
        "dengue_threshold": threshold,
        "target_dengue_recall": TARGET_DENGUE_RECALL,
        "cv_metrics": cv_results.round(4).reset_index().to_dict(orient="records"),
        "test_metrics": rounded(test),
        "test_metrics_argmax": rounded(test_argmax),
        "shap_importance": importance,
        "temporal_validation": rounded(temporal),
        "external_validation": external,
        "data": {"n_total": int(len(X)), "n_train": int(len(X_train)), "n_test": int(len(X_test)),
                 "year": TRAIN_YEAR, "validation_year": VALIDATION_YEAR,
                 "class_counts": {c: int((y == i).sum()) for i, c in enumerate(CLASSES)},
                 "source": "SINAN (Brazil, DATASUS) - DENGBR and CHIKBR final files, lab-confirmed cases"},
        "versions": {"scikit-learn": sklearn.__version__, "xgboost": xgboost.__version__,
                     "lightgbm": lightgbm.__version__},
        "trained_at": datetime.now().isoformat(timespec="seconds"),
    }
    if save:
        MODEL_PATH.parent.mkdir(exist_ok=True)
        joblib.dump(artifact, MODEL_PATH, compress=3)
        summary = {k: v for k, v in artifact.items() if k != "model"}
        MODEL_PATH.with_name("metrics.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2, default=float), encoding="utf-8")
        print(f"\nSaved: {MODEL_PATH} and metrics.json")
    return artifact, {"models": models, "X_train": X_train, "X_test": X_test,
                      "y_train": y_train, "y_test": y_test}


if __name__ == "__main__":
    run_training()

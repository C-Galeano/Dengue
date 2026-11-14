"""CRISP-ML(Q) Phase 6 (Monitoring and Maintenance): detect when the deployed model needs retraining.

Three signals are tracked:
- Data drift: Population Stability Index (PSI) of every feature, training year vs. a newer year.
- Prior drift: the share of each lab-confirmed diagnosis in the surveillance data, year by year.
- Performance drift: macro ROC-AUC of the deployed model on the newer year (temporal validation).
Each signal is compared with a retraining rule. The report is saved to models/monitoring.json and shown on
/methodology/6.

    python -m src.monitoring
"""
import json

import numpy as np
import pandas as pd

from src.preprocessing import BINARY, CLASSES, FEATURE_LABELS, FEATURES, PROCESSED_DIR, ROOT_DIR
from src.sinan import TRAIN_YEAR, VALIDATION_YEAR, dataset_path

MONITORING_PATH = ROOT_DIR / "models" / "monitoring.json"
METRICS_PATH = ROOT_DIR / "models" / "metrics.json"

# Usual PSI reading: < 0.10 stable, 0.10-0.25 moderate change, > 0.25 significant change.
PSI_MODERATE, PSI_SIGNIFICANT = 0.10, 0.25
MIN_ROC_AUC = 0.80        # Same success criterion as Phase 1
MAX_PRIOR_CHANGE = 0.05   # 5 percentage points in the share of a diagnosis
NUMERIC_BINS = 10
EPSILON = 1e-4            # Avoids log(0) for empty bins


def psi(reference, current, numeric=False):
    """Population Stability Index between two samples of one feature.

    PSI = sum((current% - reference%) * ln(current% / reference%)) over bins. Numeric features use the deciles of the
    reference sample as bins; binary and categorical features use their own categories.
    """
    reference, current = pd.Series(reference).dropna(), pd.Series(current).dropna()
    if numeric:
        edges = np.unique(np.quantile(reference, np.linspace(0, 1, NUMERIC_BINS + 1)))
        edges[0], edges[-1] = -np.inf, np.inf
        ref_share = pd.cut(reference, edges).value_counts(normalize=True, sort=False)
        cur_share = pd.cut(current, edges).value_counts(normalize=True, sort=False)
    else:
        categories = sorted(set(reference) | set(current))
        ref_share = reference.value_counts(normalize=True).reindex(categories, fill_value=0)
        cur_share = current.value_counts(normalize=True).reindex(categories, fill_value=0)
    ref_share, cur_share = ref_share.to_numpy() + EPSILON, cur_share.to_numpy() + EPSILON
    return float(np.sum((cur_share - ref_share) * np.log(cur_share / ref_share)))


def psi_status(value):
    if value > PSI_SIGNIFICANT:
        return "significant"
    return "moderate" if value > PSI_MODERATE else "stable"


def data_drift(reference, current):
    """PSI of every model feature, sorted from the largest change."""
    rows = []
    for feature in FEATURES:
        numeric = feature in ("age", "days_since_onset")
        value = psi(reference[feature], current[feature], numeric=numeric)
        rows.append({"feature": feature, "label": FEATURE_LABELS[feature], "psi": round(value, 4),
                     "status": psi_status(value),
                     "reference": round(float(reference[feature].mean()), 3) if feature in BINARY else None,
                     "current": round(float(current[feature].mean()), 3) if feature in BINARY else None})
    return sorted(rows, key=lambda r: r["psi"], reverse=True)


def prior_drift(reference_year, current_year):
    """Share of each lab-confirmed diagnosis in the full surveillance files (not the balanced samples)."""
    shares = {}
    for year in (reference_year, current_year):
        counts = json.loads((PROCESSED_DIR / f"arbovirus_{year}_summary.json").read_text(encoding="utf-8"))["lab_confirmed"]
        total = sum(counts.values())
        shares[year] = {c: counts[c] / total for c in CLASSES}
    return [{"class": c, "reference": round(shares[reference_year][c], 4), "current": round(shares[current_year][c], 4),
             "change": round(shares[current_year][c] - shares[reference_year][c], 4)} for c in CLASSES]


def performance_drift(metrics):
    """Macro ROC-AUC on the training-year test set vs. the newer year (computed by src/train.py)."""
    return {"reference": metrics["test_metrics"]["roc_auc"], "current": metrics["temporal_validation"]["roc_auc"],
            "per_class": [{"class": r["class"], "label": r["label"], "roc_auc": round(r["roc_auc"], 4),
                           "recall": round(r["recall"], 4)} for r in metrics["temporal_validation"]["per_class"]]}


def retraining_decision(drift, priors, performance, top_features):
    """Retraining rules. Any rule that fires is a reason to retrain (besides the yearly schedule)."""
    drifted = [r["label"] for r in drift if r["feature"] in top_features and r["status"] == "significant"]
    prior_moves = [p["class"] for p in priors if abs(p["change"]) > MAX_PRIOR_CHANGE]
    rules = [
        {"rule": f"Macro ROC-AUC on new data below {MIN_ROC_AUC:.2f}",
         "value": f"{performance['current']:.2f}", "fired": performance["current"] < MIN_ROC_AUC},
        {"rule": f"Significant drift (PSI > {PSI_SIGNIFICANT}) in one of the top-5 SHAP features",
         "value": ", ".join(drifted) or "none", "fired": bool(drifted)},
        {"rule": f"Share of a diagnosis changes by more than {MAX_PRIOR_CHANGE * 100:.0f} percentage points",
         "value": ", ".join(prior_moves) or "none", "fired": bool(prior_moves)},
    ]
    return {"rules": rules, "retrain": any(r["fired"] for r in rules)}


def build_report(reference_year=TRAIN_YEAR, current_year=VALIDATION_YEAR, save=True):
    metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    reference = pd.read_parquet(dataset_path(reference_year))
    current = pd.read_parquet(dataset_path(current_year))
    top_features = [r["feature"] for r in metrics["shap_importance"][:5]]

    drift = data_drift(reference, current)
    priors = prior_drift(reference_year, current_year)
    performance = performance_drift(metrics)
    report = {"reference_year": reference_year, "current_year": current_year,
              "thresholds": {"psi_moderate": PSI_MODERATE, "psi_significant": PSI_SIGNIFICANT,
                             "min_roc_auc": MIN_ROC_AUC, "max_prior_change": MAX_PRIOR_CHANGE},
              "data_drift": drift, "prior_drift": priors, "performance": performance,
              "decision": retraining_decision(drift, priors, performance, top_features),
              "model_trained_at": metrics.get("trained_at")}
    if save:
        MONITORING_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def load_report():
    """The saved report, or None if `python -m src.monitoring` has not been run."""
    return json.loads(MONITORING_PATH.read_text(encoding="utf-8")) if MONITORING_PATH.exists() else None


if __name__ == "__main__":
    r = build_report()
    print(f"Monitoring {r['reference_year']} -> {r['current_year']}")
    for row in r["data_drift"][:8]:
        print(f"  PSI {row['psi']:.3f}  {row['status']:<11} {row['label']}")
    for p in r["prior_drift"]:
        print(f"  share {p['class']:<14} {p['reference']:.1%} -> {p['current']:.1%}")
    print(f"  macro ROC-AUC {r['performance']['reference']:.2f} -> {r['performance']['current']:.2f}")
    print("  RETRAIN" if r["decision"]["retrain"] else "  no retraining needed",
          [x["rule"] for x in r["decision"]["rules"] if x["fired"]])

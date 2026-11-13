"""SHAP explanations: how much each feature pushed the model toward each diagnosis.

SHAP (SHapley Additive exPlanations, Lundberg & Lee 2017) splits a prediction into one contribution per
feature. For tree models the exact values are computed with the TreeSHAP algorithm:
- XGBoost and LightGBM implement TreeSHAP natively (``pred_contribs`` / ``pred_contrib``), so the web app
  does not need the heavy ``shap`` package. Values are in log-odds (the model's raw score).
- Random Forest uses ``shap.TreeExplainer`` (values in probability units).

Positive value = the feature pushes TOWARD that class; negative = AWAY from it.
"""
import numpy as np


def shap_values(model, X):
    """SHAP values with shape (n_rows, n_features, n_classes)."""
    estimator = model.steps[-1][1] if hasattr(model, "steps") else model
    name = type(estimator).__name__
    if name == "XGBClassifier":
        import xgboost
        contrib = estimator.get_booster().predict(xgboost.DMatrix(X), pred_contribs=True)
        return np.transpose(contrib[:, :, :-1], (0, 2, 1))  # drop the bias column
    if name == "LGBMClassifier":
        contrib = estimator.booster_.predict(X, pred_contrib=True)
        n_classes = len(estimator.classes_)
        contrib = contrib.reshape(len(X), n_classes, X.shape[1] + 1)
        return np.transpose(contrib[:, :, :-1], (0, 2, 1))
    import shap  # only needed for models without native TreeSHAP
    values = shap.TreeExplainer(estimator).shap_values(X)
    return np.stack(values, axis=-1) if isinstance(values, list) else values


def global_importance(model, X, feature_labels, classes):
    """Mean |SHAP| per feature and class, sorted by the total. Returns a list of dicts."""
    mean_abs = np.abs(shap_values(model, X)).mean(axis=0)  # (features, classes)
    rows = [{"feature": f, "label": feature_labels[f],
             **{c: round(float(mean_abs[i, k]), 4) for k, c in enumerate(classes)},
             "total": round(float(mean_abs[i].sum()), 4)}
            for i, f in enumerate(X.columns)]
    return sorted(rows, key=lambda r: r["total"], reverse=True)


def explain_patient(model, X_row, class_index, feature_labels, top=6):
    """Top features that pushed ONE patient toward or away from the class `class_index`."""
    from src.preprocessing import feature_value_text
    values = shap_values(model, X_row)[0, :, class_index]
    order = np.argsort(-np.abs(values))[:top]
    return [{"feature": X_row.columns[i], "label": feature_labels[X_row.columns[i]],
             "value": float(X_row.iloc[0, i]),
             "value_text": feature_value_text(X_row.columns[i], X_row.iloc[0, i]),
             "shap": round(float(values[i]), 4)} for i in order]

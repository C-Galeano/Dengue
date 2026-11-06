"""Phase 2: trains and compares models to predict dengue (confirmed vs. discarded) based on
age, sex, and reportable symptoms. Saves the best model to models/modelo.pkl.

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

from src.preprocesamiento import (BINARIAS, CATEGORICAS, FEATURES, GRUPOS_EDAD, MODEL_PATH,
                              NOMBRES_SINTOMAS, preparar_datos)

SEED = 42
RECALL_OBJETIVO = 0.80   # Minimum recall for the dengue class when selecting the threshold
TOLERANCIA_AUC = 0.005   # If two models differ by less than this in AUC, the simpler one is preferred
METRICAS = ["accuracy", "precision", "recall", "f1", "roc_auc"]
CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)


def preprocesador():
    """Imputes missing values (most frequent value) and encodes age/sex using one-hot encoding.

    It goes INSIDE the pipeline: this way, imputation is learned only from the training data of
    each fold (without leakage), and the web app reuses exactly the same transformation.
    """
    categoricas = Pipeline([
        ("imputar", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer([
        ("cat", categoricas, CATEGORICAS),
        ("bin", SimpleImputer(strategy="most_frequent"), BINARIAS),
    ])


def construir_modelos():
    """The three candidates. class_weight='balanced' compensates for the 68/32 class imbalance
    by giving more weight to each error in the minority class, without adding or removing rows
    (unlike SMOTE or undersampling)."""
    candidatos = {
        "Logistic Regression": LogisticRegression(class_weight="balanced", max_iter=1000),
        "Random Forest": RandomForestClassifier(
            n_estimators=300, min_samples_leaf=5, class_weight="balanced",
            n_jobs=-1, random_state=SEED),
        "Gradient Boosting": HistGradientBoostingClassifier(
            learning_rate=0.05, max_iter=200, class_weight="balanced", random_state=SEED),
    }
    return {nombre: Pipeline([("prep", preprocesador()), ("clf", clf)])
            for nombre, clf in candidatos.items()}


def validacion_cruzada(modelos, X, y):
    """5-fold stratified cross-validation (threshold 0.5). Returns the mean and standard deviation."""
    filas = []
    for nombre, modelo in modelos.items():
        res = cross_validate(modelo, X, y, cv=CV, scoring=METRICAS, n_jobs=-1)
        fila = {"modelo": nombre}
        for m in METRICAS:
            fila[m] = res[f"test_{m}"].mean()
            fila[f"{m}_std"] = res[f"test_{m}"].std()
        filas.append(fila)
    return pd.DataFrame(filas).set_index("modelo")


def elegir_mejor(resultados_cv):
    """The model with the highest ROC-AUC; in a technical tie (< TOLERANCIA_AUC), the first
    model in the list wins because it is the simplest and most interpretable."""
    mejor_auc = resultados_cv["roc_auc"].max()
    return resultados_cv.index[resultados_cv["roc_auc"] >= mejor_auc - TOLERANCIA_AUC][0]


def elegir_umbral(modelo, X, y, recall_objetivo=RECALL_OBJETIVO):
    """Selects the HIGHEST threshold whose dengue recall (on out-of-fold predictions) is
    >= recall_objetivo.

    We prioritize recall because failing to detect dengue is more serious than a false alarm;
    however, we do not lower the threshold more than necessary, because with a threshold near 0,
    the model would classify everyone as having "dengue".
    """
    proba = cross_val_predict(modelo, X, y, cv=CV, method="predict_proba", n_jobs=-1)[:, 1]
    for umbral in np.arange(0.95, 0.04, -0.01):
        if recall_score(y, proba >= umbral) >= recall_objetivo:
            return round(float(umbral), 2)
    return 0.05


def metricas(y_true, proba, umbral):
    pred = (proba >= umbral).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred).ravel()
    return {
        "accuracy": accuracy_score(y_true, pred),
        "precision": precision_score(y_true, pred),
        "recall": recall_score(y_true, pred),
        "especificidad": tn / (tn + fp),
        "f1": f1_score(y_true, pred),
        "roc_auc": roc_auc_score(y_true, proba),
        "matriz_confusion": [[int(tn), int(fp)], [int(fn), int(tp)]],
    }


def importancia(modelo, X, y):
    """Permutation importance: how much ROC-AUC decreases when a variable is shuffled.
    It works equally well for all three models and uses the original variables (not the one-hot encoded ones)."""
    r = permutation_importance(modelo, X, y, scoring="roc_auc", n_repeats=10,
                               random_state=SEED, n_jobs=-1)
    nombres = {**NOMBRES_SINTOMAS, "grupo_edad": "Age group", "sexo": "Sex"}
    return (pd.DataFrame({"variable": X.columns, "nombre": [nombres[c] for c in X.columns],
                          "importancia": r.importances_mean, "std": r.importances_std})
            .sort_values("importancia", ascending=False).reset_index(drop=True))


def validacion_temporal(modelo, X, y, fechas):
    """Additional robustness test: does the model work in a period it did not see during training?

    - 2024 -> 2023: train on the epidemic year and evaluate on the previous year.
    - 2023-2024 -> 2025-2026: train on past data and evaluate on the post-epidemic period.
    """
    anio = fechas.dt.year
    escenarios = {"Train 2024, evaluate 2023": (anio == 2024, anio == 2023),
                  "Train 2023-2024, evaluate 2025-2026": (anio <= 2024, anio >= 2025)}
    resultados = []
    for nombre, (train, test) in escenarios.items():
        m = sklearn.base.clone(modelo).fit(X[train], y[train])
        resultados.append({
            "escenario": nombre, "n_train": int(train.sum()), "n_test": int(test.sum()),
            "prevalencia_test": round(float(y[test].mean()), 4),
            "roc_auc": round(float(roc_auc_score(y[test], m.predict_proba(X[test])[:, 1])), 4)})
    return resultados


def entrenar(guardar=True):
    X, y, fechas = preparar_datos()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=SEED)
    print(f"Data: {len(X):,} cases ({y.mean():.1%} dengue) | train {len(X_train):,} | test {len(X_test):,}")

    modelos = construir_modelos()
    resultados_cv = validacion_cruzada(modelos, X_train, y_train)
    print("\nCross-validation (5 folds, threshold 0.5):")
    print(resultados_cv[METRICAS].round(3).to_string())

    nombre = elegir_mejor(resultados_cv)
    mejor = modelos[nombre]
    umbral = elegir_umbral(mejor, X_train, y_train)
    print(f"\nBest model: {nombre} | selected threshold: {umbral}")

    mejor.fit(X_train, y_train)
    proba_test = mejor.predict_proba(X_test)[:, 1]
    test = metricas(y_test, proba_test, umbral)
    test_05 = metricas(y_test, proba_test, 0.5)
    print("\nTest (20% reserved):")
    for k in ["accuracy", "precision", "recall", "especificidad", "f1", "roc_auc"]:
        print(f"  {k:<14} threshold {umbral}: {test[k]:.3f}   | threshold 0.5: {test_05[k]:.3f}")
    print(f"  Confusion matrix [[TN, FP], [FN, TP]]: {test['matriz_confusion']}")

    imp = importancia(mejor, X_test, y_test)
    print("\nImportance (ROC-AUC decrease after permutation):")
    print(imp[["nombre", "importancia"]].round(4).to_string(index=False))

    temporal = validacion_temporal(mejor, X, y, fechas)
    print("\nTemporal validation:")
    for t in temporal:
        print(f"  {t['escenario']:<38} ROC-AUC = {t['roc_auc']:.3f}  (n_test={t['n_test']:,})")

    # Final model: retrained with ALL the data. The reported metrics are those from the test set,
    # which the model had not seen; retraining with more data can only improve the estimate.
    final = sklearn.base.clone(mejor).fit(X, y)

    artefacto = {
        "modelo": final,
        "nombre_modelo": nombre,
        "features": FEATURES,
        "umbral": umbral,
        "grupos_edad": GRUPOS_EDAD,
        "nombres_sintomas": NOMBRES_SINTOMAS,
        "metricas_cv": resultados_cv.round(4).reset_index().to_dict(orient="records"),
        "metricas_test": {k: (round(v, 4) if isinstance(v, float) else v) for k, v in test.items()},
        "metricas_test_umbral_05": {k: (round(v, 4) if isinstance(v, float) else v) for k, v in test_05.items()},
        "importancia": imp.round(4).to_dict(orient="records"),
        "validacion_temporal": temporal,
        "datos": {"n_total": int(len(X)), "prevalencia_dengue": round(float(y.mean()), 4),
                  "n_train": int(len(X_train)), "n_test": int(len(X_test)),
                  "fuente": "SIVIGILA Tuluá - datos.gov.co (dq2k-gub9)"},
        "recall_objetivo": RECALL_OBJETIVO,
        "sklearn_version": sklearn.__version__,
        "fecha_entrenamiento": datetime.now().isoformat(timespec="seconds"),
    }
    if guardar:
        MODEL_PATH.parent.mkdir(exist_ok=True)
        joblib.dump(artefacto, MODEL_PATH)
        resumen = {k: v for k, v in artefacto.items() if k != "modelo"}
        MODEL_PATH.with_name("metricas.json").write_text(
            json.dumps(resumen, ensure_ascii=False, indent=2, default=float), encoding="utf-8")
        print(f"\nSaved: {MODEL_PATH} and metricas.json")
    return artefacto, {"modelos": modelos, "X_train": X_train, "X_test": X_test,
                       "y_train": y_train, "y_test": y_test, "fechas": fechas}


if __name__ == "__main__":
    entrenar()
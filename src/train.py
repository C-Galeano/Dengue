"""Fase 2: entrena y compara modelos para predecir dengue (confirmado vs. descartado) a partir de
edad, sexo y síntomas reportables. Guarda el mejor en models/modelo.pkl.

Uso:
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
RECALL_OBJETIVO = 0.80   # Recall mínimo de la clase dengue al elegir el umbral
TOLERANCIA_AUC = 0.005   # Si dos modelos difieren menos que esto en AUC, se prefiere el más simple
METRICAS = ["accuracy", "precision", "recall", "f1", "roc_auc"]
CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)


def preprocesador():
    """Imputa nulos (valor más frecuente) y codifica edad/sexo con one-hot.

    Va DENTRO del pipeline: así la imputación se aprende solo con los datos de entrenamiento de
    cada pliegue (sin fuga) y la app web reutiliza exactamente la misma transformación.
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
    """Los tres candidatos. class_weight='balanced' compensa el desbalance 68/32 dando más peso
    a cada error en la clase minoritaria, sin inventar ni borrar filas (a diferencia de SMOTE o
    del submuestreo)."""
    candidatos = {
        "Regresión Logística": LogisticRegression(class_weight="balanced", max_iter=1000),
        "Random Forest": RandomForestClassifier(
            n_estimators=300, min_samples_leaf=5, class_weight="balanced",
            n_jobs=-1, random_state=SEED),
        "Gradient Boosting": HistGradientBoostingClassifier(
            learning_rate=0.05, max_iter=200, class_weight="balanced", random_state=SEED),
    }
    return {nombre: Pipeline([("prep", preprocesador()), ("clf", clf)])
            for nombre, clf in candidatos.items()}


def validacion_cruzada(modelos, X, y):
    """Validación cruzada estratificada de 5 pliegues (umbral 0.5). Devuelve media y desviación."""
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
    """El de mayor ROC-AUC; en empate técnico (< TOLERANCIA_AUC) gana el primero de la lista,
    que es el más simple e interpretable."""
    mejor_auc = resultados_cv["roc_auc"].max()
    return resultados_cv.index[resultados_cv["roc_auc"] >= mejor_auc - TOLERANCIA_AUC][0]


def elegir_umbral(modelo, X, y, recall_objetivo=RECALL_OBJETIVO):
    """Elige el umbral MÁS ALTO cuyo recall de dengue (en predicciones fuera de pliegue) sea
    >= recall_objetivo.

    Priorizamos recall porque no detectar un dengue es más grave que una falsa alarma; pero no
    bajamos el umbral más de lo necesario, porque con umbral ~0 el modelo diría "dengue" a todos.
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
    """Importancia por permutación: cuánto cae el ROC-AUC si se desordena una variable.
    Sirve igual para los tres modelos y trabaja sobre las variables originales (no las one-hot)."""
    r = permutation_importance(modelo, X, y, scoring="roc_auc", n_repeats=10,
                               random_state=SEED, n_jobs=-1)
    nombres = {**NOMBRES_SINTOMAS, "grupo_edad": "Grupo de edad", "sexo": "Sexo"}
    return (pd.DataFrame({"variable": X.columns, "nombre": [nombres[c] for c in X.columns],
                          "importancia": r.importances_mean, "std": r.importances_std})
            .sort_values("importancia", ascending=False).reset_index(drop=True))


def validacion_temporal(modelo, X, y, fechas):
    """Prueba extra de robustez: ¿el modelo sirve en un período que no vio al entrenar?

    - 2024 -> 2023: entrenar con el año epidémico y evaluar en el año anterior.
    - 2023-2024 -> 2025-2026: entrenar con el pasado y evaluar en el período post-epidemia.
    """
    anio = fechas.dt.year
    escenarios = {"Entrena 2024, evalúa 2023": (anio == 2024, anio == 2023),
                  "Entrena 2023-2024, evalúa 2025-2026": (anio <= 2024, anio >= 2025)}
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
    print(f"Datos: {len(X):,} casos ({y.mean():.1%} dengue) | train {len(X_train):,} | test {len(X_test):,}")

    modelos = construir_modelos()
    resultados_cv = validacion_cruzada(modelos, X_train, y_train)
    print("\nValidación cruzada (5 pliegues, umbral 0.5):")
    print(resultados_cv[METRICAS].round(3).to_string())

    nombre = elegir_mejor(resultados_cv)
    mejor = modelos[nombre]
    umbral = elegir_umbral(mejor, X_train, y_train)
    print(f"\nMejor modelo: {nombre} | umbral elegido: {umbral}")

    mejor.fit(X_train, y_train)
    proba_test = mejor.predict_proba(X_test)[:, 1]
    test = metricas(y_test, proba_test, umbral)
    test_05 = metricas(y_test, proba_test, 0.5)
    print("\nTest (20% reservado):")
    for k in ["accuracy", "precision", "recall", "especificidad", "f1", "roc_auc"]:
        print(f"  {k:<14} umbral {umbral}: {test[k]:.3f}   | umbral 0.5: {test_05[k]:.3f}")
    print(f"  Matriz de confusión [[VN, FP], [FN, VP]]: {test['matriz_confusion']}")

    imp = importancia(mejor, X_test, y_test)
    print("\nImportancia (caída de ROC-AUC al permutar):")
    print(imp[["nombre", "importancia"]].round(4).to_string(index=False))

    temporal = validacion_temporal(mejor, X, y, fechas)
    print("\nValidación temporal:")
    for t in temporal:
        print(f"  {t['escenario']:<38} ROC-AUC = {t['roc_auc']:.3f}  (n_test={t['n_test']:,})")

    # Modelo final: se reentrena con TODOS los datos. Las métricas reportadas son las del test,
    # que el modelo no había visto; reentrenar con más datos solo puede mejorar la estimación.
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
        print(f"\nGuardado: {MODEL_PATH} y metricas.json")
    return artefacto, {"modelos": modelos, "X_train": X_train, "X_test": X_test,
                       "y_train": y_train, "y_test": y_test, "fechas": fechas}


if __name__ == "__main__":
    entrenar()

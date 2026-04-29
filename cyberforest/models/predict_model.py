
"""
predict_model.py — Evaluación de modelos supervisado.
Tarea: clasificacion
"""
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    roc_auc_score, confusion_matrix, classification_report,
    ConfusionMatrixDisplay,
)



import mlflow
import mlflow.sklearn


from cyberforest.utils.paths import FIGURES_DIR, MODELS_DIR, REPORTS_DIR


# Umbral de decisión. Bajar (e.g. 0.3) aumenta recall de clase minoritaria.
DECISION_THRESHOLD: float = 0.5



def evaluate_models(
    models: dict,
    X_train,
    y_train,
    X_test,
    y_test,

    threshold: float = DECISION_THRESHOLD,

) -> pd.DataFrame:
    """
    Evalúa todos los modelos sobre train y test.


    Métricas: Accuracy, F1 weighted, Precision, Recall, ROC-AUC (binario).
    Genera matrices de confusión en figures/.


    Returns
    -------
    pd.DataFrame ordenado por métrica principal.
    """
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}\n  Evaluación — clasificacion (umbral={threshold})\n{'='*60}")



    mlflow.set_experiment("cyberforest")


    results = []
    for name, model in models.items():
        print(f"\n--- {name} ---")


        if threshold != 0.5 and hasattr(model, "predict_proba"):
            proba_test   = model.predict_proba(X_test)[:, 1]
            y_pred_test  = (proba_test >= threshold).astype(int)
            proba_train  = model.predict_proba(X_train)[:, 1]
            y_pred_train = (proba_train >= threshold).astype(int)
        else:
            y_pred_test  = model.predict(X_test)
            y_pred_train = model.predict(X_train)

        acc_train = accuracy_score(y_train, y_pred_train)
        acc_test  = accuracy_score(y_test,  y_pred_test)
        f1_train  = f1_score(y_train, y_pred_train, average="weighted", zero_division=0)
        f1_test   = f1_score(y_test,  y_pred_test,  average="weighted", zero_division=0)
        prec_test = precision_score(y_test, y_pred_test, average="weighted", zero_division=0)
        rec_test  = recall_score(y_test,  y_pred_test,  average="weighted", zero_division=0)
        roc_auc   = None
        if hasattr(model, "predict_proba") and len(np.unique(y_test)) == 2:
            roc_auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])

        print(f"  Accuracy  → train: {acc_train:.3f} | test: {acc_test:.3f}")
        print(f"  F1 (w)    → train: {f1_train:.3f}  | test: {f1_test:.3f}")
        print(f"  Precision → {prec_test:.3f}  | Recall → {rec_test:.3f}")
        if roc_auc is not None:
            print(f"  ROC-AUC   → {roc_auc:.3f}")
        print()
        print(classification_report(y_test, y_pred_test, zero_division=0))
        _plot_confusion_matrix(y_test, y_pred_test, name)

        row = {
            "Modelo":    name,
            "Acc_train": round(acc_train, 4), "Acc_test":  round(acc_test,  4),
            "F1_train":  round(f1_train,  4), "F1_test":   round(f1_test,   4),
            "Prec_test": round(prec_test, 4), "Rec_test":  round(rec_test,  4),
        }
        if roc_auc is not None:
            row["ROC_AUC"] = round(roc_auc, 4)


        with mlflow.start_run(run_name=f"{name}_eval"):
            mlflow.log_metrics({
                "acc_train": acc_train, "acc_test": acc_test,
                "f1_train":  f1_train,  "f1_test":  f1_test,
                "prec_test": prec_test, "rec_test":  rec_test,
            })
            if roc_auc is not None:
                mlflow.log_metric("roc_auc", roc_auc)
            mlflow.log_artifact(str(FIGURES_DIR / f"cm_{name}.png"))



        results.append(row)


    df_results = pd.DataFrame(results).sort_values("Acc_test", ascending=False)

    out_csv = REPORTS_DIR / "resultados_modelos.csv"
    df_results.to_csv(out_csv, index=False)
    print(f"\n{'='*60}\n  Resumen:\n{'='*60}")
    print(df_results.to_string(index=False))
    print(f"\n  Guardado → {out_csv}")
    return df_results



def _plot_confusion_matrix(y_true, y_pred, model_name: str) -> None:
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
    cm   = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    fig, ax = plt.subplots(figsize=(7, 6))
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(f"Matriz de confusion — {model_name}", fontsize=13)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / f"cm_{model_name}.png", dpi=150)
    plt.close(fig)
    print(f"    cm_{model_name}.png guardado")




def predict_new(model_name: str, X_new) -> np.ndarray:
    """Carga un modelo y predice sobre nuevas muestras (ya preprocesadas)."""
    path = MODELS_DIR / f"{model_name}.joblib"
    if not path.exists():
        raise FileNotFoundError(f"Modelo no encontrado: {path}")
    return joblib.load(path).predict(X_new)



def predict_proba_new(model_name: str, X_new) -> np.ndarray:
    """Carga un modelo y devuelve probabilidades de clase."""
    path = MODELS_DIR / f"{model_name}.joblib"
    if not path.exists():
        raise FileNotFoundError(f"Modelo no encontrado: {path}")
    model = joblib.load(path)
    if not hasattr(model, "predict_proba"):
        raise ValueError(f"{model_name} no soporta predict_proba")
    return model.predict_proba(X_new)


# ---------------------------------------------------------------------------
# Búsqueda automática de umbral óptimo por F1
# ---------------------------------------------------------------------------
# ACTIVAR solo en clasificación BINARIA (dos clases).
# Si tu problema es multiclase, mantén esta función comentada y usa
# DECISION_THRESHOLD = 0.5 (o ajústalo manualmente).
#
# Uso típico en train_model.py, tras entrenar el mejor modelo:
#
#   from sklearn.metrics import precision_recall_curve
#   import numpy as np
#   from cyberforest.models.predict_model import find_best_threshold
#   from cyberforest.utils.paths import ARTIFACTS_DIR
#   import joblib
#
#   proba_val = best_model.predict_proba(X_val)[:, 1]
#   threshold, f1 = find_best_threshold(y_val, proba_val)
#   print(f"Umbral óptimo: {threshold:.4f}  |  F1: {f1:.4f}")
#   joblib.dump(threshold, ARTIFACTS_DIR / "threshold.joblib")
#
# En main.py, carga el umbral antes de evaluar:
#
#   threshold = joblib.load(ARTIFACTS_DIR / "threshold.joblib")
#   evaluate_models(models, X_train, y_train, X_test, y_test, threshold=threshold)
#
# def find_best_threshold(y_true, y_proba):
#     """
#     Calcula el umbral de decisión que maximiza el F1-score binario.
#
#     Parameters
#     ----------
#     y_true  : array-like con etiquetas reales (0/1)
#     y_proba : array-like con probabilidades de clase positiva
#
#     Returns
#     -------
#     best_threshold : float — umbral que maximiza F1
#     best_f1        : float — F1 en ese umbral
#     """
#     from sklearn.metrics import precision_recall_curve
#     precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
#     # f1_scores tiene longitud N; thresholds tiene longitud N-1 → recortamos
#     f1_scores = (2 * precision * recall) / (precision + recall + 1e-9)
#     best_idx       = np.nanargmax(f1_scores[:-1])
#     best_threshold = thresholds[best_idx]
#     best_f1        = f1_scores[best_idx]
#     return best_threshold, best_f1


# ---------------------------------------------------------------------------
# Modo prueba: carga artefactos y evalúa sobre datos nuevos introducidos
# por el usuario en tiempo de ejecución.
# ---------------------------------------------------------------------------
def test_model() -> None:
    """
    Modo interactivo de prueba del modelo entrenado.

    Flujo:
      1. Lista los modelos disponibles en models/ y pide elegir uno.
      2. Carga el modelo y los artefactos de preprocesado (scaler, PCA,
         encoders, threshold) guardados durante el entrenamiento.
      3. Pide al usuario los valores de cada feature por consola.
      4. Preprocesa la entrada con process_input() de build_features.
      5. Imprime la predicción (y probabilidad si está disponible).

    Requisitos previos:
      - Haber ejecutado run_full_pipeline() al menos una vez para que
        existan los joblibs en artifacts/ y los modelos en models/.
    """
    from cyberforest.features.build_features import process_input
    from cyberforest.utils.paths import ARTIFACTS_DIR, PROCESSED_DATA_DIR
    import pandas as pd

    # ── 1. Elegir modelo ────────────────────────────────────────────────
    available = sorted(MODELS_DIR.glob("*.joblib"))
    if not available:
        print("No hay modelos entrenados en models/. Ejecuta primero la opción 0.")
        return

    print("\nModelos disponibles:")
    for i, p in enumerate(available):
        print(f"  [{i}] {p.stem}")
    try:
        idx = int(input("Elige modelo (número): "))
        model = joblib.load(available[idx])
        model_name = available[idx].stem
    except (ValueError, IndexError):
        print("Selección inválida.")
        return

    # ── 2. Cargar nombres de features ──────────────────────────────────
    feat_path = ARTIFACTS_DIR / "feature_names.joblib"
    if feat_path.exists():
        feature_names = joblib.load(feat_path)
    else:
        x_train_path = PROCESSED_DATA_DIR / "X_train.csv"
        if x_train_path.exists():
            feature_names = pd.read_csv(x_train_path).columns.tolist()
        else:
            print("No se encontró feature_names.joblib ni X_train.csv. Ejecuta primero run_full_pipeline().")
            return

    # ── 3. Pedir valores al usuario ────────────────────────────────────
    print(f"\nIntroduce los valores para el modelo '{model_name}':")
    print("  (deja en blanco para usar 0 como valor por defecto)\n")
    row = {}
    for feat in feature_names:
        raw = input(f"  {feat}: ").strip()
        try:
            row[feat] = float(raw) if raw else 0.0
        except ValueError:
            row[feat] = raw if raw else 0.0

    df_input = pd.DataFrame([row])

    # ── 4. Preprocesar ─────────────────────────────────────────────────
    try:
        X_new = process_input(df_input)
    except Exception as e:
        print(f"\nError en preprocesado: {e}")
        return

    # ── 5. Cargar umbral (si existe) ───────────────────────────────────
    threshold_path = ARTIFACTS_DIR / "threshold.joblib"
    threshold = joblib.load(threshold_path) if threshold_path.exists() else 0.5

    # ── 6. Predecir ────────────────────────────────────────────────────
    print(f"\n{'='*50}")
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X_new)[0]
        pred  = int(proba[1] >= threshold)
        print(f"  Modelo     : {model_name}")
        print(f"  Umbral     : {threshold:.4f}")
        print(f"  Predicción : {pred}")
        print(f"  Probabilidades: {dict(enumerate(proba.round(4).tolist()))}")
    else:
        pred = model.predict(X_new)[0]
        print(f"  Modelo     : {model_name}")
        print(f"  Predicción : {pred}")
    print(f"{'='*50}\n")




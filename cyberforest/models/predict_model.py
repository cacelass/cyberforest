
"""
predict_model.py — Evaluación de modelos supervisado.
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

from cyberforest.utils.paths import FIGURES_DIR, MODELS_DIR

# Umbral de decisión para clasificación binaria.
# Bajar (e.g., 0.3) aumenta recall de clase minoritaria a costa de más FP.
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
    Guarda matrices de confusión en figures/ y resultados en CSV.

    Returns
    -------
    pd.DataFrame ordenado de mayor a menor Acc_test.
    """
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    print(f"\n{'='*60}\n  Evaluación de modelos (umbral={threshold})\n{'='*60}")

    results = []
    for name, model in models.items():
        print(f"\n--- {name} ---")

        if threshold != 0.5 and hasattr(model, "predict_proba"):
            proba_test  = model.predict_proba(X_test)[:, 1]
            y_pred_test = (proba_test >= threshold).astype(int)
            proba_train = model.predict_proba(X_train)[:, 1]
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

        roc_auc = None
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
        results.append(row)

    df_results = pd.DataFrame(results).sort_values("Acc_test", ascending=False)
    df_results.to_csv(FIGURES_DIR / "resultados_modelos.csv", index=False)

    print(f"\n{'='*60}\n  Resumen (ordenado por Acc_test):\n{'='*60}")
    print(df_results.to_string(index=False))
    return df_results


def _plot_confusion_matrix(y_true, y_pred, model_name: str) -> None:
    cm   = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    fig, ax = plt.subplots(figsize=(7, 6))
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(f"Matriz de confusión — {model_name}", fontsize=13)
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



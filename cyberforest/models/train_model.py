
import numpy as np
import joblib

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score

from cyberforest.utils.paths import MODELS_DIR


# ---------------------------------------------------------------------------
# Configuración de modelos
# ---------------------------------------------------------------------------

def _build_models() -> dict:
    """
    Define los modelos a entrenar.

    KNN            → lazy learner, sin suposiciones sobre los datos.
                     Requiere features escaladas. Sensible a k y a dimensiones altas.

    LogisticReg    → modelo base en clasificación binaria. Rápido, interpretable
                     y genera probabilidades calibradas.

    DecisionTree   → caja blanca, fácil de interpretar. Propenso a overfitting
                     → regularizar con max_depth, min_samples_leaf.

    RandomForest   → ensemble de árboles. Robusto y buen rendimiento por defecto.
                     Permite calcular importancia de variables (feature_importances_).

    GradBoost      → mayor precisión que RF en muchos casos, pero más lento
                     y más sensible a hiperparámetros.

    SVM (RBF)      → potente en espacios de alta dimensión. Lento en datasets grandes.
                     El pipeline incluye StandardScaler propio.

    Pipelines con PCA integrado:
      PCA_LogReg   → PCA(95%) → LogReg. Útil con muchas features correladas.
      PCA_SVM      → PCA(95%) → SVM RBF. Acelera SVM enormemente en alta dim.

    ¿Cuándo incluir los pipelines PCA?
      - Muchas features muy correladas entre sí (|r| > 0.8)
      - Alta dimensionalidad (>50 features)
      - SVM o KNN muy lentos sin reducción previa
      - Quieres comparar directamente si PCA mejora o no el rendimiento
    """
    return {
        # --- KNN: el valor de n_neighbors se optimiza automáticamente en train_models() ---
        "KNN": KNeighborsClassifier(n_neighbors=7, weights="distance"),

        # --- Regresión Logística: modelo base rápido e interpretable ---
        "LogisticRegression": LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=42,
        ),

        # --- Árbol de Decisión: ajustar max_depth para evitar overfitting ---
        "DecisionTree": DecisionTreeClassifier(
            max_depth=7,
            min_samples_leaf=5,
            class_weight="balanced",
            random_state=42,
        ),

        # --- Random Forest: robusto, con importancia de variables ---
        "RandomForest": RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            max_features="sqrt",   # sqrt(n_features) por árbol
            max_samples=0.8,       # bootstrap sample del 80%
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        ),

        # --- Gradient Boosting: alta precisión, mayor coste computacional ---
        # "GradientBoosting": GradientBoostingClassifier(
        #     n_estimators=200, max_depth=5, learning_rate=0.05, random_state=42
        # ),

        # --- SVM RBF: muy eficaz en dimensiones altas ---
        # El pipeline escala internamente; no necesita X ya escalado.
        # "SVM": Pipeline([
        #     ("scaler", StandardScaler()),
        #     ("clf", SVC(
        #         kernel="rbf", C=1.0, gamma="scale",
        #         class_weight="balanced", probability=True, random_state=42,
        #     )),
        # ]),

        # --- Pipeline PCA + Regresión Logística ---
        # "PCA_LogReg": Pipeline([
        #     ("pca", PCA(n_components=0.95, random_state=42)),
        #     ("clf", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)),
        # ]),

        # --- Pipeline PCA + SVM RBF ---
        # SVM es O(n²~n³); PCA lo acelera mucho. probability=True para ROC-AUC.
        # "PCA_SVM": Pipeline([
        #     ("pca", PCA(n_components=0.95, random_state=42)),
        #     ("clf", SVC(kernel="rbf", C=1.0, gamma="scale",
        #                 class_weight="balanced", probability=True, random_state=42)),
        # ]),
    }


def _find_best_k(X_train, y_train, k_range=range(1, 21)) -> int:
    """
    Busca el mejor k para KNN por validación cruzada (5-fold, métrica F1_weighted).
    Devuelve el k con mayor F1 medio, priorizando k más alto en empates.
    """
    print("    Buscando mejor k para KNN...")
    best_k, best_score = 1, 0.0
    for k in k_range:
        knn = KNeighborsClassifier(n_neighbors=k, weights="distance")
        score = cross_val_score(knn, X_train, y_train, cv=5, scoring="f1_weighted").mean()
        if score >= best_score:   # >= → preferimos k más alto en empates
            best_k, best_score = k, score
    print(f"    Mejor k = {best_k}  (F1_weighted CV = {best_score:.3f})")
    return best_k


def train_models(
    X_train,
    y_train,
    tune_knn: bool = True,
    cv_evaluate: bool = True,
) -> dict:
    """
    Entrena todos los modelos definidos en _build_models() y los guarda en models/.

    Parameters
    ----------
    X_train      : features de entrenamiento (array-like)
    y_train      : etiquetas de entrenamiento (array-like)
    tune_knn     : si True, optimiza k de KNN por cross-validation antes de entrenar.
    cv_evaluate  : si True, muestra F1_weighted (5-fold CV) de cada modelo.

    Returns
    -------
    dict : {nombre_modelo: modelo_entrenado}
    """
    print("--> Entrenando modelos supervisados...")
    models = _build_models()

    # Optimización automática de k
    if tune_knn and "KNN" in models:
        best_k = _find_best_k(X_train, y_train)
        models["KNN"] = KNeighborsClassifier(n_neighbors=best_k, weights="distance")

    trained = {}
    for name, model in models.items():
        print(f"    [{name}] entrenando...")
        model.fit(X_train, y_train)

        if cv_evaluate:
            cv_score = cross_val_score(
                model, X_train, y_train, cv=5, scoring="f1_weighted"
            ).mean()
            print(f"      F1_weighted 5-fold CV: {cv_score:.3f}")

        joblib.dump(model, MODELS_DIR / f"{name}.joblib")
        print(f"      Guardado → {name}.joblib")
        trained[name] = model

    print(f"--> {len(trained)} modelos guardados en {MODELS_DIR}")
    return trained


def load_models(model_names: list = None) -> dict:
    """
    Carga modelos desde disco.

    Parameters
    ----------
    model_names : lista de nombres sin extensión, e.g. ["RandomForest", "KNN"].
                  Si None, carga todos los .joblib disponibles en models/.

    Returns
    -------
    dict : {nombre_modelo: modelo_cargado}
    """
    if model_names is None:
        model_names = [p.stem for p in MODELS_DIR.glob("*.joblib")]

    models = {}
    for name in model_names:
        path = MODELS_DIR / f"{name}.joblib"
        if path.exists():
            models[name] = joblib.load(path)
            print(f"    Cargado: {name}")
        else:
            print(f"    ⚠ No encontrado: {path}")
    return models



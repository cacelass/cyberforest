
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.decomposition import PCA
import joblib
from cyberforest.utils.paths import PROCESSED_DATA_DIR, ARTIFACTS_DIR


# ---------------------------------------------------------------------------
# Configuración de codificación ordinal
# ---------------------------------------------------------------------------
ORDINAL_MAPPINGS: dict = {
    # Ejemplo:
    # "education": {
    #     "illiterate": 1, "basic.4y": 2, "basic.6y": 3, "basic.9y": 4,
    #     "high.school": 5, "professional.course": 6, "university.degree": 7,
    #     "unknown": 8,
    # },
}

COLS_TO_DROP: list = [
    # "duration",     # fuga de datos
    # "nr_employed",  # alta correlación con euribor3m
]


def preprocess_data(
    df: pd.DataFrame,
    target_col: str,
    scaler_type: str = "standard",
    test_size: float = 0.2,
    random_state: int = 42,
    use_pca=None,
):
    """
    Pipeline completo de preprocesado para aprendizaje supervisado.

    Pasos:
      1. Elimina duplicados
      2. Feature engineering personalizable (_feature_engineering)
      3. Codificación ordinal (ORDINAL_MAPPINGS)
      4. Elimina columnas no deseadas (COLS_TO_DROP)
      5. Rellena nulos (media/moda)
      6. LabelEncoder para categóricas
      7. Train/test split estratificado
      8. Escalado (StandardScaler o MinMaxScaler)
      9. PCA opcional (use_pca)
      10. Guarda artefactos en artifacts/

    Parameters
    ----------
    scaler_type : "standard" | "minmax"
    use_pca     : None → sin PCA
                  float (0 < n < 1) → nº componentes por varianza explicada, e.g. 0.95
                  int  → nº fijo de componentes, e.g. 10

    Returns
    -------
    X_train, X_test, y_train, y_test  (arrays numpy)
    """
    print(f"--> Preprocesando datos (target='{target_col}', scaler='{scaler_type}', PCA={use_pca})...")

    df = df.copy()

    # 1. Duplicados
    n_before = len(df)
    df.drop_duplicates(inplace=True)
    if n_before - len(df):
        print(f"    Duplicados eliminados: {n_before - len(df)}")

    # 2. Feature engineering
    df = _feature_engineering(df)

    # 3. Codificación ordinal
    for col, mapping in ORDINAL_MAPPINGS.items():
        if col in df.columns:
            df[col] = df[col].map(mapping)
            print(f"    Codificación ordinal: {col}")

    # 4. Eliminar columnas
    cols_present = [c for c in COLS_TO_DROP if c in df.columns]
    if cols_present:
        df.drop(columns=cols_present, inplace=True)
        print(f"    Columnas eliminadas: {cols_present}")

    # 5. X / y
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # 6. Nulos
    num_cols = X.select_dtypes(include=[np.number]).columns
    cat_cols = X.select_dtypes(exclude=[np.number]).columns
    X[num_cols] = X[num_cols].fillna(X[num_cols].mean())
    for col in cat_cols:
        X[col] = X[col].fillna(X[col].mode()[0])

    # 7. LabelEncoder
    le = LabelEncoder()
    for col in cat_cols:
        X[col] = le.fit_transform(X[col].astype(str))

    if y.dtype == object or str(y.dtype) == "category":
        y = le.fit_transform(y.astype(str))
        joblib.dump(le, ARTIFACTS_DIR / "target_encoder.joblib")
        print("    Target codificado → target_encoder.joblib")

    # 8. Split estratificado
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y,
    )

    # 9. Escalado
    scaler = MinMaxScaler() if scaler_type == "minmax" else StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)
    joblib.dump(scaler, ARTIFACTS_DIR / "scaler.joblib")
    print(f"    Scaler guardado → scaler.joblib")

    # 10. PCA opcional
    if use_pca is not None:
        X_train, X_test = _apply_pca(X_train, X_test, use_pca)

    print(f"    Train: {X_train.shape} | Test: {X_test.shape}")
    print(f"    Proporción clases (train): {pd.Series(y_train).value_counts(normalize=True).to_dict()}")

    # Guardar conjuntos procesados
    pd.DataFrame(X_train).to_csv(PROCESSED_DATA_DIR / "X_train.csv", index=False)
    pd.DataFrame(X_test).to_csv(PROCESSED_DATA_DIR  / "X_test.csv",  index=False)
    pd.Series(y_train).to_csv(PROCESSED_DATA_DIR / "y_train.csv", index=False)
    pd.Series(y_test).to_csv(PROCESSED_DATA_DIR  / "y_test.csv",  index=False)

    return X_train, X_test, y_train, y_test


def _apply_pca(X_train, X_test, n_components):
    """
    Aplica PCA a train/test y guarda el objeto PCA en artifacts/.

    Parameters
    ----------
    n_components : float (varianza) | int (componentes fijos)
                   Ejemplos: 0.95 → 95% varianza | 10 → 10 componentes

    ¿Cuándo usar PCA antes del clasificador?
      - Muchas features correladas (|r| > 0.8 en varios pares)
      - Alta dimensionalidad (>50 features) → riesgo de maldición dimensional
      - Modelos lentos en alta dimensión (SVM, KNN)
      - Datos con ruido: PCA elimina las componentes de menor varianza

    ¿Cuándo NO usar PCA?
      - Cuando la interpretabilidad de features es crítica
      - Árboles y ensembles (RandomForest, XGBoost): ya gestionan la
        dimensionalidad internamente; PCA no suele mejorar resultados
    """
    pca = PCA(n_components=n_components, random_state=42)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca  = pca.transform(X_test)
    joblib.dump(pca, ARTIFACTS_DIR / "pca.joblib")

    n_comp = pca.n_components_
    var_exp = pca.explained_variance_ratio_.sum()
    print(f"    PCA: {X_train.shape[1]} → {n_comp} componentes "
          f"({var_exp:.1%} varianza explicada) → pca.joblib")
    return X_train_pca, X_test_pca


def _feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transformaciones y nuevas variables antes del modelado.
    Edita esta función según las necesidades del problema.

    Ejemplos comunes:
      df['was_contacted'] = df['pdays'].apply(lambda x: 0 if x == 999 else 1)
      df['total_loans']   = df['housing'] + df['loan']
    """
    # --- Añade tus transformaciones aquí ---
    return df


def process_input(df_new: pd.DataFrame) -> np.ndarray:
    """
    Preprocesa nuevos datos para inferencia usando los artefactos guardados.
    Aplica: feature_engineering → ordinal → drop → encode → scaler → PCA (si existe).
    """
    import os
    scaler = joblib.load(ARTIFACTS_DIR / "scaler.joblib")

    df_new = df_new.copy()
    df_new = _feature_engineering(df_new)

    for col, mapping in ORDINAL_MAPPINGS.items():
        if col in df_new.columns:
            df_new[col] = df_new[col].map(mapping)

    cols_present = [c for c in COLS_TO_DROP if c in df_new.columns]
    if cols_present:
        df_new.drop(columns=cols_present, inplace=True)

    cat_cols = df_new.select_dtypes(exclude=[np.number]).columns
    le = LabelEncoder()
    for col in cat_cols:
        df_new[col] = le.fit_transform(df_new[col].astype(str))

    num_cols = df_new.select_dtypes(include=[np.number]).columns
    df_new[num_cols] = df_new[num_cols].fillna(df_new[num_cols].mean())

    X = scaler.transform(df_new)

    pca_path = ARTIFACTS_DIR / "pca.joblib"
    if pca_path.exists():
        pca = joblib.load(pca_path)
        X = pca.transform(X)

    return X



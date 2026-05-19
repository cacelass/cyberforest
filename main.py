"""
Punto de entrada principal del proyecto.
Ejecutar: python main.py
"""

from cyberforest.data.make_dataset import load_data
from cyberforest.features.build_features import preprocess_data
from cyberforest.models.train_model import train_models
from cyberforest.models.predict_model import evaluate_models, DECISION_THRESHOLD

from cyberforest.models.predict_model import test_model as _test_model_legacy

from cyberforest.visualization.visualize import (
    plot_distributions,
    plot_correlation_matrix,
    plot_class_balance,
    plot_categorical_vs_target,
    plot_feature_importance,
    plot_pca_variance,
)

# ---------------------------------------------------------------------------
# Configuracion
# ---------------------------------------------------------------------------
# DATA_FILE eliminado — load_data() carga todos los CSV de data/raw/ automáticamente
TARGET_COL   = 'Label'
SCALER_TYPE  = 'standard'   # 'standard' | 'minmax'
TEST_SIZE    = 0.2
THRESHOLD    = DECISION_THRESHOLD

# PCA opcional: reducción de dimensionalidad antes del modelado.
# None → sin PCA | 0.95 → conservar 95% varianza | int → nº componentes fijo
USE_PCA      = None   # ← ajusta: None | 0.95 | 10

# Umbral de importancia para filtrar features significativas del LightGBM
GINI_THRESHOLD = 500


def run_full_pipeline() -> None:
    print('=' * 60)
    print('1. Cargando datos...')
    df = load_data()
    print(f'   Shape: {df.shape}')

    # print('\n2. EDA visual...')
    # plot_distributions(df, target_col=TARGET_COL)
    # plot_correlation_matrix(df)
    # plot_class_balance(df, target_col=TARGET_COL)
    # plot_categorical_vs_target(df, target_col=TARGET_COL)

    print('\n3. Preprocesando...')
    X_train, X_test, y_train, y_test = preprocess_data(
        df, target_col=TARGET_COL, scaler_type=SCALER_TYPE,
        test_size=TEST_SIZE, use_pca=USE_PCA,
    )

    print('\n4. Entrenando modelos...')
    models = train_models(X_train, y_train, tune_knn=True, cv_evaluate=True)

    print('\n4b. Clustering jerárquico (Nivel 2)...')
    from cyberforest.models.cluster_model import run_level2
    import numpy as np
    run_level2(X_train, np.array(y_train))

    print('\n5. Evaluando...')
    df_results = evaluate_models(
        models, X_train, y_train, X_test, y_test, threshold=THRESHOLD
    )

    print('\n6. Importancia de variables...')
    from cyberforest.utils.paths import PROCESSED_DATA_DIR
    import pandas as pd
    try:
        feature_names = pd.read_csv(PROCESSED_DATA_DIR / 'X_train.csv').columns.tolist()
    except FileNotFoundError:
        feature_names = [f'feature_{i}' for i in range(X_train.shape[1])]
    plot_feature_importance(models, feature_names)

    if USE_PCA is not None:
        print('\n7. Varianza explicada por PCA...')
        import joblib
        from cyberforest.utils.paths import ARTIFACTS_DIR
        try:
            pca = joblib.load(ARTIFACTS_DIR / 'pca.joblib')
            plot_pca_variance(pca)
        except FileNotFoundError:
            pass

    print('\n' + '=' * 60)
    print('Pipeline completado.')
    best = df_results.sort_values('Acc_test', ascending=False).iloc[0]
    print(f'Mejor modelo: {best.to_dict()}')


def _get_significant_features(lgbm_model, feature_names: list, gini_threshold: int = GINI_THRESHOLD):
    """
    Devuelve las features con importancia (split/gini) > gini_threshold en LightGBM.

    Returns
    -------
    list of (feature_name, importance) ordenadas de mayor a menor importancia.
    """
    importances = lgbm_model.feature_importances_  # gini acumulado por feature
    sig = [
        (name, imp)
        for name, imp in zip(feature_names, importances)
        if imp > gini_threshold
    ]
    sig.sort(key=lambda x: x[1], reverse=True)
    return sig


def _explain_prediction(pred_result: dict, proba: list, sig_features: list, sample_values: dict) -> None:
    """
    Imprime una explicación en lenguaje natural de la predicción.

    Parameters
    ----------
    pred_result   : {'grupo': str, 'subtipo': str}  o  {'grupo': 'BENIGN', 'subtipo': 'BENIGN'}
    proba         : lista de probabilidades por clase (índice = clase codificada)
    sig_features  : lista de (feature, importancia) significativas
    sample_values : dict {feature: valor} con los valores introducidos por el usuario
    """
    grupo   = pred_result['grupo']
    subtipo = pred_result['subtipo']

    max_proba = max(proba)
    confianza = f"{max_proba:.1%}"

    print()
    print('=' * 60)

    if grupo == 'BENIGN':
        print('  RESULTADO: Tráfico BENIGNO')
        print(f'  Confianza del modelo: {confianza}')
        print()
        print('  ¿Por qué el modelo lo clasifica como benigno?')
        print('  El modelo no detecta patrones asociados a ningún tipo de ataque conocido.')
        print('  Las características que más pesan en esta predicción y sus valores son:')
        for feat, imp in sig_features[:8]:
            val = sample_values.get(feat, 'N/A')
            print(f'    · {feat:<35} importancia={imp:>6}  valor={val}')
    else:
        print(f'  RESULTADO: CIBERATAQUE detectado')
        print(f'  Grupo de ataque : {grupo}')
        print(f'  Subtipo         : {subtipo}')
        print(f'  Confianza del modelo: {confianza}')
        print()
        print('  ¿Por qué el modelo lo clasifica como ataque?')
        _explain_attack_group(grupo, subtipo)
        print()
        print('  Características más relevantes (importancia > {}) y sus valores en esta muestra:'.format(GINI_THRESHOLD))
        for feat, imp in sig_features[:8]:
            val = sample_values.get(feat, 'N/A')
            print(f'    · {feat:<35} importancia={imp:>6}  valor={val}')

    print('=' * 60)


def _explain_attack_group(grupo: str, subtipo: str) -> None:
    """Imprime una descripción del tipo y subtipo de ataque detectado."""
    descripciones_grupo = {
        'DoS': (
            'Ataque de Denegación de Servicio (DoS): el objetivo es saturar los recursos '
            'del servidor enviando un volumen masivo de peticiones, impidiendo el servicio '
            'a usuarios legítimos. El modelo lo detecta por tasas de paquetes y duraciones '
            'de flujo anómalas.'
        ),
        'Brute Force': (
            'Ataque de Fuerza Bruta: el atacante prueba credenciales de forma sistemática '
            'contra servicios expuestos (FTP, SSH). El modelo lo detecta por el elevado '
            'número de intentos de conexión fallidos en poco tiempo.'
        ),
        'Web Attack': (
            'Ataque Web: incluye técnicas como inyección SQL, XSS o fuerza bruta HTTP. '
            'El modelo detecta patrones en el contenido y la estructura de las peticiones '
            'HTTP que no corresponden a tráfico legítimo.'
        ),
        'PortScan': (
            'Escaneo de puertos: el atacante sondea rangos de puertos para identificar '
            'servicios activos. El modelo lo detecta por la distribución inusual de '
            'conexiones a múltiples puertos en un intervalo corto.'
        ),
    }
    descripciones_subtipo = {
        'DoS Hulk':              'Hulk: genera peticiones HTTP únicas de forma masiva para evadir cachés.',
        'DoS GoldenEye':         'GoldenEye: mantiene conexiones HTTP abiertas para agotar los hilos del servidor.',
        'DoS slowloris':         'Slowloris: envía cabeceras HTTP incompletas para mantener conexiones abiertas.',
        'DoS Slowhttptest':      'Slowhttptest: variante lenta similar a Slowloris, agota el pool de conexiones.',
        'Heartbleed':            'Heartbleed: explota la vulnerabilidad CVE-2014-0160 en OpenSSL para leer memoria.',
        'FTP-Patator':           'FTP-Patator: fuerza bruta de credenciales contra el servicio FTP.',
        'SSH-Patator':           'SSH-Patator: fuerza bruta de credenciales contra el servicio SSH.',
        'Web Attack Brute Force':'Fuerza bruta web: prueba credenciales en formularios de inicio de sesión.',
        'Web Attack XSS':        'XSS: inyecta scripts maliciosos en páginas web para atacar a los usuarios.',
        'Web Attack Sql Injection': 'SQL Injection: inserta sentencias SQL para manipular la base de datos.',
        'Infiltration':          'Infiltración: acceso no autorizado a la red para exfiltrar o comprometer datos.',
        'PortScan':              'Escaneo sistemático de puertos para mapear servicios activos.',
    }

    desc_grupo = descripciones_grupo.get(grupo, f'Grupo: {grupo}')
    print(f'  {desc_grupo}')

    desc_sub = descripciones_subtipo.get(subtipo, '')
    if desc_sub and subtipo != grupo:
        print(f'  Subtipo específico → {desc_sub}')

def test_model_dftest() -> None:
    print('=' * 60)
    print('Probando modelo con muestra de df_test...')

    import joblib
    import pandas as pd

    from cyberforest.models.cluster_model import predict_hierarchical
    from cyberforest.models.train_model import load_models
    from cyberforest.features.build_features import process_input
    from cyberforest.utils.paths import ARTIFACTS_DIR, PROCESSED_DATA_DIR

    # Cargar modelo LightGBM entrenado
    trained = load_models(["LightGBM"])
    if "LightGBM" not in trained:
        print("  No se encontró LightGBM.joblib. Ejecuta primero el pipeline (opción 0).")
        return
    lgbm = trained["LightGBM"]

    # Cargar modelos de clustering y mapeos
    cluster_path  = ARTIFACTS_DIR / 'cluster_models.joblib'
    mappings_path = ARTIFACTS_DIR / 'cluster_mappings.joblib'

    if cluster_path.exists():
        cluster_models = joblib.load(cluster_path)
    else:
        print("  No se encontraron modelos de clustering. Ejecuta primero el pipeline completo (opción 0).")
        return

    if not mappings_path.exists():
        print("  No se encontró cluster_mappings.joblib. Ejecuta primero el pipeline completo (opción 0).")
        return
    mappings = joblib.load(mappings_path)

    # Cargar nombres de features
    feat_path = ARTIFACTS_DIR / 'feature_names.joblib'
    if feat_path.exists():
        feature_names = joblib.load(feat_path)
    else:
        x_train_path = PROCESSED_DATA_DIR / 'X_train.csv'
        if x_train_path.exists():
            feature_names = pd.read_csv(x_train_path).columns.tolist()
        else:
            print("  No se encontró feature_names.joblib ni X_train.csv.")
            return
    # Cargar datos de test para seleccionar una muestra aleatoria
    X_test = pd.read_csv(PROCESSED_DATA_DIR / 'X_test.csv')
    y_test = pd.read_csv(PROCESSED_DATA_DIR / 'y_test.csv').squeeze()
    X_test.columns = feature_names
    estado = 's'
    while estado:
        # Seleccionar una muestra aleatoria de X_test para probar la predicción

        fila = X_test.sample(1, random_state=None)# aleatoria cada vez
        idx = fila.index[0]
        etiqueta_real = y_test.loc[idx]
        print(f"\n  Muestra seleccionada (índice {idx}, etiqueta real: {etiqueta_real}):")
        print(fila)

        # Obtener las features significativas para esta muestra
        sig_features = _get_significant_features(lgbm, feature_names, GINI_THRESHOLD)

        # Obtener los valores de las features significativas para esta muestra
        sample_values = {feat: fila[feat].values[0] for feat, _ in sig_features}

        # ── Predicción jerárquica ─────────────────────────────────────
        try:
            pred_result = predict_hierarchical(fila, lgbm, cluster_models, mappings)
        except Exception as e:
            print(f"\n  Error en predicción jerárquica: {e}")
            return

        # ── Probabilidades del modelo base ────────────────────────────
        proba = lgbm.predict_proba(fila)[0].tolist()

        # ── Explicación ───────────────────────────────────────────────
        _explain_prediction(pred_result, proba, sig_features, sample_values)

        estado = input("\n¿Quieres probar otra muestra de test? (s/n): ").strip().lower() == 's'

def test_model() -> None:
    print('=' * 60)
    print('Probando modelo con muestra introducida por el usuario...')

    import joblib
    import numpy as np
    import pandas as pd

    from cyberforest.models.cluster_model import predict_hierarchical
    from cyberforest.models.train_model import load_models
    from cyberforest.features.build_features import process_input
    from cyberforest.utils.paths import ARTIFACTS_DIR, PROCESSED_DATA_DIR

    # ── 1. Cargar LightGBM ───────────────────────────────────────────
    trained = load_models(["LightGBM"])
    if "LightGBM" not in trained:
        print("  No se encontró LightGBM.joblib. Ejecuta primero el pipeline (opción 0).")
        return
    lgbm = trained["LightGBM"]

    # ── 2. Cargar modelos de clustering y mapeos ─────────────────────
    from cyberforest.utils.paths import MODELS_DIR
    cluster_path  = ARTIFACTS_DIR / 'cluster_models.joblib'
    mappings_path = ARTIFACTS_DIR / 'cluster_mappings.joblib'

    # cluster_models.joblib puede estar en artifacts/ (generado por run_level2)
    # o bien los kmeans individuales en models/ (generados por train_cluster_models).
    # Reconstruimos el dict desde los ficheros individuales si hace falta.
    if cluster_path.exists():
        cluster_models = joblib.load(cluster_path)
    else:
        group_file_map = {
            'DoS':         'kmeans_DoS.joblib',
            'Brute Force': 'kmeans_Brute_Force.joblib',
            'Web Attack':  'kmeans_Web_Attack.joblib',
            'PortScan':    'kmeans_PortScan.joblib',
        }
        cluster_models = {}
        for group, fname in group_file_map.items():
            p = MODELS_DIR / fname
            if p.exists():
                cluster_models[group] = joblib.load(p)
        if not cluster_models:
            print("  No se encontraron modelos de clustering. Ejecuta primero el pipeline completo (opción 0).")
            return

    if not mappings_path.exists():
        print("  No se encontró cluster_mappings.joblib. Ejecuta primero el pipeline completo (opción 0).")
        return
    mappings = joblib.load(mappings_path)

    # ── 3. Cargar nombres de features y filtrar las significativas ───
    feat_path = ARTIFACTS_DIR / 'feature_names.joblib'
    if feat_path.exists():
        feature_names = joblib.load(feat_path)
    else:
        x_train_path = PROCESSED_DATA_DIR / 'X_train.csv'
        if x_train_path.exists():
            feature_names = pd.read_csv(x_train_path).columns.tolist()
        else:
            print("  No se encontró feature_names.joblib ni X_train.csv.")
            return

    sig_features = _get_significant_features(lgbm, feature_names, GINI_THRESHOLD)

    if not sig_features:
        print(f"  Ninguna feature supera el umbral de importancia {GINI_THRESHOLD}.")
        print("  Se usarán las 10 features de mayor importancia como referencia.")
        all_imp = sorted(
            zip(feature_names, lgbm.feature_importances_),
            key=lambda x: x[1], reverse=True
        )
        sig_features = all_imp[:10]

    # ── 4. Pedir valores al usuario ──────────────────────────────────
    print()
    print(f"  El modelo usa {len(feature_names)} features en total.")
    print(f"  Las {len(sig_features)} más significativas (importancia > {GINI_THRESHOLD}) son:")
    for i, (feat, imp) in enumerate(sig_features, 1):
        print(f"    {i:>2}. {feat:<40} (importancia={imp})")

    print()
    print("  Introduce el valor de cada feature significativa.")
    print("  (Deja en blanco para usar la media como valor por defecto)\n")

    row = {feat: 0.0 for feat in feature_names}   # inicializar todo a 0
    sample_values = {}

    feature_means = joblib.load(ARTIFACTS_DIR / "feature_means.joblib")

    for feat, imp in sig_features:
        raw = input(f"  {feat}: ").strip()
        try:
            val = float(raw) if raw else feature_means.get(feat, 0.0)  # usar media del entrenamiento si el usuario deja en blanco
        except ValueError:
            val = 0.0
        row[feat] = val
        sample_values[feat] = val

    df_input = pd.DataFrame([row])

    # ── 5. Preprocesar ───────────────────────────────────────────────
    try:
        X_new = process_input(df_input)
    except Exception as e:
        print(f"\n  Error en preprocesado: {e}")
        return

    # ── 6. Predicción jerárquica ─────────────────────────────────────
    try:
        pred_result = predict_hierarchical(X_new, lgbm, cluster_models, mappings)
    except Exception as e:
        print(f"\n  Error en predicción jerárquica: {e}")
        return

    # ── 7. Probabilidades del modelo base ────────────────────────────
    proba = lgbm.predict_proba(X_new)[0].tolist()

    # ── 8. Explicación ───────────────────────────────────────────────
    _explain_prediction(pred_result, proba, sig_features, sample_values)


def main():
    print('=' * 60)
    accion = input('Ejecutar pipeline completo (0) o probar el modelo (1) o probar con datos de test (2)? (0/1/2): ').strip()
    if accion == '0':
        run_full_pipeline()
    elif accion == '1':
        test_model()
    elif accion == '2':
        test_model_dftest()
    else:
        print('Opción no válida. Ejecutando pipeline completo por defecto.')
        run_full_pipeline()


if __name__ == '__main__':
    main()
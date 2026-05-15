"""
cluster_model.py — Nivel 2 del pipeline jerárquico.
KMeans no supervisado por grupo de ataque para identificar subtipos.
"""
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

from cyberforest.utils.paths import MODELS_DIR, ARTIFACTS_DIR, FIGURES_DIR


# ---------------------------------------------------------------------------
# Grupos y sus subtipos esperados
# ---------------------------------------------------------------------------
SUBTYPE_MAP = {
    'DoS':         ['DoS Hulk', 'DoS GoldenEye', 'DoS slowloris', 'DoS Slowhttptest', 'Heartbleed'],
    'Brute Force': ['FTP-Patator', 'SSH-Patator'],
    'Web Attack':  ['Web Attack Brute Force', 'Web Attack XSS', 'Web Attack Sql Injection', 'Infiltration'],
    'PortScan':    ['PortScan'],
}


# ---------------------------------------------------------------------------
# Extraer subsets por grupo
# ---------------------------------------------------------------------------
def get_group_subsets(X_train, y_train_encoded):
    """
    Extrae los subsets de X_train para cada grupo de ataque.
    Excluye BENIGN — no necesita subclustering.
    
    Returns
    -------
    dict : {nombre_grupo: X_subset}
    """
    encoders  = joblib.load(ARTIFACTS_DIR / 'encoders.joblib')
    le_target = encoders['__target__']
    y_named   = le_target.inverse_transform(y_train_encoded)

    subsets = {}
    for group in SUBTYPE_MAP:
        mask = y_named == group
        X_sub = X_train[mask]
        print(f"  {group}: {X_sub.shape[0]} muestras")
        subsets[group] = X_sub

    return subsets

def find_optimal_k(X_subset, k_range=range(2, 7), name=""):
    """
    Busca la k óptima para KMeans usando método del codo y silhouette score.

    Parameters
    ----------
    X_subset  : array con las muestras del grupo
    k_range   : rango de k a explorar
    name      : nombre del grupo (para el título del gráfico)

    Returns
    -------
    int : k óptima según silhouette score
    """
    inertias    = []
    silhouettes = []

    for k in k_range:
        km     = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X_subset)
        inertias.append(km.inertia_)
        sil = silhouette_score(X_subset, labels, sample_size=5000, random_state=42)
        silhouettes.append(sil)
        print(f"    k={k} | inertia={km.inertia_:.0f} | silhouette={sil:.3f}")

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(list(k_range), inertias, 'o-', color='steelblue')
    axes[0].set_title(f'{name} — Método del codo')
    axes[0].set_xlabel('k')
    axes[0].set_ylabel('Inertia')

    axes[1].plot(list(k_range), silhouettes, 'o-', color='orange')
    axes[1].set_title(f'{name} — Silhouette Score')
    axes[1].set_xlabel('k')
    axes[1].set_ylabel('Silhouette')

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / f'kmeans_k_{name}.png', dpi=120)
    plt.show()

    best_k = list(k_range)[np.argmax(silhouettes)]
    print(f"    → k óptima: {best_k}")
    return best_k

def train_cluster_models(subsets, optimal_ks):
    """
    Entrena un KMeans por grupo con la k óptima encontrada.

    Parameters
    ----------
    subsets    : dict {grupo: X_subset} — salida de get_group_subsets
    optimal_ks : dict {grupo: k}        — salida de find_optimal_k por grupo

    Returns
    -------
    dict : {grupo: modelo KMeans entrenado}
    """
    cluster_models = {}

    for group, X_subset in subsets.items():
        k = optimal_ks[group]
        print(f"  [{group}] entrenando KMeans(k={k})...")

        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(X_subset)

        # Guardar modelo
        path = MODELS_DIR / f"kmeans_{group.replace(' ', '_')}.joblib"
        joblib.dump(km, path)
        print(f"    Guardado → {path.name}")

        cluster_models[group] = km

    return cluster_models

def map_clusters_to_subtypes(cluster_models, subsets, y_train_encoded):
    encoders  = joblib.load(ARTIFACTS_DIR / 'encoders.joblib')
    le_target = encoders['__target__']
    y_named   = le_target.inverse_transform(y_train_encoded)

    mappings = {}

    for group, km in cluster_models.items():
        X_subset = subsets[group]

        # Etiquetas del train que pertenecen a este grupo — mismo tamaño que X_subset
        group_mask = y_named == group
        group_raw  = y_named[group_mask]  # shape == (len(X_subset),)

        # Predecir cluster
        cluster_labels = km.predict(X_subset)

        cluster_map = {}
        for cluster_id in range(km.n_clusters):
            mask     = cluster_labels == cluster_id
            subtypes = group_raw[mask]
            if len(subtypes) == 0:
                cluster_map[cluster_id] = group
                continue
            dominant = pd.Series(subtypes).value_counts().index[0]
            pct      = pd.Series(subtypes).value_counts(normalize=True).iloc[0]
            cluster_map[cluster_id] = dominant
            print(f"  [{group}] cluster {cluster_id} → {dominant} ({pct:.1%})")

        mappings[group] = cluster_map

    joblib.dump(mappings, ARTIFACTS_DIR / 'cluster_mappings.joblib')
    print(f"\n  Mappings guardados → cluster_mappings.joblib")

    return cluster_models, mappings  # ← asegúrate que esta línea existe

def predict_hierarchical(X_new, level1_model, cluster_models, mappings):
    """
    Predicción jerárquica completa.
    Nivel 1: clasifica el grupo (BENIGN / DoS / PortScan / Brute Force / Web Attack)
    Nivel 2: identifica el subtipo dentro del grupo con KMeans

    Parameters
    ----------
    X_new          : array con las features del flujo a clasificar
    level1_model   : modelo LightGBM entrenado (Nivel 1)
    cluster_models : dict {grupo: KMeans} — salida de train_cluster_models
    mappings       : dict {grupo: {cluster_id: subtipo}} — salida de map_clusters_to_subtypes

    Returns
    -------
    dict : {grupo: str, subtipo: str}
    """
    encoders  = joblib.load(ARTIFACTS_DIR / 'encoders.joblib')
    le_target = encoders['__target__']

    # Nivel 1 — predecir grupo
    group_encoded = level1_model.predict(X_new)[0]
    group_name    = le_target.inverse_transform([group_encoded])[0]

    # Si es BENIGN no hace falta Nivel 2
    if group_name == 'BENIGN':
        return {'grupo': 'BENIGN', 'subtipo': 'BENIGN'}

    # Nivel 2 — predecir subtipo dentro del grupo
    if group_name in cluster_models:
        km         = cluster_models[group_name]
        cluster_id = km.predict(X_new)[0]
        subtipo    = mappings[group_name].get(cluster_id, group_name)
    else:
        subtipo = group_name

    return {'grupo': group_name, 'subtipo': subtipo}

def run_level2(X_train, y_train_encoded):
    """
    Orquesta el pipeline completo del Nivel 2.

    1. Extrae subsets por grupo
    2. Busca k óptima por grupo
    3. Entrena KMeans por grupo
    4. Mapea clusters a subtipos

    Parameters
    ----------
    X_train          : array con features de entrenamiento
    y_train_encoded  : array con etiquetas numéricas del train

    Returns
    -------
    cluster_models : dict {grupo: KMeans}
    mappings       : dict {grupo: {cluster_id: subtipo}}
    """
    print("=" * 60)
    print("NIVEL 2 — Clustering jerárquico por grupo")
    print("=" * 60)

    # 1. Extraer subsets
    print("\n1. Extrayendo subsets por grupo...")
    subsets = get_group_subsets(X_train, y_train_encoded)

    # 2. Buscar k óptima por grupo
    print("\n2. Buscando k óptima...")
    optimal_ks = {}
    for group, X_subset in subsets.items():
        print(f"\n  [{group}]")
        # PortScan tiene un solo subtipo real — k=1 no tiene sentido para KMeans
        if group == 'PortScan':
            print(f"    → k=1 (un solo subtipo real, no necesita clustering)")
            optimal_ks[group] = 1
            continue
        optimal_ks[group] = find_optimal_k(X_subset, k_range=range(2, 7), name=group)

    # 3. Entrenar KMeans
    print("\n3. Entrenando KMeans por grupo...")
    cluster_models = train_cluster_models(subsets, optimal_ks)

    # 4. Mapear clusters a subtipos
    print("\n4. Mapeando clusters a subtipos reales...")
    cluster_models, mappings = map_clusters_to_subtypes(cluster_models, subsets, y_train_encoded)

    print("\n" + "=" * 60)
    print("Nivel 2 completado.")
    print("=" * 60)

    # Guardar todos los cluster_models en un único joblib
    joblib.dump(cluster_models, ARTIFACTS_DIR / 'cluster_models.joblib')
    print("  cluster_models.joblib guardado")

    return cluster_models, mappings
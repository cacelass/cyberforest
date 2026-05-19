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
from cyberforest.features.build_features import ATTACK_GROUPS


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
    inertias    = []
    silhouettes = []

    for k in k_range:
        km     = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X_subset)
        inertias.append(km.inertia_)
        sil = silhouette_score(X_subset, labels, sample_size=5000, random_state=42)
        silhouettes.append(sil)
        print(f"    k={k} | inertia={km.inertia_:.0f} | silhouette={sil:.3f}")

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
    cluster_models = {}

    for group, X_subset in subsets.items():
        k = optimal_ks[group]
        print(f"  [{group}] entrenando KMeans(k={k})...")

        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(X_subset)

        path = MODELS_DIR / f"kmeans_{group.replace(' ', '_')}.joblib"
        joblib.dump(km, path)
        print(f"    Guardado → {path.name}")

        cluster_models[group] = km

    return cluster_models


def map_clusters_to_subtypes(cluster_models, subsets, y_train_encoded):
    encoders  = joblib.load(ARTIFACTS_DIR / 'encoders.joblib')
    le_target = encoders['__target__']
    y_named   = le_target.inverse_transform(y_train_encoded)

    mappings  = {}
    group_raw = joblib.load(ARTIFACTS_DIR / "label_original_train.joblib")

    for group, km in cluster_models.items():
        X_subset = subsets[group]

        # Filtrar label_original_train por grupo y resetear índice
        group_mask         = group_raw.map(ATTACK_GROUPS) == group
        group_raw_filtered = group_raw[group_mask].reset_index(drop=True)

        # Predecir solo sobre muestras reales (sin sintéticos SMOTE)
        X_real = X_subset[:len(group_raw_filtered)]
        cluster_labels = km.predict(X_real)

        cluster_map = {}
        for cluster_id in range(km.n_clusters):
            mask     = cluster_labels == cluster_id
            subtypes = group_raw_filtered[mask]
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

    return cluster_models, mappings


def predict_hierarchical(X_new, level1_model, cluster_models, mappings):
    encoders  = joblib.load(ARTIFACTS_DIR / 'encoders.joblib')
    le_target = encoders['__target__']

    group_encoded = level1_model.predict(X_new)[0]
    group_name    = le_target.inverse_transform([group_encoded])[0]

    if group_name == 'BENIGN':
        return {'grupo': 'BENIGN', 'subtipo': 'BENIGN'}

    if group_name in cluster_models:
        km         = cluster_models[group_name]
        cluster_id = km.predict(X_new)[0]
        subtipo    = mappings[group_name].get(cluster_id, group_name)
    else:
        subtipo = group_name

    return {'grupo': group_name, 'subtipo': subtipo}


def run_level2(X_train, y_train_encoded):
    print("=" * 60)
    print("NIVEL 2 — Clustering jerárquico por grupo")
    print("=" * 60)

    print("\n1. Extrayendo subsets por grupo...")
    subsets = get_group_subsets(X_train, y_train_encoded)

    print("\n2. Buscando k óptima...")
    optimal_ks = {}
    for group, X_subset in subsets.items():
        print(f"\n  [{group}]")
        if group == 'PortScan':
            print(f"    → k=1 (un solo subtipo real, no necesita clustering)")
            optimal_ks[group] = 1
            continue
        optimal_ks[group] = find_optimal_k(X_subset, k_range=range(2, 7), name=group)

    print("\n3. Entrenando KMeans por grupo...")
    cluster_models = train_cluster_models(subsets, optimal_ks)

    print("\n4. Mapeando clusters a subtipos reales...")
    cluster_models, mappings = map_clusters_to_subtypes(cluster_models, subsets, y_train_encoded)

    print("\n" + "=" * 60)
    print("Nivel 2 completado.")
    print("=" * 60)

    joblib.dump(cluster_models, ARTIFACTS_DIR / 'cluster_models.joblib')
    print("  cluster_models.joblib guardado")

    return cluster_models, mappings
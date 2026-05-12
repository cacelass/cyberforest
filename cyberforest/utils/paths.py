# cyberforest/utils/paths.py
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_DIR / "models"
ARTIFACTS_DIR = MODELS_DIR / "artifacts"
REPORTS_DIR = PROJECT_DIR / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
RUNS_DIR = PROJECT_DIR / "runs"  # TensorBoard logs

# Mapa de grupos de ataque → subtipos reales del dataset CICIDS2017.
# Centralizado aquí para que build_features y cluster_model compartan
# la misma definición sin importaciones circulares.
SUBTYPE_MAP = {
    'DoS':         ['DoS Hulk', 'DoS GoldenEye', 'DoS slowloris', 'DoS Slowhttptest', 'Heartbleed'],
    'Brute Force': ['FTP-Patator', 'SSH-Patator'],
    'Web Attack':  ['Web Attack Brute Force', 'Web Attack XSS', 'Web Attack Sql Injection', 'Infiltration'],
    'PortScan':    ['PortScan'],
}


def make_dirs():
    for dir_path in [
        RAW_DATA_DIR,
        INTERIM_DATA_DIR,
        PROCESSED_DATA_DIR,
        MODELS_DIR,
        ARTIFACTS_DIR,
        FIGURES_DIR,
        RUNS_DIR,
    ]:
        dir_path.mkdir(parents=True, exist_ok=True)


make_dirs()
import pandas as pd
from cyberforest.utils.paths import RAW_DATA_DIR


def load_data() -> pd.DataFrame:
    """
    Carga y combina todos los CSV de CIC-IDS2017 desde data/raw/.
    Limpia espacios en nombres de columnas.
    """
    csv_files = sorted(RAW_DATA_DIR.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(
            f"\n  No se encontraron archivos CSV en: {RAW_DATA_DIR}\n"
            f"  Descarga CIC-IDS2017 y coloca los CSV en data/raw/\n"
        )

    dfs = []
    for f in csv_files:
        print(f"--> Cargando {f.name}...")
        df = pd.read_csv(f, encoding="utf-8", low_memory=False)
        df.columns = df.columns.str.strip()  # elimina espacios en nombres
        dfs.append(df)

    df = pd.concat(dfs, ignore_index=True)
    print(f"\n    Total combinado: {df.shape}")
    print(f"    Clases: {df['Label'].value_counts().to_dict()}")
    return df
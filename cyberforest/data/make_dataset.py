import pandas as pd
from cyberforest.utils.paths import RAW_DATA_DIR



def load_data(filename: str = "<nombre>.csv") -> pd.DataFrame:
    """
    Carga el dataset desde data/raw/ con pandas.

    Parameters
    ----------
    filename : nombre del CSV en data/raw/  (solo el nombre, sin ruta)

    Returns
    -------
    pd.DataFrame
    """
    file_path = RAW_DATA_DIR / filename
    print(f"--> Cargando datos desde {file_path}...")
    if not file_path.exists():
        raise FileNotFoundError(
            f"\n  Archivo no encontrado: {file_path}\n"
            f"  Coloca tu dataset en: data/raw/{filename}\n"
            f"  O cambia DATA_FILE en main.py con el nombre correcto."
        )
    df = pd.read_csv(file_path)
    print(f"    Shape: {df.shape}")
    print(f"    Tipos:\n{df.dtypes.to_string()}")
    return df
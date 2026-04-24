import pandas as pd
from cyberforest.utils.paths import RAW_DATA_DIR

def load_data(filename=None) -> pd.DataFrame:
    """
    Carga uno o varios CSVs desde data/raw/.
    
    Parameters
    ----------
    filename : str o list[str], opcional
        - Si es str: carga ese archivo (comportamiento original)
        - Si es list: carga y concatena todos
        - Si es None: carga la lista por defecto (6 archivos no redundantes)
    
    Returns
    -------
    pd.DataFrame
    """
    # Lista por defecto (archivos seleccionados)
    default_files = [
        "Monday-WorkingHours.pcap_ISCX.csv",
        "Tuesday-WorkingHours.pcap_ISCX.csv",
        "Wednesday-workingHours.pcap_ISCX.csv",
        "Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv",
        "Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv",
        "Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv"
    ]
    
    if filename is None:
        filenames = default_files
    elif isinstance(filename, str):
        filenames = [filename]
    else:
        filenames = filename
    
    dfs = []
    for fname in filenames:
        file_path = RAW_DATA_DIR / fname
        print(f"--> Cargando {file_path}...")
        df = pd.read_csv(file_path, low_memory=False)
        df.columns = df.columns.str.strip()
        print(f"    Shape: {df.shape}")
        dfs.append(df)
    
    if len(dfs) == 1:
        return dfs[0]
    else:
        print("Concatenando DataFrames...")
        df_full = pd.concat(dfs, ignore_index=True)
        print(f"Shape total: {df_full.shape}")
        return df_full
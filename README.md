# CyberForest

> ForestGuard es un modelo de aprendizaje automático supervisado diseñado para detectar y clasificar diferentes tipos de ciberataques en redes informáticas. Utiliza un Random Forest (bosque de 100 árboles de decisión) entrenado con datasets de referencia como CIC-IDS2017, logrando alta precisión en entornos con tráfico desbalanceado y ataques heterogéneos.

**Tipo de ML:** `supervisado`  
**Autor:** Alejandro Cancelas Chapela  
**Versión:** 0.1.0

---

## Datos

Este proyecto utiliza el dataset **CIC-IDS2017**. Los archivos CSV originales no se incluyen en este repositorio debido a su gran tamaño (varios GB).

Para reproducir el proyecto:

1. Descarga los archivos desde la [fuente oficial](https://www.unb.ca/cic/datasets/ids-2017.html) o desde un mirror confiable.
2. Coloca los CSV necesarios en la carpeta `data/raw/`.

### Archivos necesarios (cubren todas las clases de ataque)

- `Monday-WorkingHours.pcap_ISCX.csv` (tráfico normal)
- `Tuesday-WorkingHours.pcap_ISCX.csv` (fuerza bruta FTP/SSH)
- `Wednesday-workingHours.pcap_ISCX.csv` (DDoS y Heartbleed)
- `Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv` (ataques web: SQLi, XSS)
- `Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv` (infiltración)
- `Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv` (port scan)

### Archivos omitidos (redundantes)

Los siguientes archivos se han excluido porque sus tipos de ataque ya están cubiertos por los archivos anteriores:

- `Friday-WorkingHours-Morning.pcap_ISCX.csv` (DDoS y ataques web redundantes)
- `Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv` (DDoS redundante)

> **Nota:** El dataset completo contiene más archivos, pero esta selección mantiene todas las clases de ataque sin pérdida de variedad y reduce el volumen de datos a procesar.

---

## Estructura del proyecto

```
cyberforest/
├── data/
│   ├── raw/            ← datos originales (nunca modificar)
│   ├── interim/        ← datos en proceso
│   └── processed/      ← datos listos para modelar
├── models/             ← modelos entrenados (.joblib / .pt)
│   └── artifacts/      ← encoders, scalers, etc.
├── notebooks/
│   ├── 0-0-...-Descargadatos.ipynb
│   ├── 0-1-...-ProcesamientoDatos.ipynb
│   └── 0-2-...-Ejecucion.ipynb
├── reports/figures/    ← gráficos generados
├── cyberforest/
│   ├── data/           make_dataset.py
│   ├── features/       build_features.py
│   ├── models/         train_model.py · predict_model.py
│   ├── visualization/  visualize.py
│   └── utils/          paths.py
├── tests/
├── main.py             ← pipeline completo
├── Makefile
└── pyproject.toml
```

## Inicio rápido

```bash
# 1. Instalar dependencias
make setup

# 2. Activar entorno
source .venv/bin/activate

# 3. Colocar datos en data/raw/ y editar DATA_FILE / TARGET_COL en main.py

# 4. Explorar con notebooks
invoke lab

# 5. Pipeline completo
python main.py
```

Consulta el archivo `ayuda` para más detalles.

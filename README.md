# cyberforest

> Pipeline jerárquico de detección y clasificación de ciberataques en redes — two-level IDS con LightGBM + KMeans sobre CIC-IDS2017.

![Python](https://img.shields.io/badge/Python-3.13-blue?logo=python&logoColor=white)
![ML Type](https://img.shields.io/badge/ML-Supervised%20%2B%20Unsupervised-orange)
![Dataset](https://img.shields.io/badge/Dataset-CIC--IDS2017-lightgrey)
![Tracking](https://img.shields.io/badge/Experiment%20Tracking-MLflow-blue?logo=mlflow)
![Version](https://img.shields.io/badge/Version-0.1.0-green)
![Author](https://img.shields.io/badge/Author-Alejandro%20Cancelas%20Chapela-blueviolet)

---

## Overview

**cyberforest** es un sistema de detección de intrusiones (IDS) con arquitectura jerárquica de dos niveles, entrenado sobre el dataset de referencia CIC-IDS2017 (~2.4M flujos de red, 79 features).

El sistema va más allá de la clasificación binaria benigno/ataque — identifica la **familia de ataque** (Nivel 1) y el **subtipo específico** dentro de esa familia (Nivel 2), reflejando cómo funcionan los IDS reales en entornos de producción.

---

## Arquitectura del pipeline

```
Flujo de red entrante
        ↓
┌─────────────────────────────────────┐
│  NIVEL 1 — LightGBM (supervisado)   │
│  Clasifica la familia de ataque     │
└─────────────────────────────────────┘
        ↓
   BENIGN → fin. No es ataque.
        ↓
┌─────────────────────────────────────┐
│  NIVEL 2 — KMeans (no supervisado)  │
│  Identifica el subtipo dentro       │
│  de la familia detectada            │
└─────────────────────────────────────┘
        ↓
  Resultado: {grupo, subtipo}
```

### Familias de ataque (Nivel 1)

| Familia | Subtipos originales |
|---|---|
| **DoS** | Hulk, GoldenEye, slowloris, Slowhttptest, Heartbleed |
| **PortScan** | PortScan |
| **Brute Force** | FTP-Patator, SSH-Patator |
| **Web Attack** | Brute Force web, XSS, SQL Injection, Infiltration |
| **BENIGN** | Tráfico normal |

El agrupamiento semántico fue necesario — el dataset original tiene 13 clases con distribución extremadamente desbalanceada (Heartbleed=11 muestras, SQLi=21). Entrenar con clases individuales producía precision < 0.01 en las minoritarias.

---

## Problema

CIC-IDS2017 presenta varios retos técnicos que el pipeline aborda explícitamente:

- **Desbalance extremo** — BENIGN representa el 82% del tráfico; Heartbleed el 0.0005%
- **Columnas corruptas** — `Flow Bytes/s` y `Flow Packets/s` contienen valores infinitos y negativos por divisiones por cero en el generador del dataset
- **Columnas duplicadas** — `Fwd Header Length` y `Fwd Header Length.1` son idénticas con valores imposibles (-32 billones)
- **Alta dimensionalidad** — 79 features originales, muchas correladas entre sí

---

## Feature Engineering

| Transformación | Justificación |
|---|---|
| `replace(inf, NaN)` | Limpieza de columnas corruptas del generador CIC-IDS2017 |
| `fwd_bwd_ratio` | DoS/PortScan tienen tráfico muy asimétrico (muchos fwd, pocos bwd) |
| `avg_packet_size` | Ataques usan paquetes pequeños y uniformes vs tráfico legítimo |
| `has_idle` | Ataques automatizados tienen Idle=0; conexiones reales tienen pausas |
| `log1p` en 12 features temporales | Skew extremo en Flow Duration, IAT Mean/Std/Max, Idle Mean/Max/Min |
| Drop 11 columnas | Corruptas, constantes, o correlación > 0.98 con otras |

---

## Modelos

### Nivel 1 — Clasificación supervisada

| Modelo | Accuracy | F1 (weighted) | Precision | Recall |
|---|---|---|---|---|
| **LightGBM** | **0.9993** | **0.9993** | **0.9993** | **0.9993** |
| RandomForest | 0.9974 | 0.9974 | 0.9976 | 0.9974 |

LightGBM seleccionado como modelo principal. Todos los experimentos trackeados en **MLflow**.

### Nivel 2 — Clustering no supervisado por familia

| Familia | k óptima | Silhouette |
|---|---|---|
| DoS | 2 | 0.546 |
| Brute Force | 6 | 0.959 |
| Web Attack | 2 | 0.988 |
| PortScan | — | un solo subtipo |

k óptima seleccionada por silhouette score sobre método del codo.

---

## Resultados

### Nivel 1 — por familia (test set)

| Familia | Precision | Recall | F1 | Soporte |
|---|---|---|---|---|
| BENIGN | 1.00 | 1.00 | 1.00 | 367,239 |
| DoS | 0.98 | 1.00 | 0.99 | 1,830 |
| PortScan | 1.00 | 1.00 | 1.00 | 18,164 |
| Brute Force | 0.99 | 1.00 | 0.99 | 38,752 |
| Web Attack | 0.59 | 0.98 | 0.74 | 436 |

> Web Attack tiene precision más baja por el desbalance residual incluso tras el agrupamiento (1,743 muestras vs 367k BENIGN).

---

## Dataset

**CIC-IDS2017 — Canadian Institute for Cybersecurity**

Dataset de referencia en detección de intrusiones. Captura de tráfico real de red durante una semana, con ataques generados de forma controlada. ~2.4M flujos, 79 features de red (IAT, longitud de paquetes, flags TCP, etc.).

### Archivos utilizados

| Archivo | Contenido |
|---|---|
| `Monday-WorkingHours.pcap_ISCX.csv` | Tráfico normal |
| `Tuesday-WorkingHours.pcap_ISCX.csv` | FTP/SSH Brute Force |
| `Wednesday-workingHours.pcap_ISCX.csv` | DoS + Heartbleed |
| `Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv` | SQLi, XSS |
| `Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv` | Infiltración |
| `Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv` | Port Scan |

Los archivos del viernes (DDoS, Web Attacks redundantes) se omiten por solapamiento con clases ya cubiertas.

---

## Project Structure

```
cyberforest/
├── data/
│   ├── raw/                  ← CSVs originales (nunca modificar)
│   ├── interim/              ← procesamiento intermedio
│   └── processed/            ← datos listos para modelar
│
├── models/
│   ├── artifacts/            ← encoders, scaler, mappings (.joblib)
│   ├── LightGBM.joblib
│   ├── kmeans_DoS.joblib
│   ├── kmeans_Brute_Force.joblib
│   └── kmeans_Web_Attack.joblib
│
├── notebooks/
│   ├── 0-0-DescargaDatos.ipynb
│   ├── 0-1-ProcesamientoDatos.ipynb
│   └── 0-2-Ejecucion.ipynb
│
├── reports/figures/          ← distribuciones, matrices, silhouette plots
│
├── cyberforest/
│   ├── data/                 make_dataset.py
│   ├── features/             build_features.py
│   ├── models/               train_model.py · predict_model.py · cluster_model.py
│   ├── visualization/        visualize.py
│   └── utils/                paths.py
│
├── tests/
├── main.py                   ← pipeline completo
├── Makefile
└── pyproject.toml
```

---

## Quick Start

```bash
# 1. Instalar dependencias
make setup

# 2. Activar entorno
source .venv/bin/activate

# 3. Descargar CIC-IDS2017 y colocar CSVs en data/raw/

# 4. Explorar notebooks
invoke lab

# 5. Pipeline completo
python main.py
```

---

## Design Philosophy

- **Arquitectura jerárquica** — refleja cómo funcionan los IDS reales: primero detectar, luego clasificar
- **Agrupamiento semántico** — las 13 clases originales se fusionan en 5 familias con criterio de dominio, no estadístico
- **Nivel 2 no supervisado** — KMeans descubre subtipos sin usar etiquetas, validando que los patrones de red emergen naturalmente
- **Pipeline modular** — cada etapa (ingestión, features, Nivel 1, Nivel 2) es independientemente testeable
- **Sin PCA** — LightGBM gestiona la dimensionalidad internamente; PCA eliminaría la interpretabilidad de features de red
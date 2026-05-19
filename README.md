# cyberforest

> Un modelo de machine learning que detecta ciberataques — pipeline jerárquico de dos niveles con LightGBM + KMeans sobre CIC-IDS2017.

![Python](https://img.shields.io/badge/Python-3.13-blue?logo=python&logoColor=white)
![ML Type](https://img.shields.io/badge/ML-Supervisado%20%2B%20No%20Supervisado-orange)
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

## El problema del desbalanceo

CIC-IDS2017 presenta un desbalanceo severo: BENIGN representa el 82% del tráfico original. Un modelo entrenado directamente sobre los datos originales aprende a predecir siempre benigno — obteniendo 82% de accuracy sin detectar ningún ataque real.

### Estrategia aplicada

El pipeline aplica tres técnicas combinadas para corregir el desbalanceo:

**1. Submuestreo de BENIGN** — Se elimina el 90% de las muestras BENIGN (con prioridad a las que tienen nulos), reduciéndolas de 1.98M a ~198k. Esto equilibra la distribución sin generar datos artificiales para la clase mayoritaria.

**2. SMOTE selectivo** — Se aplica oversampling sintético únicamente sobre las clases con suficientes muestras reales para que la interpolación tenga sentido: Web Attack (~1,743 muestras → 2,000) y Brute Force (~7,322 muestras → ampliado). Clases con menos de 50 muestras (Heartbleed=11, Infiltration=36, SQLi=21) se excluyen — con tan pocas muestras reales, SMOTE solo replicaría los mismos patrones.

**3. `class_weight="balanced"`** — Ambos modelos (LightGBM y RandomForest) reciben pesos inversamente proporcionales a la frecuencia de cada clase durante el entrenamiento.

### Distribución resultante (train)

| Clase | Proporción original | Proporción final |
|---|---|---|
| BENIGN | 82.3% | 38.1% |
| DoS | 9.6% | 40.2% |
| PortScan | 6.6% | 18.8% |
| Brute Force | 0.6% | 1.9% |
| Web Attack | 0.07% | 1.0% |

---

## Feature Selection

LightGBM calcula la importancia de cada feature mediante **Gini acumulado** (número de splits ponderados por ganancia de información). Las 19 features con importancia > 500 son las que más contribuyen a las decisiones del modelo:

| Rank | Feature | Importancia |
|---|---|---|
| 1 | Flow IAT Min | 4,381 |
| 2 | Destination Port | 4,132 |
| 3 | Fwd IAT Min | 3,849 |
| 4 | Init_Win_bytes_forward | 3,521 |
| 5 | Init_Win_bytes_backward | 2,296 |
| ... | ... | ... |
| 19 | Fwd Packet Length Max | 564 |

El umbral de 500 no es arbitrario — separa las features con contribución real de las que el modelo usa marginalmente. Las 49 features restantes (importancia < 500) aportan menos del 15% de la ganancia total combinada.

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
| `feature_means.joblib` | Media de entrenamiento guardada para rellenar features desconocidas en inferencia |

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
| DoS | 3 | 0.551 |
| Brute Force | 6 | 0.965 |
| Web Attack | 2 | 0.983 |
| PortScan | — | un solo subtipo |

k óptima seleccionada por silhouette score sobre método del codo.

---

## Resultados

### Por familia (test set, LightGBM)

| Familia | Precision | Recall | F1 | Soporte |
|---|---|---|---|---|
| BENIGN | 1.00 | 1.00 | 1.00 | 36,724 |
| Brute Force | 1.00 | 1.00 | 1.00 | 1,830 |
| DoS | 1.00 | 1.00 | 1.00 | 38,752 |
| PortScan | 1.00 | 1.00 | 1.00 | 18,164 |
| Web Attack | 0.99 | 1.00 | 0.99 | 436 |

> Web Attack pasó de F1=0.74 (dataset original desbalanceado) a F1=0.99 tras aplicar la estrategia de equilibrado.

---

## Dataset

**CIC-IDS2017 — Canadian Institute for Cybersecurity**

Dataset de referencia en detección de intrusiones. Captura de tráfico real de red durante una semana, con ataques generados de forma controlada. ~2.4M flujos, 79 features de red (IAT, longitud de paquetes, flags TCP, etc.).

| Archivo | Contenido |
|---|---|
| `Monday-WorkingHours.pcap_ISCX.csv` | Tráfico normal |
| `Tuesday-WorkingHours.pcap_ISCX.csv` | FTP/SSH Brute Force |
| `Wednesday-workingHours.pcap_ISCX.csv` | DoS + Heartbleed |
| `Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv` | SQLi, XSS |
| `Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv` | Infiltración |
| `Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv` | Port Scan |

---

## Uso

```bash
# 1. Instalar dependencias
make setup

# 2. Activar entorno
source .venv/bin/activate

# 3. Descargar CIC-IDS2017 y colocar CSVs en data/raw/

# 4. Pipeline completo (carga → preprocesado → entrenamiento → evaluación)
python main.py
# → Opción 0
```

### Modos de inferencia

```
python main.py
```

| Opción | Descripción |
|---|---|
| `0` | Pipeline completo — carga datos, entrena y evalúa |
| `1` | Predicción manual — introduce valores de features por teclado |
| `2` | Predicción sobre test real — selecciona una fila aleatoria del conjunto de test y verifica la predicción contra la etiqueta real |

> **Nota sobre la opción 1:** el modelo fue entrenado con 68 features que interactúan entre sí. Introducir valores parciales (las 19 más importantes) con el resto rellenado por la media del training puede producir predicciones poco fiables. La opción 2 es la prueba más honesta del modelo — usa datos reales que nunca vio durante el entrenamiento.

---

## Project Structure

```
cyberforest/
├── data/
│   ├── raw/                  ← CSVs originales (nunca modificar)
│   ├── interim/              ← procesamiento intermedio
│   └── processed/            ← X_train, X_test, y_train, y_test escalados
│
├── models/
│   ├── artifacts/            ← encoders, scaler, feature_means, mappings (.joblib)
│   ├── LightGBM.joblib
│   ├── RandomForest.joblib
│   ├── kmeans_DoS.joblib
│   ├── kmeans_Brute_Force.joblib
│   └── kmeans_Web_Attack.joblib
│
├── notebooks/
│   ├── 0-0-DescargaDatos.ipynb
│   ├── 0-1-ProcesamientoDatos.ipynb
│   └── 0-2-Ejecucion.ipynb
│
├── reports/figures/          ← distribuciones, matrices de confusión, silhouette plots
│
├── cyberforest/
│   ├── data/                 make_dataset.py  ← carga + submuestreo BENIGN
│   ├── features/             build_features.py ← preprocesado + SMOTE
│   ├── models/               train_model.py · predict_model.py · cluster_model.py
│   ├── visualization/        visualize.py
│   └── utils/                paths.py
│
├── tests/
├── main.py                   ← punto de entrada (opciones 0/1/2)
├── Makefile
└── pyproject.toml
```

---

## Design Philosophy

- **Arquitectura jerárquica** — refleja cómo funcionan los IDS reales: primero detectar, luego clasificar
- **Agrupamiento semántico** — las 13 clases originales se fusionan en 5 familias con criterio de dominio, no estadístico
- **Nivel 2 no supervisado** — KMeans descubre subtipos sin usar etiquetas, validando que los patrones de red emergen naturalmente
- **Desbalanceo por capas** — submuestreo + SMOTE selectivo + class_weight, cada técnica aplicada donde tiene sentido
- **Inferencia honesta** — `feature_means.joblib` garantiza que los valores desconocidos en inferencia se rellenan con la distribución real del training, no con ceros
- **Pipeline modular** — cada etapa es independientemente testeable
- **Sin PCA** — LightGBM gestiona la dimensionalidad internamente; PCA eliminaría la interpretabilidad de features de red

---

## Limitación conocida

El modo de inferencia manual (opción 1) está pensado para demostración. En un entorno real, el sistema recibiría flujos de red capturados con herramientas como **CICFlowMeter** o **tcpdump**, que generan automáticamente las 68 features en el formato exacto del dataset de entrenamiento. La introducción manual de features parciales no refleja el caso de uso productivo.

---

## Créditos

Proyecto creado con la plantilla **[dskit](https://github.com/cacelass/dskit)** — estructura de proyectos ML reproducibles.

Dataset: [CIC-IDS2017](https://www.unb.ca/cic/datasets/ids-2017.html) — Canadian Institute for Cybersecurity, University of New Brunswick.
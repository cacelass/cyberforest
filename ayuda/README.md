# Ayuda

Carpeta para recursos de referencia del proyecto. Papers, cheatsheets, notas
metodológicas o cualquier documentación de apoyo que no forme parte del código.

---

## Comandos esenciales

```bash
# Entorno
uv sync --extra dev --extra supervisado
source .venv/bin/activate

# Pipeline
make run          # main.py completo
make data         # solo carga/preproceso de datos
make train        # solo entrenamiento
make predict      # solo predicciones → reports/

# Calidad
make test         # pytest completo
make smoke        # tests de humo (rápidos)
make lint         # ruff check
make format       # ruff format

# Debug de rendimiento
make profile      # cProfile → reports/profile.prof
                  # luego: snakeviz reports/profile.prof

# Limpieza
make clean        # __pycache__ y cachés
make clean-models # borra .joblib / .pt
make clean-all    # todo

```

---

## Tipo de ML: `supervisado` · Tarea: `clasificacion` · MLflow activo


### MLflow

```bash
make mlflow        # UI en http://localhost:5000
```

Cada entrenamiento crea un run en el experimento `cyberforest`.
Los modelos se registran en el Model Registry como `cyberforest_<NombreModelo>`.
Artifacts: pesos `.joblib` / `.pt` + figuras de evaluacion.





### Modelos disponibles (activo: `RandomForest`)

| Modelo | Cuándo usar |
|---|---|
| KNN | Lazy learner, buena línea base. Escalar features antes |
| LogisticRegression | Clasificación binaria, interpretable, probabilidades calibradas |
| DecisionTree | Caja blanca, útil para explicabilidad |
| RandomForest | Robusto, feature importance, buen por defecto |

| LightGBM | Leaf-wise boosting. Más rápido que XGBoost en datos grandes |

Cambiar el modelo activo: edita `model_type` en `json` y regenera,
o descomenta/comenta modelos directamente en `_build_models()` de `train_model.py`.








---

## Estructura de outputs

```
reports/
├── figures/
│   ├── cm_<modelo>.png        # matriz de confusión
│   └── proba_dist_*.png     # distribución de probabilidades (binario)
└── resultados.csv            # métricas comparativas
```

---

> Esta carpeta no se publica ni se incluye en el paquete. Es solo un espacio de trabajo local.
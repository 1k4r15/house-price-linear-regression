[README_house_price.md](https://github.com/user-attachments/files/33015414/README_house_price.md)
# Predicción de Precios de Vivienda con Regresión Lineal — Descenso de Gradientes vs. OLS

Segundo proyecto del portafolio de Machine Learning. Implementa regresión lineal **desde cero** con descenso de gradientes (NumPy puro) y la valida comparándola contra la solución analítica (Mínimos Cuadrados Ordinarios) y contra `sklearn`.

## Objetivo

Predecir el precio de una vivienda a partir del ingreso promedio de la zona, la antigüedad promedio de las casas, el número promedio de habitaciones y dormitorios, y la población del área — implementando el algoritmo de entrenamiento manualmente para demostrar comprensión matemática, no solo uso de una librería.

## Dataset

**[USA Housing Dataset](https://github.com/huzaifsayed/Linear-Regression-Model-for-House-Price-Prediction/blob/master/USA_Housing.csv)** — dataset sintético, 5,000 filas, 7 columnas originales (`Avg. Area Income`, `Avg. Area House Age`, `Avg. Area Number of Rooms`, `Avg. Area Number of Bedrooms`, `Area Population`, `Price`, `Address`). Sin valores nulos. Se descarta `Address`.

## Metodología

### 1. Exploración de datos (EDA)
- A diferencia del dataset de autos del Bloque 1, `Price` tiene una distribución **simétrica** (campana centrada en ~$1.2M) — no fue necesario aplicar transformación logarítmica.
- Matriz de correlación: `Avg. Area Income` es la variable más correlacionada con `Price` (0.64), seguida de `House Age` (0.45), `Population` (0.41) y `Rooms` (0.34). `Bedrooms` es la más débil (0.17).
- Se detectó multicolinealidad moderada entre `Rooms` y `Bedrooms` (0.46) — no lo suficientemente alta (el umbral de preocupación suele estar cerca de 0.8–0.9) como para eliminar alguna de las dos.

### 2. Preparación
- Split holdout 80/20 (4,000 train / 1,000 test, `random_state=42`).
- Estandarización (`StandardScaler`) ajustada **solo** sobre el set de entrenamiento.

### 3. Modelado: tres implementaciones comparadas
- **Descenso de Gradientes (desde cero, NumPy):** algoritmo iterativo (`lr=0.1`, 1000 epochs), con cálculo explícito del gradiente del error cuadrático medio y actualización manual de pesos.
- **Mínimos Cuadrados Ordinarios — ecuación normal (desde cero, NumPy):** solución analítica cerrada, `w = (XᵀX)⁻¹Xᵀy`.
- **`sklearn.LinearRegression`:** usado como referencia de validación.

## Resultados

**Validación de la implementación — los tres métodos convergen exactamente al mismo resultado:**

| Feature | Gradiente Descendente | OLS (ecuación normal) | Sklearn |
|---|---|---|---|
| Intercepto | 1,229,576.99 | 1,229,576.99 | 1,229,576.99 |
| Avg. Area Income | 231,741.88 | 231,741.88 | 231,741.88 |
| Avg. Area House Age | 163,580.78 | 163,580.78 | 163,580.78 |
| Avg. Area Number of Rooms | 120,724.77 | 120,724.77 | 120,724.77 |
| Avg. Area Number of Bedrooms | 2,992.45 | 2,992.45 | 2,992.45 |
| Area Population | 152,235.90 | 152,235.90 | 152,235.90 |

**Evaluación sobre el set de prueba (1,000 casas nunca vistas por el modelo):**

| Métrica | Valor |
|---|---|
| MAE | 80,879 (≈ 6.6% del precio promedio) |
| RMSE | 100,444 |
| R² | 0.918 |

## Hallazgos clave

- La implementación manual de descenso de gradientes converge al **mismo óptimo exacto** que la solución analítica (OLS) y que `sklearn` — los pesos coinciden hasta el segundo decimal.
- El descenso de gradientes convergió casi por completo en las primeras ~15–20 iteraciones de las 1,000 totales — para este problema, muchas menos iteraciones hubieran bastado.
- El RMSE del set de prueba es casi idéntico al estimado a partir del costo final de entrenamiento — el modelo generaliza bien, sin señales de sobreajuste.
- A diferencia del proyecto de KNN del Bloque 1, acá **no hay subestimación sistemática en ningún rango de precio**: las predicciones siguen la diagonal real-vs-predicho de forma consistente en todo el rango. La diferencia estructural entre ambos proyectos: en KNN el problema era escasez de datos en ciertas zonas del espacio de features; acá la relación entre features y precio es genuinamente lineal (el dataset es sintético, generado así a propósito) — justo el supuesto que una regresión lineal necesita para funcionar bien.
- Con las features estandarizadas, los coeficientes son directamente comparables entre sí: `Avg. Area Income` pesa más que cualquier otra variable, consistente con tener la correlación más alta con el precio en la exploración inicial.

## Limitaciones

- El dataset es sintético — la relación lineal casi perfecta entre features y precio no necesariamente se replica en datos de vivienda reales, donde suelen existir no linealidades (efectos de ubicación, umbrales de tamaño, etc.).
- Solo 5 features disponibles; un dataset real de vivienda normalmente incluye decenas de variables (ubicación exacta, año de construcción, condición, remodelaciones, etc.).
- No se probó regularización (Ridge/Lasso) ni términos polinomiales — relevantes si el supuesto de linealidad no se sostuviera tan bien como en este caso.

## Estructura del repositorio

```
house-price-linear-regression/
├── README.md
├── data/
│   └── raw/
│       └── USA_Housing.csv
├── notebooks/
│   └── 01_regresion_lineal_gradiente_vs_ols.ipynb
└── requirements.txt
```

## Cómo ejecutar el proyecto

1. Abre el notebook en Google Colab (botón "Open in Colab" disponible desde la vista del archivo en GitHub).
2. Instala las dependencias: `pip install -r requirements.txt`
3. Corre el notebook de principio a fin.

## Tecnologías

Python · NumPy · pandas · scikit-learn · matplotlib · seaborn · Google Colab

---

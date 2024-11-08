# Optimización y Evaluación de Modelos de Aprendizaje Automático

Este repositorio contiene dos carpetas: `classification(hc-mci)` contiene el código y datos para la **clasificación** de participantes entre HC y MCI, y `regression(t-moca)` contiene el código y datos para la **regresión** (predicción numérica) de la puntuación T-MoCA de los participantes.

En cada carpeta, hay dos scripts de Python que realizan optimización de selección de features y ajuste/predicción de modelos en un conjunto de datos. Adicionalmente, hay un Jupyter Notebook que documenta y contiene todo el código de ambos scripts `feature_selection.py` y `fit_predict.py`.

## Dependencias
Los scripts requieren las siguientes bibliotecas de Python:
- `numpy`
- `pandas`
- `matplotlib`
- `scikit-learn`

## Configuración
Todos los scripts permiten personalizar los modelos de ML a evaluar modificando la lista `estimators`. Cada modelo está representado como un diccionario con las siguientes claves:
- `model`: instancia del modelo de aprendizaje automático
- `name`: identificador único para el modelo

## Guías de Uso

### Ejecutar la Selección de features
```bash
# Dentro de ./classification(hc-mci) o ./regression(t-moca)
python feature_selection.py
```
- Asegúrese de tener las dependencias requeridas instaladas
- Asegúrese de que los archivos CSV de entrada estén en el directorio de trabajo (`digimoca_results_totales.csv` y `datos_participantes_totales.csv`)
- Crea un directorio `results` si no existe
- Genera gráficos de importancia de features y archivos CSV con las mejores features

### Ejecutar la Predicción del Modelo
```bash
# Dentro de ./classification(hc-mci) o ./regression(t-moca)
python fit_predict.py
```
- Requiere que el paso de selección de features se haya completado primero
- Utiliza los archivos CSV de mejores features del paso de selección de features
- Genera el scatter plot (regresión) o curva ROC (clasificación)
- Para cada modelo, calcula y muestra el R² y RMSE (regresión) o el AUC (clasificación)

## Optimización de Selección de features

El script `feature_selection.py` realiza la optimización de selección de features en un conjunto de datos utilizando varios modelos de aprendizaje automático.

### features
- Soporta una variedad de modelos de aprendizaje automático para la optimización de selección de features
- Calcula la importancia por permutación de cada característica para determinar su importancia relativa
- Evalúa el rendimiento del modelo (R² para regresión, AUC para clasificación) a medida que aumenta el número de features
- Identifica el número óptimo de features para cada modelo
- Guarda las mejores features para cada modelo en archivos CSV separados
- Genera un gráfico para cada modelo, visualizando las importancias de las features y las puntuaciones R²/AUC

### Salida
El script genera la siguiente salida:
1. Archivos CSV que contienen las mejores features para cada modelo, guardados en el directorio `results/` con la convención de nombres `best_features_<model_name>.csv`.
2. Imágenes PNG que muestran los gráficos de importancia de features para cada modelo, guardadas en el directorio `results/` con la convención de nombres `feature_importances_<model_name>.png`.

## Ajuste y Predicción del Modelo

El script `fit_predict.py` está diseñado para realizar el ajuste y predicción del modelo en un conjunto de datos, utilizando las mejores features identificadas en el script anterior de optimización de selección de features.

1. En el caso de regresión, genera un scatter plot de los valores predichos vs. valores reales para 4 modelos de ML seleccionados.
2. En el caso de clasificación, genera un gráfico de curva ROC para cada modelo de ML.

### features
- Carga las mejores features para cada modelo de aprendizaje automático del script anterior de optimización de selección de features
- Realiza predicciones de validación cruzada para cada modelo usando las mejores features de `results/best_features_<model_name>.csv`
- **REGRESIÓN**
    - Genera un scatter plot 2x2 de los valores predichos vs. valores reales para todos los modelos
    - Calcula y muestra la puntuación R² y el Error Cuadrático Medio (RMSE) para cada modelo
- **CLASIFICACIÓN**
    - Genera un gráfico de curva ROC para todos los modelos
    - Calcula y muestra el Área Bajo la Curva (AUC)

### Salida
El script genera la siguiente salida:
1. **REGRESIÓN:** Una imagen PNG que muestra el scatter plot de los valores predichos vs. valores reales para 4 modelos seleccionados, guardada en `results/scatter_plot_<model_name_1>_<model_name_2>_..._.png`.
2. **CLASIFICACIÓN:** Una imagen PNG que muestra el gráfico de curva ROC para todos los modelos, guardada en `results/roc_<model_name_1>_<model_name_2>_..._.png`.

# Cuadernos de Machine Learning con PySpark

A continuación se detalla el contenido y flujo de trabajo de cada cuaderno desarrollado para tareas de **Machine Learning** con PySpark.

---

## 1. ML_Preprocesing.ipynb

**Objetivo:**  
Aprender y demostrar sentencias clave de PySpark para el **preprocesamiento de datos**.

**Técnicas:**
- Identificación de filas duplicadas.
- Uso de la función `explode` para expandir columnas con estructuras de datos complejas (e.g., arrays/listas) en múltiples filas.

**Conceptos Clave:**
- Manejo de DataFrames de PySpark.
- Consulta y transformación de esquemas.

---

## 2. ML_NoSupervisado00.ipynb

Cuaderno dedicado a la implementación de un modelo de **Aprendizaje No Supervisado**.

**Objetivo:**  
Aplicar el algoritmo de **K-Medias (K-Means)** para clustering de datos, con enfoque en preprocesamiento para MLlib.

**Conjunto de Datos:**  
- Nombre: Iris dataset (clásico conjunto de datos de biología).  
- Uso: Demostrar la segmentación de datos en clústeres.

**Flujo de Trabajo con MLlib:**

1. **Carga y Limpieza:**
   - Carga del CSV en un DataFrame de PySpark.
   - Chequeo de valores nulos.

2. **Preparación de Características (Feature Engineering):**
   - `StringIndexer`: Convierte la variable categórica `Species` a un índice numérico (`Species_Indexed`).
   - `VectorAssembler`: Combina todas las características numéricas en un único vector (`features`), requerido por MLlib.

3. **Escalado de Datos:**
   - Uso de `StandardScaler` para normalizar las características (`scaledFeatures`), crucial para algoritmos basados en distancia como K-Means.

4. **Modelado:**
   - Entrenamiento del modelo **KMeans** sobre las características escaladas.

5. **Evaluación:**
   - Se utiliza `ClusteringEvaluator` con **Silhouette Score** para determinar el número óptimo de clústeres (`k`).
   - Resultado Clave:  
     - k=2 → Silhouette Score ≈ 0.7728  
     - k=3 → Silhouette Score ≈ 0.6523

6. **Visualización:**  
   - Resultados visualizados utilizando Seaborn/Matplotlib.

---

## 3. ML_Supervisado.ipynb

Cuaderno para un problema de **Aprendizaje Supervisado (Clasificación)** usando PySpark ML.

**Objetivo:**  
Implementar y optimizar un modelo de **Regresión Logística** utilizando **Pipelines** y técnicas de **validación cruzada** para ajuste de hiperparámetros.

**Flujo de Trabajo:**

1. **Inicialización:**
   - Configuración de `SparkSession` y carga de librerías necesarias.

2. **Preprocesamiento:**
   - Limpieza de datos (manejo de nulos).
   - Preparación de características con `VectorAssembler`.

3. **Pipeline:**
   - Construcción de un `Pipeline` que encadena preprocesamiento y el estimador principal (`LogisticRegression`).

4. **Ajuste de Hiperparámetros (Tuning):**
   - `ParamGridBuilder`: Define la grilla de hiperparámetros (`regParam`, `elasticNetParam`).
   - `TrainValidationSplit`: Divide el conjunto de entrenamiento y evalúa cada combinación, seleccionando el mejor modelo.

5. **Evaluación:**
   - Medida de rendimiento con `BinaryClassificationEvaluator`.
   - Métrica: **Área Bajo la Curva ROC (areaUnderROC)**.

6. **Resultado Final:**  
   - Mejor modelo ajustado: `areaUnderROC = 0.90625`.

---
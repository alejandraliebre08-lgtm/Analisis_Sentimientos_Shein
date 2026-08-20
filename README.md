# Análisis de Sentimientos en Reseñas de SHEIN

Proyecto Final — Data Science III (Coderhouse)
**Autora:** Alejandra Lorena Liebre

## Objetivo

Clasificar automáticamente el sentimiento de reseñas reales de usuarios de la app de SHEIN en tres categorías: **negativo**, **neutral** y **positivo**, comparando el desempeño de una red neuronal (Deep Learning) contra modelos clásicos de Machine Learning.

## Dataset

Se utilizó el dataset público **ShoppingAppReviews** (Kaggle, autora Jocelyn Dumlao), que contiene reseñas reales de usuarios sobre 12 apps de compras. Se filtró el subconjunto correspondiente a **SHEIN** para este análisis.

La variable objetivo (`sentimiento`) se derivó de forma directa y verificable a partir del puntaje en estrellas que cada usuario asignó a su reseña (1-2 = negativo, 3 = neutral, 4-5 = positivo), en lugar de inferirse con un modelo externo.

## Metodología

1. **EDA y Data Wrangling**: exploración del dataset, conversión de tipos, filtrado de valores fuera de rango, eliminación de columnas irrelevantes, tratamiento de nulos.
2. **Preprocesamiento de NLP**: limpieza de texto (URLs, caracteres especiales), eliminación de stopwords y lematización con spaCy.
3. **Vectorización**: 
   - Tokenización + padding para la red neuronal.
   - TF-IDF para los modelos clásicos.
4. **Modelado**:
   - Red neuronal (Embedding + capas densas), con balanceo de clases (`class_weight`).
   - Regresión Logística y Random Forest como modelos de comparación.
5. **Evaluación**: accuracy, classification report, matriz de confusión, curvas de accuracy/loss.
6. **Análisis complementario en NLP**: WordClouds por sentimiento, palabras más determinantes por coeficiente del modelo lineal, y relación entre longitud de reseña y sentimiento.

## Resultados

| Modelo | Accuracy |
|---|---|
| Regresión Logística | 89% |
| Random Forest | 87% |
| Red Neuronal | 82% |

**Hallazgo principal:** los modelos clásicos superaron a la red neuronal. Esto se explica principalmente por el fuerte desbalance de clases (la categoría "neutral" está muy subrepresentada en el dataset) y porque la red tuvo que aprender el significado de las palabras desde cero, mientras que TF-IDF aprovecha directamente la frecuencia de términos discriminantes. La clase neutral fue, para todos los modelos, la más difícil de predecir.

## Limitaciones identificadas

- La lista de stopwords por defecto de spaCy elimina negadores ("not", "no"), lo que puede invertir el sentido de frases cortas como "not good".
- La clase neutral, al tener pocos ejemplos, limita el aprendizaje de todos los modelos por igual.

## Tecnologías utilizadas

Python · pandas · scikit-learn · TensorFlow/Keras · spaCy · seaborn/matplotlib · WordCloud


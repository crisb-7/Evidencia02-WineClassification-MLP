# Portafolio de Implementación TC3006C

Uso de framework o biblioteca de aprendizaje máquina para la implementación de una solución.

Por Cristofer Becerra Sánchez
## Datos Generales

- **Dataset**: [Vinos](https://raw.githubusercontent.com/crisb-7/WineClassification-MLP/main/wine.data)
- **Biblioteca**: Scikit-learn
- **Modelo**: Multi-layer Perceptron Classifier
- **Número de características**: 13

## Desempeño del modelo

### Conjunto de Entrenamiento

- Exactitud promedio: 99%
- Precisión promedio: 99%
- Sensibilidad (Recall) promedio: 99%
- F1 promedio: 99%

### Conjunto de Validación

- Exactitud promedio: 100%
- Precisión promedio: 100%
- Sensibilidad (Recall) promedio: 100%
- F1 promedio: 100%

### Conjunto de Prueba
- Exactitud promedio: 100%
- Precisión promedio: 100%
- Sensibilidad (Recall) promedio: 100%
- F1 promedio: 100%

### Predicciones

Valores de entrada:

| Alcohol	| MalicAcid | Ash | AlcalinityOfAsh | Magnesium |	TotalPhenols |	Flavanoids |	NonflavanoidPhenols |	Proanthocyanins |	ColorIntensity |	Hue |	OD280/OD315 | Proline |
| ------- | --------- | --- | --------------- | --------- | ------------ | ----------- | -------------------- | --------------- | -------------- | ---- | ----------- | ------- |
| 0.531579 |	0.616601 |	0.513369 |	0.613402 |	0.163043 |	0.231034 |	0.263713 |	0.905660 |	0.381703 |	0.300341 |	0.292683 |	0.271062 |	0.169044 |
| 0.365789 |	0.357708 |	0.486631 |	0.587629 |	0.217391 |	0.241379 |	0.316456 |	1.000000 |	0.318612 |	0.121160 |	0.308943 |	0.743590 |	0.026391 |
| 0.500000 |	0.604743 |	0.689840 |	0.412371 |	0.347826 |	0.493103 |	0.436709 |	0.226415 |	0.495268 |	0.274744 |	0.447154 |	0.824176 |	0.350927 |
| 0.160526 |	0.260870 |	0.588235 | 0.567010	| 0.152174 |	0.334483 |	0.284810 |	0.660377 |	0.296530 |	0.129693 |	0.422764 |	0.542125 | 0.286733 |
| 0.444737 |	0.199605 |	0.491979 |	0.613402 |	0.152174 |	0.137931 |	0.299578 |	0.660377 |	0.384858 |	0.172355 |	0.325203 |	0.421245 |	0.149786 |

Valor de salida esperado (real) contra predicción del modelo:

| Real | Predicción |
| ---- | ---------- |
|	1 |	1 |
| 1 |	1 |
| 0 |	0 |
|	1 |	1 |
| 1 |	1 |

## Archivos del repositorio

- `PortafolioImplementacion-E2-MLPC.ipynb` es el proyecto de Jupyter Notebook donde se desarrolló la implementación del modelo.
- `PortafolioImplementacion-E2-MLPC.py` es el archivo .py exportado del proyecto de Jupyter Notebook.
- `PortafolioImplementacion-MLPC.pdf` es el documento PDF exportado del proyecto de Jupyter Notebook.
- `wine.data` es el archivo con los datos utilizados (se puede utilizar el permalink de este archivo o el link proporcionado en la sección de datos generales).

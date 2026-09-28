# Fundamentos de Programación para la Inteligencia Artificial

Curso práctico de **aprendizaje profundo con PyTorch** en seis cuadernos Jupyter. Cada cuaderno se apoya en el anterior e introduce técnicas y conceptos progresivamente más avanzados: desde un perceptrón multicapa que ajusta funciones en una dimensión hasta un autoencoder variacional capaz de generar imágenes nuevas.

Además de los modelos, a lo largo del curso se va construyendo, pieza a pieza, un **pipeline completo de *machine learning***: partición de datos, preprocesado sin fugas de información, entrenamiento con validación, selección de hiperparámetros, evaluación con métricas adecuadas y persistencia del modelo.

> Trabajo realizado en la asignatura *Programación para la Inteligencia Artificial* (3.º curso de grado). Autor: **William Lu Bjornestad**.

---

## Contenido

| # | Cuaderno | Problema | Técnicas principales |
|---|---|---|---|
| 01 | [Regresión multisalida con un MLP](notebooks/01_regresion_multisalida_mlp.ipynb) | Ajustar tres funciones $f_1, f_2, f_3$ en $[-120, 120]$ a partir de datos con ruido | MLP, `TensorDataset` / `DataLoader`, bucle train/val, estandarización |
| 02 | [Representación neuronal de una imagen](notebooks/02_representacion_neuronal_imagen.ipynb) | Aprender una imagen como función $(x, y) \to (R, G, B)$ y generarla a mayor resolución | `Dataset` propio, codificación posicional, sesgo espectral, aliasing |
| 03 | [Clasificación en CIFAR-10: CPU frente a GPU](notebooks/03_clasificacion_cifar10_cpu_gpu.ipynb) | Clasificar 10 tipos de objeto y medir la aceleración por hardware | *Early stopping*, benchmark de carga de datos, matriz de confusión, CNN |
| 04 | [Clasificación jerárquica en CIFAR-100](notebooks/04_clasificacion_jerarquica_cifar100.ipynb) | Combinar un clasificador de 100 clases y otro de 20 superclases | Probabilidad condicionada, composición de modelos, métricas top-5 y jerárquicas |
| 05 | [Autoencoder de eliminación de ruido](notebooks/05_autoencoder_eliminacion_ruido.ipynb) | Eliminar ruido gaussiano de imágenes de Fashion-MNIST | DAE, estimador MMSE, fórmula de Tweedie, PCA, t-SNE, KNN sobre el espacio latente |
| 06 | [Autoencoder variacional](notebooks/06_autoencoder_variacional.ipynb) | Generar prendas nuevas muestreando el espacio latente | VAE, ELBO, truco de reparametrización, compromiso β, interpolación |

## Progresión del curso

El curso avanza en tres ejes a la vez: el **tipo de problema**, la **arquitectura** y la **madurez del pipeline**.

```
Regresión 1D  ──►  Regresión 2D  ──►  Clasificación  ──►  Clasificación   ──►  Representación  ──►  Generación
(01)               (02)               (03)                jerárquica (04)      no supervisada (05)  (06)

MLP pequeño   ──►  MLP + codific.  ──►  MLP y CNN     ──►  CNN × 2          ──►  Autoencoder     ──►  Autoencoder
                   posicional                                                                        variacional
```

### Conceptos de pipeline de ML por cuaderno

Cada cuaderno incorpora nuevas etapas de pipeline sobre las anteriores:

| Cuaderno | Etapas de pipeline introducidas |
|---|---|
| 01 | Reproducibilidad con semillas · partición train/val/test · estandarización ajustada solo con train (evitar *data leakage*) · *checkpoint* del mejor modelo · evaluación final en unidades originales |
| 02 | `Dataset` personalizado · ingeniería de características (codificación posicional) · validación sobre datos no vistos · barrido de hiperparámetros con registro de resultados · reentrenamiento final con todos los datos |
| 03 | `transforms.Compose` y normalización por canal · código independiente del dispositivo · precálculo y precarga de datos en GPU · *early stopping* con restauración de pesos · informe de clasificación por clase · guardar el modelo junto con su preprocesado |
| 04 | Lectura del formato original de un dataset · partición compartida entre modelos · reutilización de componentes · composición probabilística de modelos · métricas adaptadas al problema (top-5, gravedad del error) |
| 05 | Corrupción dinámica de datos (aumento en cada lote) · evaluación determinista · líneas base · el codificador como extractor de características · `sklearn.pipeline.Pipeline` |
| 06 | Pérdidas compuestas monitorizadas por término · barrido de un hiperparámetro crítico · inferencia determinista · diagnóstico del modelo (unidades activas) · generación desde un *checkpoint* |

## Qué se aprende

**Fundamentos de PyTorch**
- Tensores, `nn.Module`, `nn.Sequential`, `Dataset` y `DataLoader`.
- Bucles de entrenamiento y validación escritos a mano: `model.train()` / `model.eval()`, `torch.no_grad()`, `zero_grad → backward → step`.
- Código que funciona igual en CPU y GPU, y cómo alimentar bien una GPU.

**Entrenamiento y generalización**
- Interpretar curvas de pérdida: suelo de ruido, sobreajuste, inestabilidad numérica.
- Normalización de entradas y salidas, optimizadores (SGD frente a Adam), *early stopping*.
- El compromiso sesgo-varianza visto desde el preprocesado (número de frecuencias de la codificación posicional) y no solo desde el tamaño del modelo.

**Arquitecturas**
- Perceptrones multicapa, redes convolucionales y el concepto de **sesgo inductivo**.
- Codificación posicional y sesgo espectral de las redes neuronales.
- Autoencoders, autoencoders de eliminación de ruido y autoencoders variacionales.

**Probabilidad aplicada**
- Clasificación como estimación de probabilidades; regla de la cadena para combinar modelos jerárquicos.
- Estimador MMSE, fórmula de Tweedie y función de puntuación (base de los modelos de difusión).
- Inferencia variacional, ELBO, divergencia KL y truco de reparametrización.

**Evaluación y análisis**
- Matrices de confusión, precisión / recall / F1, exactitud top-5, PSNR.
- Visualización de espacios latentes con PCA y t-SNE, y evaluación cuantitativa con KNN.
- Benchmarks justos: cambiar un solo factor cada vez.

## Tareas realizadas

- [x] **01** · MLP que ajusta simultáneamente tres funciones con entrenamiento ruidoso ($\sigma = 5$) y validación/test de 500 ejemplos. Diagnóstico de las curvas de error, pipeline estandarizado, análisis del error por región y de la extrapolación.
- [x] **02** · Reconstrucción de una imagen de $35 \times 30$ y generación a $70 \times 60$ con un MLP y codificación posicional. Diagnóstico de los artefactos como aliasing, barrido del número de frecuencias y reentrenamiento final.
- [x] **03** · Clasificador de CIFAR-10 con *early stopping*. Benchmark de CPU y GPU con cinco configuraciones de carga de datos, análisis de errores por clase y comparación MLP frente a CNN.
- [x] **04** · Clasificadores fino (100 clases) y grueso (20 superclases) en CIFAR-100. Cuatro estrategias de combinación, métricas jerárquicas y análisis de la información que aporta el modelo grueso.
- [x] **05** · DAE con espacio latente de 8 dimensiones. Fundamento bayesiano, comparación con un autoencoder estándar, visualización del espacio latente y evaluación de la representación con KNN.
- [x] **06** · VAE con espacio latente de 8 dimensiones. Barrido de β, unidades activas, generación de imágenes nuevas, interpolación entre clases y generación desde un *checkpoint*.

## Resultados destacados

| # | Resultado | Lección |
|---|---|---|
| 01 | Error medio absoluto en test: **3.35 → 0.56** con la misma arquitectura | Estandarizar entradas y salidas importa tanto como el modelo. La validación por debajo del entrenamiento se explica por el suelo de ruido $\sigma\sqrt{2/\pi}$, no por sobreajuste |
| 02 | PSNR a $70 \times 60$ frente a la imagen real: **13.6 dB → 21.6 dB** | Los artefactos a doble resolución eran *aliasing* (frecuencias por encima del límite de Nyquist). La validación sobre la misma rejilla no lo detectaba |
| 03 | **≈27× más rápido** por época · MLP **49.7 %** frente a CNN **74.1 %** con un 38 % menos de parámetros | Gran parte de la "ventaja de la GPU" era el pipeline de datos. El sesgo inductivo rinde más que la capacidad |
| 04 | Top-1 fino: **18.3 %** (versión inicial, lineal) → **40.4 %** (CNN) → **42.2 %** combinando con el modelo grueso | El producto funciona como *ensemble*. La combinación "exacta" empeora porque el modelo grueso predice superclases peor que el fino agregado |
| 05 | PSNR con ruido: **15.9 dB → 18.3 dB** · KNN sobre 8 dimensiones latentes: **83.1 %** (píxeles: 85.1 %, PCA-8: 78.3 %) | Representaciones útiles sin etiquetas. El DAE rinde peor con entradas limpias, que para él están fuera de distribución |
| 06 | Con β = 1, casi todas las muestras del prior son prendas reconocibles, con 5 de 8 dimensiones activas | β regula el compromiso entre reconstruir y generar: la configuración original (β = 0.01) reconstruía bien pero generaba mal |

## Ejecución

Los cuadernos están ejecutados y guardados con sus resultados, así que se pueden leer directamente en GitHub. Para reproducirlos:

```bash
git clone <url-del-repositorio>
cd ai-programming-course
python -m venv .venv
source .venv/bin/activate        # En Windows: .venv\Scripts\activate
pip install -r requirements.txt
jupyter notebook notebooks/
```

Para usar GPU, instala la versión de PyTorch con CUDA adecuada a tu equipo siguiendo las instrucciones de [pytorch.org](https://pytorch.org/get-started/locally/). Todos los cuadernos funcionan también en CPU, aunque los que usan CIFAR y Fashion-MNIST tardan bastante más.

- Los conjuntos de datos (CIFAR-10, CIFAR-100 y Fashion-MNIST) se descargan automáticamente en `data/` la primera vez.
- Los modelos guardados se escriben en `checkpoints/`. Ambas carpetas están excluidas del control de versiones.
- Los cuadernos deben ejecutarse desde la carpeta `notebooks/`, ya que usan rutas relativas (`../data`, `../assets`).
- Todos los experimentos usan semillas fijas. Aun así, en GPU pueden aparecer pequeñas variaciones numéricas entre ejecuciones.

Los resultados guardados se obtuvieron con PyTorch 2.5 y una GPU NVIDIA GeForce RTX 2050 (4 GB). Los tiempos del cuaderno 03 dependen del hardware, pero la comparación relativa entre configuraciones es general.

## Estructura del repositorio

```
ai-programming-course/
├── README.md
├── requirements.txt
├── assets/
│   └── shin_chan_progIA.png      # Imagen de referencia del cuaderno 02
├── data/                         # (generado) datasets descargados, ignorado por git
├── checkpoints/                  # (generado) modelos guardados, ignorado por git
└── notebooks/
    ├── 01_regresion_multisalida_mlp.ipynb
    ├── 02_representacion_neuronal_imagen.ipynb
    ├── 03_clasificacion_cifar10_cpu_gpu.ipynb
    ├── 04_clasificacion_jerarquica_cifar100.ipynb
    ├── 05_autoencoder_eliminacion_ruido.ipynb
    └── 06_autoencoder_variacional.ipynb
```

## Tecnologías

Python 3.11 · PyTorch · torchvision · scikit-learn · NumPy · pandas · Matplotlib · Jupyter

# Benchmark Cuantitativo QCNN vs. Baselines Clásicos (MNIST Binario)

Pipeline experimental modular y reproducible  para evaluar **Redes Neuronales Convolucionales Cuánticas (QCNN)** frente a modelos clásicos bajo **igualdad estricta de parámetros entrenables ($T$)** y distintos regímenes de escasez de datos ($N_{\text{train}}$).

---

## 1. Propósito del Proyecto

En *Quantum Machine Learning (QML)* sobre dispositivos NISQ, comparar un circuito cuántico pequeño contra una red clásica de decenas de miles de parámetros confunde el efecto del ansatz cuántico con la diferencia de capacidad. Este framework evalúa de forma controlada:

1. **Eficiencia Muestral ($N_{\text{train}} \in \{10, 20, 50, 100, 200, 500\}$)**: Desempeño cuando los datos etiquetados son escasos.
2. **Eficiencia Paramétrica ($T \in \{15, 30, 45, 90\}$)**: Exactitud y densidad de representación ($\text{Accuracy} / T$) bajo presupuestos idénticos de parámetros entrenables.
3. **Complejidad Geométrica de la Tarea**:
   - **`mnist_01` (Dígitos 0 vs. 1)**: Alta separabilidad lineal en las primeras componentes principales.
   - **`mnist_38` (Dígitos 3 vs. 8)**: Alto solape morfológico no lineal (escenario de estrés).

---

## 2. Principios Metodológicos Garantizados

- **Cero Fuga de Datos (*Zero Data Leakage*)**: En cada corrida `(tarea, N_train, semilla)`, la reducción dimensional (`PCA` a 8 componentes) y el escalado (`StandardScaler` / `MinMaxScaler` y proyección angular $[0, \pi]$) se ajustan (`fit`) **exclusivamente sobre `X_train`** y luego se aplican (`transform`) sobre `X_val` y `X_test`.
- **Partición Estática Inmutable**: Antes de cualquier entrenamiento, para cada tarea se construye y guarda en `data/splits/` un conjunto de prueba balanceado fijo ($N_{\text{test}} = 1000$, 500 por clase) y un conjunto de validación fijo ($N_{\text{val}} = 200$, 100 por clase).
- **Selección de Modelo en Validación**: Durante el entrenamiento de los modelos PyTorch/PennyLane, el mejor checkpoint se selecciona evaluando únicamente en `X_val`. El conjunto `X_test` se evalúa una sola vez al finalizar.
- **Paridad Exacta de Parámetros ($T$)**: Para cada presupuesto $T \in \{15, 30, 45, 90\}$, los tres modelos variacionales (`QCNN`, `CNN_Matched`, `MLP_Compact`) tienen exactamente el mismo número de parámetros entrenables, verificado en tiempo de ejecución mediante aserciones (`assert`).
- **Rigor Estadístico**: Reporte de media, desviación estándar, intervalos de confianza del 95% (distribución $t$ de Student) y pruebas no paramétricas pareadas por semilla (**Wilcoxon signed-rank**) con tamaño del efecto ($d_z$ de Cohen) y corrección de **Holm-Bonferroni**.

---

## 3. Estructura del Proyecto

```text
qcnn_benchmark/
├── config.yaml                  # Configuración central única (modo, tareas, N_train, T, semillas, modelos)
├── main.py                      # Único punto de entrada CLI para entrenamiento, evaluación y reportes
├── requirements.txt             # Dependencias del proyecto (PennyLane, PyTorch, Scikit-Learn, etc.)
├── README.md                    # Documentación del proyecto
├── Context/
│   └── plan_investigacion_qcnn_vs_modelos_clasicos.md
├── data/
│   ├── MNIST/                   # Dataset MNIST crudo descargado automáticamente por torchvision
│   └── splits/                  # Particiones estáticas inmutables (.npz) por tarea
├── results/
│   └── <modo_o_run_name>/       # CSVs crudos, tablas estadísticas, snapshots y gráficos a 600 DPI
└── src/
    ├── config.py                # Validación del YAML y construcción del plan experimental (ExperimentPlan)
    ├── data/
    │   ├── loader.py            # Creación/carga de particiones estáticas y submuestreo balanceado
    │   └── preprocessing.py     # Pipeline PCA + Scaler sin fuga de datos y vistas por modelo
    ├── models/
    │   ├── qcnn.py              # Circuito QCNN en PennyLane (8 qubits, T ∈ {15, 30, 45, 90})
    │   ├── cnn_matched.py       # CNN 1D en PyTorch igualada exactamente a T ∈ {15, 30, 45, 90}
    │   ├── cnn_full.py          # CNN 2D clásica sobre imágenes 28x28 (T = 26,449, techo de referencia)
    │   ├── baselines.py         # MLP_Compact (igualado a T), Regresión Logística y SVM (Lineal y RBF)
    │   └── registry.py          # Constructor centralizado de modelos y mapeo de vistas de entrada
    ├── training/
    │   ├── trainer.py           # Loop de entrenamiento PyTorch unificado con BCEWithLogitsLoss
    │   └── runner.py            # Ejecutor de un pliegue (tarea, N_train, semilla) sin corridas redundantes
    └── utils/
        ├── reproducibility.py   # Control determinista de semillas e información del entorno
        ├── metrics.py           # Cálculo de Accuracy, Balanced Accuracy, F1, Precision, Recall y Loss
        ├── logger.py            # Guardado incremental en CSV con soporte de reanudación (resume)
        ├── statistics.py        # IC 95%, matriz de puntos de cruce y test pareado de Wilcoxon
        └── visualizer.py        # Generación de 6 figuras comparativas a 600 DPI
```

---

## 4. Modelos Evaluados y Arquitecturas

| Modelo | Tipo | Entrada | Presupuestos $T$ Soportados | Descripción |
| :--- | :--- | :--- | :---: | :--- |
| **`QCNN`** | Cuántico Variacional (PennyLane) | 8 ángulos $\in [0, \pi]$ (PCA 8D) | `15, 30, 45, 90` | Circuito de 8 qubits inspirado en Cong et al. (2019): *Angle Encoding* + 3 etapas de convolución/pooling cuántico reduciendo $8 \to 4 \to 2 \to 1$ qubit y medición $\langle Z_0 \rangle$. |
| **`CNN_Matched`** | Clásico Igualado (PyTorch) | Vector PCA 8D $(1 \times 8)$ | `15, 30, 45, 90` | CNN 1D + AdaptiveAvgPool1d + cabezal lineal diseñada algebraicamente para coincidir de forma exacta con el presupuesto $T$ de la QCNN. |
| **`MLP_Compact`** | Clásico Igualado (PyTorch) | Vector PCA 8D | `15, 30, 45, 90` | Perceptrón multicapa con cuello de botella lineal o activación `tanh` ajustado exactamente a $T$ parámetros. |
| **`Logistic_Regression`** | Baseline Clásico (Scikit-Learn) | Vector PCA 8D | Fijo ($T = 9$) | Clasificador lineal regularizado ($L_2$, 8 pesos + 1 sesgo). |
| **`SVM_Linear`** | Baseline Clásico (Scikit-Learn) | Vector PCA 8D | Fijo ($T = 9$) | Máquina de vectores de soporte con hiperplano lineal. |
| **`SVM_RBF`** | Baseline Clásico (Scikit-Learn) | Vector PCA 8D | $9 + n_{\text{SV}}$ | SVM con kernel Gaussiano RBF (clásico no lineal fuerte en pocos datos). |
| **`CNN_Full`** | Techo Clásico (PyTorch) | Imagen $1 \times 28 \times 28$ | Fijo ($T = 26,449$) | CNN 2D de dos bloques convolucionales sobre la imagen completa sin reducción PCA. |

### Cómo se logran los presupuestos exactos $T \in \{15, 30, 45, 90\}$

- **En `QCNN` (`src/models/qcnn.py`)**:
  - $T = 15$: Convolución reducida de 3 parámetros por capa ($3 \times 3 = 9$) + pooling controlado de 2 parámetros por capa ($3 \times 2 = 6$).
  - $T = 30$: Convolución estándar de 9 parámetros por capa ($3 \times 9 = 27$) + pooling reducido de 1 parámetro por capa ($3 \times 1 = 3$).
  - $T = 45$: Ansatz completo de Cong et al. con bloques $SU(4)$ de 12 parámetros por capa convolucional ($3 \times 12 = 36$) + pooling de 3 parámetros por capa ($3 \times 3 = 9$).
  - $T = 90$: Ansatz profundo (*2-brick deep*) con 2 bloques $SU(4)$ por capa convolucional ($3 \times 24 = 72$) + pooling extendido de 6 parámetros por capa ($3 \times 6 = 18$).

---

## 5. Instalación y Requisitos

1. **Crear y activar el entorno virtual** (en Windows PowerShell):
   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

2. **Instalar dependencias**:
   ```powershell
   pip install -r requirements.txt
   ```

---

## 6. Uso y Modos de Ejecución (`main.py` y `config.yaml`)

El proyecto cuenta con **un único archivo ejecutable (`main.py`)**. Todo el diseño experimental se controla editando `config.yaml` o pasando argumentos por línea de comandos.

### 6.1 Configuración en `config.yaml`

En la sección `run:` de `config.yaml` se define qué experimento correr:

```yaml
run:
  mode: pilot                     # pilot | data_size | param_budget | grid
  tasks: [mnist_01, mnist_38]     # Puedes poner [mnist_01], [mnist_38] o ambas [mnist_01, mnist_38]
  models: [QCNN, CNN_Matched, MLP_Compact, Logistic_Regression, SVM_Linear, SVM_RBF, CNN_Full]
  results_dir: ./results
  run_name: null                  # null usa el nombre del modo (ej. results/pilot)
  resume: true                    # Si se interrumpe, reanuda sin repetir entrenamientos ya guardados
  n_jobs: 1                       # Procesos en paralelo (-1 usa todos los núcleos de CPU)
```

### 6.2 Los 4 Modos Experimentales

| Modo (`run.mode`) | Qué evalúa | Valores por defecto en `config.yaml` |
| :--- | :--- | :--- |
| **`pilot`** | Prueba rápida de extremo a extremo validando tamaños, presupuestos y gráficos. | $N_{\text{train}} \in \{20, 50\}$, $T \in \{15, 45\}$, 3 semillas, `epochs_scale: 0.5` |
| **`data_size`** | **Barrido de cantidad de imágenes ($N_{\text{train}}$)** con presupuesto $T$ fijo (curvas de aprendizaje). | $N_{\text{train}} \in \{10, 20, 50, 100, 200, 500\}$, $T = 45$, 15 semillas |
| **`param_budget`** | **Barrido de presupuesto de parámetros ($T$)** con cantidad de imágenes $N_{\text{train}}$ fija. | $N_{\text{train}} = 50$, $T \in \{15, 30, 45, 90\}$, 15 semillas |
| **`grid`** | **Barrido cruzado completo ($N_{\text{train}} \times T$)** para identificar todos los puntos de cruce. | $N_{\text{train}} \in \{10, 20, 50, 100, 200, 500\} \times T \in \{15, 30, 45, 90\}$, 15 semillas |

### 6.3 Comandos de Ejecución

- **Ejecutar el modo configurado actualmente en `config.yaml`**:
  ```powershell
  .\.venv\Scripts\python.exe main.py
  ```

- **Sobrescribir el modo desde la terminal sin editar `config.yaml`**:
  ```powershell
  .\.venv\Scripts\python.exe main.py --mode pilot
  .\.venv\Scripts\python.exe main.py --mode data_size
  .\.venv\Scripts\python.exe main.py --mode param_budget
  .\.venv\Scripts\python.exe main.py --mode grid
  ```

- **Usar un archivo de configuración alternativo**:
  ```powershell
  .\.venv\Scripts\python.exe main.py --config otro_config.yaml
  ```

- **Regenerar únicamente las tablas estadísticas y gráficos a 600 DPI** (a partir de resultados ya calculados, sin volver a entrenar):
  ```powershell
  .\.venv\Scripts\python.exe main.py --report-only results/pilot
  ```

---

## 7. Flujo Interno de Ejecución (Paso a Paso)

Cuando ejecutas `python main.py`, el sistema realiza automáticamente las siguientes etapas:

1. **Lectura y Validación (`src/config.py`)**: Valida tareas, modelos, semillas, tamaños $N_{\text{train}}$ y presupuestos $T$, construyendo el objeto `ExperimentPlan`. Guarda una copia exacta en `results/<modo>/config_snapshot.yaml` y las versiones de librerías/hardware en `environment.json`.
2. **Partición Estática (`src/data/loader.py`)**: Verifica si existen `data/splits/mnist_01_static_test.npz` y/o `mnist_38_static_test.npz`. Si no existen, descarga MNIST, filtra las clases correspondientes y separa de forma determinista el conjunto de prueba ($1000$ muestras), validación ($200$ muestras) y el pool de entrenamiento restante.
3. **Ejecución por Pliegue (`src/training/runner.py`)**:
   - Para cada combinación `(tarea, N_train, semilla)`:
     - Extrae un subconjunto balanceado de tamaño `N_train` ($N_{\text{train}}/2$ por clase).
     - Ajusta `PreprocessingPipeline` (`PCA(8)` + `StandardScaler` + mapeo angular $[0, \pi]$) **únicamente con `X_train`**.
     - Entrena los modelos de capacidad fija (`Logistic_Regression`, `SVM_Linear`, `SVM_RBF`, `CNN_Full`) **una sola vez** para ese pliegue.
     - Recorre cada presupuesto $T$ configurado y entrena los modelos variacionales (`QCNN`, `CNN_Matched`, `MLP_Compact`) verificando que `count_trainable_params(model) == T`.
     - Registra métricas en validación y prueba, tiempos de entrenamiento/inferencia y recursos del circuito cuántico (`n_qubits`, `circuit_depth`, `n_gates`).
     - Escribe cada resultado inmediatamente en `raw_results.csv` (`ResultStore`), permitiendo pausar y reanudar (`resume: true`) sin perder progreso.
4. **Análisis Estadístico (`src/utils/statistics.py`)**:
   - Calcula medias, desviaciones estándar e intervalos de confianza del 95% (`summary_ci95.csv`).
   - Construye la matriz de diferencias $\Delta\text{Accuracy}$ entre `QCNN`, `CNN_Matched` y `MLP_Compact` (`crossover_grid.csv`).
   - Ejecuta pruebas pareadas de Wilcoxon con corrección de Holm-Bonferroni (`statistical_tests_wilcoxon.csv`).
5. **Visualización Científica (`src/utils/visualizer.py`)**: Genera 6 gráficos en formato PNG a **600 DPI** listos para publicación.

---

## 8. Artefactos y Gráficos Generados (`results/<modo>/`)

Cada corrida produce en su carpeta de salida:

### Archivos de Datos y Auditoría
- `raw_results.csv`: Registro granular de cada semilla, modelo, tarea, $N_{\text{train}}$ y $T$ (27 columnas).
- `summary_ci95.csv`: Tabla agregada con media, desviación estándar, error estándar, IC 95%, exactitud por parámetro ($\text{Acc}/T$) y tiempo medio.
- `crossover_grid.csv`: Comparativa directa de capacidad igualada ($\Delta\text{Acc}_{\text{QCNN}-\text{CNN}}$ y $\Delta\text{Acc}_{\text{QCNN}-\text{MLP}}$).
- `statistical_tests_wilcoxon.csv`: Valor $p$ crudo, valor $p$ ajustado por Holm-Bonferroni, tamaño de efecto $d_z$ y modelo ganador.
- `config_snapshot.yaml` y `environment.json`: Trazabilidad completa del experimento.

### Gráficos a 600 DPI
1. **`01_learning_curves_by_task_600dpi.png`**: Curvas de aprendizaje ($\text{Test Accuracy vs. } N_{\text{train}}$) comparando todos los modelos y presupuestos $T$ con bandas de IC 95%.
2. **`02_parameter_efficiency_by_task_600dpi.png`**: Exactitud vs. número de parámetros entrenables (escala logarítmica), mostrando todas las corridas individuales, medias $\pm$ DE y líneas de tendencia a través de $T$.
3. **`03_accuracy_per_parameter_by_task_600dpi.png`**: Panel doble que muestra arriba la **Exactitud absoluta ($\uparrow$)** al crecer $T$ y abajo la **Densidad de exactitud por parámetro ($\text{Acc}/T$)**, ilustrando la ley de rendimientos marginales decrecientes por parámetro.
4. **`04_joint_grid_qcnn_vs_cnn_600dpi.png`**: Mapas de calor 2D ($N_{\text{train}} \times T$) con la ventaja neta $\Delta\text{Accuracy}$ (%) de la `QCNN` frente a `CNN_Matched` y `MLP_Compact`.
5. **`05_paired_seed_trajectories_by_task_600dpi.png`**: Gráfico de coordenadas paralelas conectando el rendimiento de la misma semilla a través de cada familia de modelos.
6. **`06_master_dashboard_600dpi.png`**: Panel ejecutivo multipanel con curvas de aprendizaje, escalado paramétrico, distribución por semilla y tiempo de cómputo vs. exactitud.


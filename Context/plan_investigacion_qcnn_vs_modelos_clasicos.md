# Plan de investigación: evaluación de QCNN frente a modelos clásicos

## Objetivo general

Diseñar y ejecutar un estudio experimental reproducible para evaluar en qué condiciones una red neuronal convolucional cuántica (QCNN) puede ofrecer ventajas frente a modelos clásicos comparables, considerando el desempeño predictivo, el presupuesto de parámetros, la cantidad de datos y el costo computacional.

---

## Fase A. Definición del problema

1. **Fijar la pregunta de investigación y la hipótesis.**
   - Pregunta de ejemplo: *¿Hasta qué presupuesto de parámetros y tamaño de entrenamiento la QCNN supera en accuracy por parámetro a una CNN de la misma arquitectura?*
   - Definir antes de experimentar qué resultados confirmarían la hipótesis y cuáles la refutarían.

2. **Definir qué se considera un parámetro.**
   - Especificar si el conteo incluye únicamente parámetros entrenables.
   - Reportar por separado qubits, profundidad del circuito y número de compuertas.
   - Mantener esta definición consistente en todos los modelos y experimentos.

3. **Definir las variables del estudio.**
   - **Variables independientes:** presupuesto de parámetros, tamaño del conjunto de entrenamiento, número de qubits y profundidad.
   - **Variables dependientes:** accuracy, balanced accuracy (cuando exista desbalance), tiempo de entrenamiento y consumo de memoria.

## Fase B. Datos

4. **Seleccionar los datasets y las tareas de clasificación binaria.**
   - **Validación inicial:** MNIST, con las tareas 0 vs. 1 y una tarea más difícil, como 3 vs. 8.
   - **Comparabilidad con la literatura:** BreastMNIST.
   - **Caso de aplicación realista:** imágenes de tumores cerebrales, con las tareas glioma vs. meningioma y tumor vs. no tumor.

5. **Auditar los datos de tumores cerebrales.**
   - Buscar imágenes duplicadas o casi duplicadas.
   - Identificar posibles diferencias de fuente, formato o adquisición entre clases.
   - Verificar, cuando la información esté disponible, que imágenes de un mismo paciente no queden repartidas entre entrenamiento, validación y prueba.
   - Documentar los hallazgos y las decisiones tomadas.

6. **Balancear las clases mediante submuestreo.**
   - Construir conjuntos balanceados para que la accuracy sea interpretable.
   - Documentar cuántas muestras se conservan y descartan por clase.

7. **Particionar los datos antes de cualquier transformación o aumento.**
   - Crear conjuntos de entrenamiento, validación y prueba con una semilla fija.
   - Evitar que el conjunto de prueba intervenga en el ajuste de modelos, hiperparámetros o transformaciones.
   - Mantener el conjunto de prueba aislado hasta la evaluación final.

8. **Crear subconjuntos de entrenamiento para estudiar el régimen de pocos datos.**
   - Usar tamaños de referencia de 10, 20, 50, 100 y 200 muestras, siempre que el dataset lo permita.
   - Mantener fijo y completo el conjunto de prueba.
   - Definir claramente si los tamaños indicados son totales o por clase; se recomienda reportar el número por clase y el total.

## Fase C. Preprocesamiento común

9. **Definir un único pipeline de reducción de datos.**
   - Considerar redimensionamiento, normalización, PCA o downsampling hasta obtener 8 o 16 features.
   - Aplicar exactamente el mismo pipeline a la QCNN y a los baselines compactos.
   - Ajustar PCA exclusivamente con los datos de entrenamiento de cada partición; después, aplicar la transformación aprendida a validación y prueba.

10. **Definir la entrada de cada modelo.**
    - **QCNN y baselines compactos:** utilizar las features reducidas.
    - **CNN de resolución completa:** utilizar las imágenes originales como referencia de desempeño o techo comparativo.
    - Dejar explícito que la CNN completa no participa necesariamente en la comparación de presupuesto igualado si su entrada y capacidad son distintas.

11. **Elegir y justificar la codificación cuántica.**
    - Seleccionar una codificación por ángulos o por amplitud.
    - Justificar la elección según el número de features, los recursos cuánticos y el costo de preparación del estado.
    - Si el tiempo lo permite, comparar ambas codificaciones como experimento de ablación.

## Fase D. Modelos

12. **Definir la arquitectura QCNN.**
    - Utilizar un ansatz inspirado en Cong et al., con operaciones de convolución y pooling.
    - Evaluar configuraciones de 8 o 16 qubits.
    - Fijar optimizador, función de pérdida, tasa de aprendizaje y demás decisiones de entrenamiento.

13. **Definir los baselines clásicos.**
    - CNN con presupuesto de parámetros igualado.
    - CNN completa como referencia de desempeño.
    - MLP pequeño.
    - Regresión logística.
    - Máquina de vectores de soporte (SVM).

14. **Igualar el presupuesto de parámetros entre QCNN y CNN.**
    - Realizar la comparación para cada punto del barrido de presupuestos.
    - Registrar el conteo exacto de parámetros entrenables de cada modelo.
    - Reportar por separado los recursos específicos de cada implementación cuántica y clásica.

15. **Asignar el mismo esfuerzo de ajuste de hiperparámetros a todos los modelos.**
    - Utilizar el mismo número de pruebas por modelo, por ejemplo mediante búsqueda aleatoria o bayesiana.
    - Seleccionar hiperparámetros exclusivamente con el conjunto de validación.
    - Documentar el espacio de búsqueda, el presupuesto de búsqueda y el criterio de selección.
    - Evitar favorecer a un modelo mediante más pruebas o una búsqueda más amplia.

## Fase E. Ejecución experimental

16. **Realizar un experimento piloto pequeño.**
    - Seleccionar un dataset y pocas configuraciones.
    - Ejecutar el piloto con pocas semillas.
    - Verificar que el pipeline completo funciona y estimar el tiempo y los recursos de cómputo necesarios.

17. **Ejecutar el barrido experimental completo.**
    - Utilizar al menos 10 a 20 semillas por configuración, sujeto al presupuesto computacional.
    - Variar tanto la inicialización como el submuestreo del conjunto de entrenamiento.
    - Aplicar las mismas particiones o semillas comparables a los modelos que se contrastan.

18. **Registrar todos los elementos necesarios para reproducibilidad.**
    - Guardar las configuraciones en archivos YAML o JSON.
    - Registrar semillas, versiones de librerías, hardware y entorno de ejecución.
    - Almacenar resultados y métricas en archivos CSV.
    - Conservar los logs y, cuando sea posible, los checkpoints de los modelos.

19. **Evaluar el efecto del ruido como experimento adicional.**
    - Ejecutar simulaciones con un modelo de ruido documentado.
    - Si existe acceso y resulta viable, realizar pruebas complementarias en hardware cuántico real.
    - Separar claramente los resultados ideales, simulados con ruido y obtenidos en hardware.

## Fase F. Análisis de resultados

20. **Generar las curvas principales.**
    - Accuracy vs. número de parámetros.
    - Accuracy vs. tamaño del conjunto de entrenamiento.
    - Incluir una curva por modelo y barras de error o intervalos de confianza.
    - Añadir balanced accuracy cuando corresponda.

21. **Aplicar pruebas estadísticas.**
    - Comparar QCNN y baselines mediante pruebas pareadas sobre las semillas, por ejemplo Wilcoxon signed-rank.
    - Verificar que el emparejamiento de resultados sea válido.
    - Reportar tamaños de efecto e intervalos de confianza, no únicamente valores p.
    - Considerar correcciones por comparaciones múltiples cuando se realicen muchas pruebas.

22. **Identificar el punto de cruce.**
    - Determinar el presupuesto de parámetros o tamaño de datos a partir del cual la ventaja de la QCNN aparece, desaparece o cambia de dirección.
    - Indicar si el punto de cruce es estable entre datasets y tareas.

23. **Comparar los costos computacionales.**
    - Tiempo de entrenamiento e inferencia, cuando sea medible.
    - Consumo de memoria.
    - Número de qubits, profundidad del circuito y número de compuertas.
    - Costo de codificación o preparación de datos.
    - Distinguir los costos de simulación clásica de los costos de ejecución cuántica.

24. **Investigar resultados inesperados o sospechosamente favorables.**
    - Revisar posibles fugas de datos.
    - Verificar las particiones y el preprocesamiento.
    - Confirmar que los baselines estén correctamente implementados y ajustados.
    - Revisar duplicados, diferencias entre fuentes y posibles sesgos del dataset.
    - No interpretar una ventaja grande como evidencia suficiente sin descartar primero estos problemas.

## Fase G. Amenazas a la validez y reporte

25. **Documentar explícitamente las limitaciones del estudio.**
    - Simulabilidad clásica de circuitos con pocos qubits.
    - Problema de barren plateaus y dificultades de optimización.
    - Costo de codificar o preparar los datos.
    - Falta de equivalencia directa entre un parámetro cuántico y uno clásico.
    - Dependencia de los resultados respecto a la compresión y representación de las imágenes.
    - Limitaciones del simulador, del modelo de ruido y del hardware utilizado.

26. **Reportar también los resultados negativos.**
    - Incluir configuraciones y datasets en los que la QCNN no supere a los baselines.
    - Explicar resultados nulos o inconsistentes sin seleccionar únicamente los experimentos favorables.
    - Mantener trazabilidad entre las configuraciones ejecutadas y los resultados reportados.

27. **Evitar afirmaciones clínicas.**
    - Presentar los datasets médicos como tareas de comparación de modelos.
    - No afirmar que el sistema sirve para diagnóstico, toma de decisiones clínicas o uso médico.
    - Aclarar que los resultados experimentales no constituyen una validación clínica.

---

## Entregables esperados

- [ ] Pregunta de investigación, hipótesis y criterios de confirmación/refutación.
- [ ] Inventario y auditoría de datasets.
- [ ] Protocolo de partición y preprocesamiento.
- [ ] Implementación de QCNN y baselines clásicos.
- [ ] Registro de conteos de parámetros y recursos cuánticos.
- [ ] Configuraciones reproducibles y registro de semillas/versiones.
- [ ] Resultados experimentales en CSV.
- [ ] Gráficos comparativos con medidas de incertidumbre.
- [ ] Análisis estadístico y de costos.
- [ ] Discusión de limitaciones, amenazas a la validez y resultados negativos.
- [ ] Informe final con conclusiones proporcionales a la evidencia.

## Criterio general de rigor

La comparación debe ser justa, reproducible y transparente. Las decisiones metodológicas —en particular particiones, reducción de datos, conteo de parámetros y presupuesto de ajuste de hiperparámetros— deben fijarse antes de observar los resultados del conjunto de prueba. La conclusión debe distinguir entre una mejora predictiva, una mejora por parámetro y una posible ventaja computacional: son afirmaciones diferentes y requieren evidencia independiente.

===========================================================================================
  PORTAFOLIO DE MACHINE LEARNING: CLUSTERING JERÁRQUICO, EIGENFACES (PCA)
                 Y SISTEMAS DE RECOMENDACIÓN (NMF)
================================================================================

Python 3.8+  |  scikit-learn 1.0+  |  Licencia MIT

Repositorio que documenta el desarrollo integral de la Actividad 05 del curso de
Machine Learning. Este proyecto implementa, experimenta y evalúa de forma crítica
tres familias de técnicas de aprendizaje no supervisado sobre datasets estándar
de scikit-learn:

  1. Clustering Jerárquico para minería de texto (20 Newsgroups).
  2. Reconocimiento Facial mediante Eigenfaces (PCA) sobre Olivetti Faces.
  3. Sistemas de Recomendación vía Factorización de Matrices No Negativas (NMF).

================================================================================
1. RESUMEN DE RESULTADOS CLAVE (Ejecuciones Reales)
================================================================================

  CASO 1: Clustering Jerárquico + SVD
    Mejor configuración: Linkage 'average', Métrica 'coseno', SVD (200D)
    Métrica principal: ARI (Adjusted Rand Index)
    Resultado: 0.1039

  CASO 2: Eigenfaces (PCA) + SVM
    Mejor configuración: PCA con 50 componentes, SVM kernel RBF (C=10, gamma=0.01)
    Métrica principal: Accuracy en Test
    Resultado: 95.00%

  CASO 3: NMF + Similitud de Coseno
    Mejor configuración: NMF con 20 temas latentes, TF-IDF (5000 palabras)
    Métrica principal: Coherencia Temática (Caso ilustrativo)
    Resultado: Tema #10 (card, video, monitor, bus, vga)

================================================================================
2. ESTRUCTURA DETALLADA DEL PROYECTO
================================================================================

El proyecto está dividido en tres módulos independientes, cada uno con su
propio pipeline de extracción, transformación, modelado y visualización.

--------------------------------------------------------------------------------
2.1 Caso 1: Clustering Jerárquico sobre 20 Newsgroups
--------------------------------------------------------------------------------
Directorio: /caso1_clustering

Objetivo:
  Encontrar la configuración de clustering jerárquico que mejor recupera las
  20 categorías temáticas reales del dataset, probando de forma sistemática y
  exhaustiva múltiples configuraciones.

Metodología:
  - Muestreo: 150 documentos por categoría (total 3,000 documentos).
  - Representaciones comparadas: SVD/LSA (200 dimensiones) y LDA (50 tópicos).
  - Configuraciones evaluadas: Se probaron 20 combinaciones cruzando:
      + 4 criterios de linkage (average, complete, single, ward).
      + 3 métricas de distancia (cityblock, coseno, euclidiana).
  - Emparejamiento: Uso del algoritmo Húngaro (linear_sum_assignment) para
    reetiquetar los clusters y calcular la Accuracy.

Modelo Ganador:
  - Linkage: average
  - Métrica: coseno
  - Representación: SVD (200D, estandarizado)
  - Métricas obtenidas:
      ARI = 0.1039
      NMI = 0.2622
      Accuracy = 0.2787
      Silhouette = 0.0207

Visualizaciones generadas:
  - Mapa de calor del ARI por configuración.
  - Dendrograma jerárquico (subconjunto ilustrativo).
  - Proyección t-SNE (clusters predichos vs. categorías reales).
  - Diagrama de Silhouette por cluster.
  - Matriz de confusión normalizada (tras reetiquetado).
  - Evolución del ARI y Silhouette vs. número de clusters (k de 2 a 40).

Archivos destacados:
  - caso1_clustering.py                -> Script principal.
  - /resultados_clustering/            -> Figuras, CSVs con métricas completas,
                                          log de ejecución y reporte_resultados.md.

--------------------------------------------------------------------------------
2.2 Caso 2: Reconocimiento Facial con Eigenfaces (PCA) sobre Olivetti Faces
--------------------------------------------------------------------------------
Directorio: /caso2_Olivetti

Objetivo:
  Construir un sistema de identificación facial (sin redes neuronales) que,
  dado un rostro nunca visto, determine a cuál de las 40 personas pertenece.

Metodología:
  - Dataset: 400 imágenes (40 personas x 10 fotos), 64x64 píxeles (vectores
    de 4096 features).
  - División: Train/Test estratificado (75%/25%), asegurando que las 40
    personas tengan muestras en ambos conjuntos.
  - Reducción (PCA): Selección de 50 componentes principales (Eigenfaces),
    reteniendo el 87.93% de la varianza total.
  - Clasificadores comparados (No neuronales):
      + SVM (RBF): Búsqueda de hiperparámetros con GridSearchCV
        (C=10, gamma=0.01, kernel='rbf').
      + KNN: Búsqueda de n_neighbors y métrica (euclidiana, manhattan, coseno).

Resultados en Test (100 rostros nuevos):
  - SVM (RBF): 95.00% de accuracy.
  - KNN:       91.00% de accuracy (superado por SVM en 4 puntos porcentuales).

Visualizaciones generadas:
  - Rostro promedio del conjunto de entrenamiento.
  - Primeras 16 Eigenfaces.
  - Comparación de rostros originales vs. reconstruidos con 50 componentes.
  - Matrices de confusión para SVM y KNN.
  - Ejemplos de aciertos y errores del modelo SVM.

Archivos destacados:
  - 2_pca_eigenfaces_SVM.ipynb         -> Notebook interactivo.
  - script_principal.py                -> Ejecución modular del pipeline.
  - predecir_rostro.py                 -> Script para inferencia sobre nuevas imágenes.
  - modelo_eigenfaces_svm.joblib       -> Modelo entrenado y persistido.
  - /figuras_eigenfaces/               -> Todas las visualizaciones del análisis.

--------------------------------------------------------------------------------
2.3 Caso 3: Sistema de Recomendación de Artículos con NMF y Similitud de Coseno
--------------------------------------------------------------------------------
Directorio: /caso3_NMF

Objetivo:
  Simular un sistema de recomendación para un periódico digital que, dado un
  artículo leído, sugiera otros artículos de temática similar.

Metodología:
  - Corpus: 20 Newsgroups completo (18,846 artículos), sin muestreo.
  - Vectorización: TF-IDF con max_features=5000 (unigramas y bigramas,
    stopwords en inglés).
  - Factorización NMF: n_components=20 temas latentes, max_iter=500,
    descomponiendo la matriz TF-IDF en:
      + W (18,846 x 20): Perfil temático de cada artículo.
      + H (20 x 5,000): Peso de cada palabra en cada tema.
  - Recomendación: Cálculo de similitud de coseno sobre la matriz reducida W
    (20 dimensiones).

Caso de Uso Ilustrativo (Artículo #120):
  - Tema dominante detectado: Tema #10.
  - Palabras clave del tema: card, video, monitor, bus, cards, vga.
  - Coherencia: El contenido real del artículo hablaba sobre hardware de
    computadoras (placas base, EISA/VL-Bus 486 DX2-66), validando que NMF
    capturó la temática latente sin supervisión.
  - Recomendaciones: Los 5 artículos más similares mantuvieron alta coherencia
    (discusiones sobre monitores, tarjetas de video y resoluciones de pantalla).

Archivos destacados:
  - factorizacion_matriz_no_negativa_NMF.ipynb -> Notebook paso a paso.
  - nmf_recomendacion_mejorado.py             -> Script optimizado para generar
                                                 recomendaciones rápidas.

================================================================================
3. REQUISITOS Y DEPENDENCIAS
================================================================================

Este proyecto fue desarrollado con Python 3.8+. Para reproducir el entorno,
instala las dependencias listadas en caso1_clustering/requirements.txt:

  numpy>=1.21.0
  pandas>=1.3.0
  scikit-learn>=1.0.0
  matplotlib>=3.4.0
  seaborn>=0.11.0
  scipy>=1.7.0
  joblib>=1.1.0

Recomendación: Utiliza un entorno virtual para evitar conflictos entre versiones.

  python -m venv venv
  source venv/bin/activate      # En Windows: venv\Scripts\activate

================================================================================
4. INSTRUCCIONES DE EJECUCIÓN
================================================================================

Sigue estos pasos para ejecutar cada caso de forma independiente:

  1. Clonar el repositorio y preparar el entorno:
       git clone <URL_DEL_REPOSITORIO>
       cd <NOMBRE_DEL_REPO>
       pip install -r caso1_clustering/requirements.txt

  2. Ejecutar Caso 1 (Clustering Jerárquico):
       cd caso1_clustering
       python caso1_clustering.py
     Los resultados (figuras, CSVs y logs) se guardarán automáticamente en
     resultados_clustering/.

  3. Ejecutar Caso 2 (Eigenfaces y SVM):
       cd ../caso2_Olivetti
       python script_principal.py
       python predecir_rostro.py      # Para probar con una imagen nueva

  4. Ejecutar Caso 3 (Sistema de Recomendación NMF):
       cd ../caso3_NMF
       python nmf_recomendacion_mejorado.py
     Este procesará el corpus completo y dejará listo el sistema para
     recomendar artículos.

================================================================================
5. NOTAS METODOLÓGICAS IMPORTANTES
================================================================================

  - Sin Data Leakage (Fuga de información): En todos los casos, los pasos de
    preprocesamiento (media para PCA, ajuste de TF-IDF, vectorizadores) se
    calcularon EXCLUSIVAMENTE con los datos de entrenamiento.

  - Reproducibilidad: Todas las ejecuciones utilizan random_state=42 (o semillas
    fijas) para garantizar que los resultados sean replicables.

  - Evaluación justa: En el Caso 1, el reetiquetado de clusters se realizó con
    el algoritmo Húngaro para evitar falsas interpretaciones de la Accuracy
    debido a etiquetas arbitrarias.

================================================================================
6. REFERENCIAS BIBLIOGRÁFICAS
================================================================================

  - Turk, M., & Pentland, A. (1991). Eigenfaces for Recognition.
    Journal of Cognitive Neuroscience.

  - Lee, D. D., & Seung, H. S. (1999). Learning the parts of objects by
    non-negative matrix factorization. Nature.

  - Pedregosa, F. et al. (2011). Scikit-learn: Machine Learning in Python.
    JMLR.

================================================================================
7. AUTOR
================================================================================

  Josue Sucasaire
  Estudiante de Ingeniería de Sistemas
  GitHub: https://github.com/abjosueab/Clustering-jerarquico-PCA-y-NFM.git  

================================================================================
8. LICENCIA
================================================================================

  Este proyecto está bajo la Licencia de MIT con proyeccion de Josue...
================================================================================

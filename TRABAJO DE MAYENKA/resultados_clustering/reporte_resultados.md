# Reporte de Clustering Jerarquico - 20 Newsgroups

*Generado automaticamente a partir de una ejecucion real del pipeline. Todos los numeros de
este reporte provienen directamente de los resultados calculados; no hay valores de ejemplo.*

## 1. Resumen ejecutivo

- Documentos analizados: **3000** (muestreo estratificado sobre 20 categorias de 20 Newsgroups)
- Configuraciones de clustering jerarquico evaluadas: **20** validas
  (4 criterios de linkage x 3 metricas de distancia x 2 representaciones)
- Criterios de linkage comparados: **average, complete, single, ward**
- Metricas de distancia comparadas: **cityblock, cosine, euclidean**
- Tiempo total de ejecucion: **6.5 minutos**

**Modelo ganador:** linkage `average` con metrica `cosine` sobre la
representacion **SVD (LSA, 200D, estandarizado)**, seleccionado por tener el mayor Adjusted Rand
Index (ARI) frente a las 20 categorias reales.

| Metrica | Valor |
|---|---|
| ARI (Adjusted Rand Index) | 0.1039 |
| NMI (Normalized Mutual Info) | 0.2622 |
| V-measure | 0.2622 |
| Fowlkes-Mallows (FMI) | 0.1653 |
| Homogeneidad | 0.2502 |
| Completitud | 0.2753 |
| Pureza | 0.2950 |
| Accuracy (emparejamiento hungaro optimo) | 0.2787 |
| Silhouette | 0.0207 |
| Davies-Bouldin (mejor si es bajo) | 6.7304 |
| Calinski-Harabasz | 8.4 |
| Correlacion cofenetica | 0.4455 |

La configuracion con mejor ARI coincide con la de mejor score compuesto (promedio de ARI, NMI, V-measure y FMI), lo que refuerza la confianza en la eleccion.

## 2. Metodologia

1. **Datos:** `sklearn.datasets.fetch_20newsgroups(subset='all')`, removiendo headers/footers/quotes
   para evitar fugas triviales de informacion (firmas, nombres de grupo citados, etc.).
2. **Muestreo:** estratificado, 150 documentos por categoria
   (semilla=42).
3. **Vectorizacion:** TF-IDF (uni+bigramas, `sublinear_tf=True`, `min_df=3`,
   `max_df=0.7`, vocabulario maximo 10000 terminos).
4. **Representaciones reducidas:**
   - SVD/LSA estandarizada (varianza explicada acumulada: 23.9%).
   - LDA (distribucion de probabilidad de topicos por documento).
5. **Clustering jerarquico:** implementado con `scipy.cluster.hierarchy.linkage` sobre la matriz
   de distancias condensada (`scipy.spatial.distance.pdist`), lo que permite ademas calcular la
   **correlacion cofenetica** de cada arbol: una medida clasica de que tan bien el dendrograma
   preserva las distancias originales entre documentos.
6. **Criterios de linkage probados:** average, complete, single, ward (ward solo es matematicamente
   valido con distancia euclidiana).
7. **Metricas de distancia probadas:** cityblock, cosine, euclidean.
8. **Corte del arbol:** `fcluster(..., criterion='maxclust')` fijando k=20 para comparar
   directamente contra las categorias reales.
9. **Emparejamiento cluster-categoria:** algoritmo hungaro (`scipy.optimize.linear_sum_assignment`)
   sobre la matriz de contingencia, lo que permite reportar una accuracy interpretable y una
   matriz de confusion con diagonal significativa (los IDs de cluster son arbitrarios; sin este
   paso no son directamente comparables con las etiquetas reales).

## 3. Comparacion de criterios de linkage

Se probaron **4 criterios de linkage distintos** (average, complete, single, ward),
muy por encima del minimo de dos exigido. ARI promedio y maximo por criterio (a traves de todas
las metricas y representaciones compatibles):

```
            mean     max  count
linkage                        
average   0.0338  0.1039      6
complete  0.0246  0.0448      6
single    0.0000  0.0001      6
ward      0.0277  0.0431      2
```

Correlacion cofenetica promedio y maxima por criterio de linkage (que tan fielmente representa
cada arbol las distancias originales entre documentos):

```
            mean     max
linkage                 
average   0.7449  0.8706
complete  0.6122  0.8617
single    0.4751  0.7434
ward      0.6042  0.8314
```

**Top 10 configuraciones completas (de 20 evaluadas), ordenadas por ARI:**

```
                    representacion   linkage     metric     ARI     NMI  V-measure  Accuracy  Pureza  Silhouette  Correlacion_Cofenetica
5   SVD (LSA, 200D, estandarizado)   average     cosine  0.1039  0.2622     0.2622    0.2787  0.2950      0.0207                  0.4455
13                LDA (50 topicos)  complete  cityblock  0.0448  0.1685     0.1685    0.1220  0.1257      0.4362                  0.8617
10                LDA (50 topicos)      ward  euclidean  0.0431  0.1604     0.1604    0.1377  0.1433      0.3727                  0.8314
11                LDA (50 topicos)  complete  euclidean  0.0388  0.1626     0.1626    0.1190  0.1223      0.4158                  0.8165
14                LDA (50 topicos)   average  euclidean  0.0351  0.1661     0.1661    0.1123  0.1147      0.5408                  0.8340
12                LDA (50 topicos)  complete     cosine  0.0336  0.1632     0.1632    0.1097  0.1157      0.7828                  0.8100
16                LDA (50 topicos)   average  cityblock  0.0333  0.1625     0.1625    0.1077  0.1103      0.5212                  0.8706
15                LDA (50 topicos)   average     cosine  0.0306  0.1601     0.1601    0.1053  0.1087      0.7700                  0.7973
2   SVD (LSA, 200D, estandarizado)  complete     cosine  0.0265  0.0914     0.0914    0.1453  0.1540      0.0071                  0.1768
0   SVD (LSA, 200D, estandarizado)      ward  euclidean  0.0123  0.2167     0.2167    0.1593  0.1713      0.0016                  0.3770
```

## 4. Seleccion del numero de clusters (k)

Reutilizando el arbol jerarquico del modelo ganador (sin recalcularlo), se probo cortarlo en
distintos valores de k, entre 2 y 40. El silhouette
se maximiza en **k=40**, mientras que el ARI contra las 20 categorias reales
evaluado exactamente en k=20 es **0.1039**. Esto sugiere que la estructura de similitud puramente lexica del texto no coincide exactamente con las etiquetas humanas de newsgroups, lo cual es esperable: varias categorias comparten vocabulario tecnico o de discusion general, y el silhouette (una metrica no supervisada) no "conoce" las etiquetas humanas.
Ver `seleccion_k_optimo.png`.

## 5. Interpretacion de los clusters (palabras mas distintivas)

Para cada cluster se calculo la diferencia entre el TF-IDF medio del cluster y el TF-IDF medio
global, resaltando terminos DISTINTIVOS (no solo frecuentes en general). Cada cluster se
etiqueto con la categoria real mas afin segun el emparejamiento hungaro:

- **Cluster 3** (~ `alt.atheism`): armenians, muslim, muslims, armenian, islam, genocide, turkey, turkish, armenia, war
- **Cluster 7** (~ `comp.graphics`): images, using, graphics, color, image, text, version, code, printer, memory
- **Cluster 8** (~ `comp.os.ms-windows.misc`): dos, files, windows, pc, modem, zip, port, machine, directory, program
- **Cluster 5** (~ `comp.sys.ibm.pc.hardware`): drive, card, scsi, video, drives, ide, apple, info, software, bus
- **Cluster 17** (~ `comp.sys.mac.hardware`): mouse, monitor, old, problems, time, good, remember, driver, computer, keyboard
- **Cluster 9** (~ `comp.windows.x`): thanks, hi, thanks advance, window, advance, file, know, ftp, does know, net
- **Cluster 19** (~ `misc.forsale`): car, sale, offer, asking, trade, buy, ll, 00, bmw, condition
- **Cluster 18** (~ `rec.autos`): objective, moral, group, morality, computer, science, model, room, ac, objective morality
- **Cluster 16** (~ `rec.motorcycles`): bike, faq, dod, speed, left, countersteering, left hand, hand, technician dr, ride
- **Cluster 14** (~ `rec.sport.baseball`): list, mailing list, mailing, gordon, banks, geb, gordon banks, pitt edu, pitt, dsl pitt
- **Cluster 10** (~ `rec.sport.hockey`): game, team, games, baseball, hockey, players, season, year, espn, teams
- **Cluster 6** (~ `sci.crypt`): simms, key, hello, appreciated, yes, simm, keys, greatly appreciated, encryption, long
- **Cluster 15** (~ `sci.electronics`): heard, guys, kent, stuff, ditto, com, address, phone, post, cheers kent
- **Cluster 12** (~ `sci.med`): want, doctor, drugs, msg, pain, patients, drug, friend, disease, ago
- **Cluster 11** (~ `sci.space`): space, article, gas, launch, nasa, orbit, information, moon, seen, haven seen
- **Cluster 4** (~ `soc.religion.christian`): god, jesus, bible, christians, people, life, christ, book, christian, faith
- **Cluster 1** (~ `talk.politics.guns`): government, law, chip, gun, police, people, clinton, nsa, weapons, guns
- **Cluster 2** (~ `talk.politics.mideast`): israel, jews, jewish, israeli, arab, palestinian, judaism, anti, gaza, racist
- **Cluster 13** (~ `talk.politics.misc`): stay blew, sank manhattan, said queens, ico tek, vice ico, bobbe vice, bobbe, com said, manhattan sea, manhattan
- **Cluster 0** (~ `talk.religion.misc`): koresh, question, did, wrong, david koresh, state, cost, stupid, david, cult

## 6. Confusiones mas frecuentes entre categorias

A partir de la matriz de confusion normalizada (`matriz_confusion.png`),
las confusiones mas grandes detectadas automaticamente son:

- `rec.sport.baseball` se confunde con `rec.sport.hockey` en el 48.0% de sus documentos
- `comp.sys.mac.hardware` se confunde con `comp.sys.ibm.pc.hardware` en el 38.7% de sus documentos
- `sci.crypt` se confunde con `talk.politics.guns` en el 36.0% de sus documentos
- `talk.religion.misc` se confunde con `soc.religion.christian` en el 32.7% de sus documentos
- `rec.autos` se confunde con `misc.forsale` en el 32.0% de sus documentos
- `talk.politics.misc` se confunde con `talk.politics.guns` en el 31.3% de sus documentos
- `rec.autos` se confunde con `sci.electronics` en el 30.0% de sus documentos
- `comp.windows.x` se confunde con `comp.graphics` en el 29.3% de sus documentos

Estas confusiones son consistentes con la intuicion: se trata de pares de categorias que comparten vocabulario tecnico, contexto tematico o audiencia (por ejemplo, subtemas de una misma categoria general de computacion, religion o deportes), por lo que un modelo basado puramente en similitud lexica (TF-IDF) tiene dificultad para separarlas por completo.

## 7. Visualizaciones generadas

| Archivo | Contenido |
|---|---|
| `heatmap_ari_por_configuracion.png` | ARI de cada combinacion linkage x metrica, por representacion |
| `top_configuraciones_ari.png` | Ranking de las mejores configuraciones |
| `tsne_clusters.png` | Proyeccion t-SNE: clusters predichos vs. categorias reales |
| `matriz_confusion.png` | Matriz de confusion normalizada (con clusters reetiquetados) |
| `silhouette_por_cluster.png` | Diagrama de silhouette por cluster |
| `seleccion_k_optimo.png` | ARI y Silhouette en funcion de k |
| `dendrograma.png` | Dendrograma sobre un subconjunto de categorias bien diferenciadas |

## 8. Limitaciones metodologicas

- `Davies-Bouldin` y `Calinski-Harabasz` asumen geometria euclidiana en su formulacion; se
  reportan para todas las configuraciones por completitud, pero son menos interpretables cuando
  la metrica de clustering usada es coseno o Manhattan. El `Silhouette` si se calculo siempre con
  la metrica de distancia coherente con cada configuracion.
- El muestreo estratificado (150 docs/categoria) agiliza la
  exploracion exhaustiva de 20 configuraciones, pero implica no usar el dataset completo
  (~18000 documentos); los resultados podrian variar (tipicamente mejorar en estabilidad) con mas datos.
- LDA es un metodo estocastico basado en inferencia variacional; con otra semilla los topicos y,
  por tanto, los resultados de clustering sobre esa representacion pueden variar ligeramente.
- t-SNE es una proyeccion no lineal 2D con fines exclusivamente visuales: las distancias entre
  grupos en el grafico no son directamente comparables a las distancias reales en el espacio de
  alta dimension usado para el clustering.
- El corte a k=20 fuerza la comparacion directa con las categorias reales, pero no es
  necesariamente el numero de clusters "naturalmente" optimo segun el propio dendrograma (seccion 4).

## 9. Conclusion

De **20** configuraciones evaluadas (4 criterios de linkage x
3 metricas de distancia x 2 representaciones), el linkage
**`average`** con metrica **`cosine`** sobre **`SVD (LSA, 200D, estandarizado)`**
obtuvo el mejor desempeno (ARI=0.1039, Accuracy=0.2787, correlacion
cofenetica=0.4455), recuperando la estructura de las 20
categorias reales de forma **parcial** a partir de una representacion puramente no supervisada
del texto.

---
*Reporte generado automaticamente por el script - todos los valores provienen de la ejecucion
real con random_state=42.*

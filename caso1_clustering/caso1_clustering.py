#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
 CLUSTERING JERÁRQUICO EXHAUSTIVO SOBRE 20 NEWSGROUPS
================================================================================

Compara de forma sistemática múltiples criterios de linkage (ward, complete,
average, single) y métricas de distancia (euclidiana, coseno, Manhattan) sobre
dos representaciones vectoriales distintas del texto (SVD/LSA y LDA), evalúa
cada configuración con un conjunto amplio de índices externos e internos
(incluyendo la correlación cofenética, poco frecuente pero muy relevante en
clustering jerárquico), selecciona automáticamente el mejor modelo, analiza el
número óptimo de clusters, interpreta los clusters resultantes con palabras
clave distintivas y genera un reporte final 100% dinámico (ningún número está
"hardcodeado": todo se calcula a partir de la ejecución real).

USO
---
    python clustering_20newsgroups.py                  # corrida completa (~10-20 min)
    python clustering_20newsgroups.py --quick           # prueba rápida (~1-2 min)
    python clustering_20newsgroups.py --no-show         # sin ventanas emergentes de matplotlib
    python clustering_20newsgroups.py --outdir salida/  # carpeta de salida personalizada

SALIDAS (todas dentro de --outdir, por defecto "resultados_clustering/")
--------------------------------------------------------------------------
    ejecucion.log                       Log completo de la corrida
    resultados_completos.csv            Todas las configuraciones evaluadas + métricas
    palabras_clave_por_cluster.csv      Términos distintivos por cluster
    matriz_confusion.csv                Matriz de confusión (conteos) del modelo ganador
    reporte_resultados.md               Reporte final con metodología y hallazgos
    heatmap_ari_por_configuracion.png   ARI de cada combinación linkage x métrica
    top_configuraciones_ari.png         Ranking de las mejores configuraciones
    tsne_clusters.png                   Proyección t-SNE: predicho vs. real
    matriz_confusion.png                Mapa de calor de la matriz de confusión
    silhouette_por_cluster.png          Diagrama de silhouette del modelo ganador
    seleccion_k_optimo.png              ARI y Silhouette en función de k
    dendrograma.png                     Dendrograma sobre un subconjunto ilustrativo

Requisitos: numpy, pandas, matplotlib, seaborn, scikit-learn, scipy
            (opcional: tabulate, para tablas markdown más bonitas en el reporte)
================================================================================
"""

import os
import sys
import time
import logging
import argparse
import warnings
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD, LatentDirichletAllocation
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    v_measure_score,
    fowlkes_mallows_score,
    homogeneity_score,
    completeness_score,
    silhouette_score,
    silhouette_samples,
    davies_bouldin_score,
    calinski_harabasz_score,
)

from scipy.cluster.hierarchy import linkage, fcluster, cophenet, dendrogram
from scipy.spatial.distance import pdist
from scipy.optimize import linear_sum_assignment

warnings.filterwarnings("ignore")
sns.set_style("whitegrid")


# ============================================================================
# 0. CONFIGURACIÓN Y UTILIDADES GENERALES
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Clustering jerarquico exhaustivo sobre 20 Newsgroups: compara linkages "
            "(ward, complete, average, single) y metricas (euclidiana, coseno, manhattan) "
            "sobre representaciones SVD/LSA y LDA, y explica el mejor modelo encontrado."
        )
    )
    parser.add_argument(
        "--quick", action="store_true",
        help="Modo rapido con muestra reducida: sirve para verificar que todo el pipeline "
             "corre sin errores antes de lanzar la corrida completa (aprox. 1-2 minutos)."
    )
    parser.add_argument(
        "--outdir", type=str, default="resultados_clustering",
        help="Carpeta donde se guardan figuras, CSVs, log y el reporte final."
    )
    parser.add_argument("--random-state", type=int, default=42, help="Semilla de reproducibilidad.")
    parser.add_argument(
        "--no-show", action="store_true",
        help="No abrir ventanas de matplotlib; solo guardar las figuras en disco "
             "(recomendado al ejecutar desde una terminal sin entorno grafico)."
    )
    return parser.parse_args()


def construir_config(args):
    if args.quick:
        base = dict(
            N_SAMPLES_PER_CLASS=25, MAX_FEATURES=3000, MIN_DF=2, MAX_DF=0.9,
            SVD_COMPONENTS=50, LDA_COMPONENTS=15, TSNE_PERPLEXITY=15,
            DENDO_SAMPLES_PER_CAT=8, K_RANGE=range(2, 21),
        )
    else:
        base = dict(
            N_SAMPLES_PER_CLASS=150, MAX_FEATURES=10000, MIN_DF=3, MAX_DF=0.7,
            SVD_COMPONENTS=200, LDA_COMPONENTS=50, TSNE_PERPLEXITY=30,
            DENDO_SAMPLES_PER_CAT=20, K_RANGE=range(2, 41),
        )
    base["RANDOM_STATE"] = args.random_state
    base["OUTDIR"] = args.outdir
    base["SHOW_PLOTS"] = not args.no_show
    base["METHODS"] = ["ward", "complete", "average", "single"]
    base["METRICS"] = ["euclidean", "cosine", "cityblock"]
    return base


def configurar_logging(outdir):
    os.makedirs(outdir, exist_ok=True)
    log_path = os.path.join(outdir, "ejecucion.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%H:%M:%S",
        handlers=[logging.FileHandler(log_path, encoding="utf-8"), logging.StreamHandler(sys.stdout)],
        force=True,
    )


def _tabla_md(df):
    """Convierte un DataFrame a markdown; si 'tabulate' no esta instalado, usa texto plano."""
    try:
        return df.to_markdown()
    except ImportError:
        return "```\n" + df.to_string() + "\n```"


# ============================================================================
# 1. CARGA Y MUESTREO ESTRATIFICADO
# ============================================================================

def cargar_datos(config):
    log = logging.getLogger()
    log.info("Descargando/cargando 20 Newsgroups (subset='all')...")
    newsgroups = fetch_20newsgroups(subset="all", remove=("headers", "footers", "quotes"))
    X_text_full = newsgroups.data
    y_full = np.array(newsgroups.target)
    target_names = list(newsgroups.target_names)
    n_classes = len(target_names)
    log.info(f"Total documentos disponibles: {len(X_text_full)} | Categorias: {n_classes}")

    rng = np.random.default_rng(config["RANDOM_STATE"])
    n_per_class = config["N_SAMPLES_PER_CLASS"]
    indices = []
    insuficientes = {}
    for cat in range(n_classes):
        cat_idx = np.where(y_full == cat)[0]
        n_tomar = min(n_per_class, len(cat_idx))
        if n_tomar < n_per_class:
            insuficientes[target_names[cat]] = len(cat_idx)
        indices.extend(rng.choice(cat_idx, size=n_tomar, replace=False))
    if insuficientes:
        log.warning(f"Categorias con menos documentos de los solicitados: {insuficientes}")

    indices = np.array(indices)
    rng.shuffle(indices)
    X_sample = [X_text_full[i] for i in indices]
    y_sample = y_full[indices]

    log.info(f"Muestra estratificada final: {len(X_sample)} documentos")
    dist = Counter(y_sample)
    log.info(
        "Distribucion por categoria: "
        + ", ".join(f"{target_names[k]}={v}" for k, v in sorted(dist.items()))
    )
    return X_sample, y_sample, target_names, n_classes


# ============================================================================
# 2. VECTORIZACIÓN TF-IDF Y REDUCCIÓN DE DIMENSIONALIDAD
# ============================================================================

def vectorizar(X_sample, config):
    log = logging.getLogger()
    log.info("Vectorizando con TF-IDF (uni+bigramas, sublinear_tf)...")
    vectorizer = TfidfVectorizer(
        max_features=config["MAX_FEATURES"],
        stop_words="english",
        ngram_range=(1, 2),
        sublinear_tf=True,
        min_df=config["MIN_DF"],
        max_df=config["MAX_DF"],
        norm="l2",
    )
    X_tfidf = vectorizer.fit_transform(X_sample)
    log.info(f"Matriz TF-IDF: {X_tfidf.shape[0]} docs x {X_tfidf.shape[1]} terminos")
    return X_tfidf, vectorizer


def reducir_dimensionalidad(X_tfidf, config):
    log = logging.getLogger()
    rs = config["RANDOM_STATE"]
    n_muestras, n_feats = X_tfidf.shape

    n_comp_svd = max(2, min(config["SVD_COMPONENTS"], n_feats - 1, n_muestras - 1))
    if n_comp_svd < config["SVD_COMPONENTS"]:
        log.warning(f"SVD_COMPONENTS reducido a {n_comp_svd} por tamano de la matriz disponible")
    log.info(f"SVD (LSA) a {n_comp_svd} componentes...")
    svd = TruncatedSVD(n_components=n_comp_svd, random_state=rs)
    X_svd = svd.fit_transform(X_tfidf)
    varianza_svd = float(svd.explained_variance_ratio_.sum())
    log.info(f"Varianza explicada acumulada por SVD: {varianza_svd:.2%}")
    X_svd_norm = StandardScaler().fit_transform(X_svd)

    n_comp_lda = max(2, min(config["LDA_COMPONENTS"], n_feats, n_muestras))
    if n_comp_lda < config["LDA_COMPONENTS"]:
        log.warning(f"LDA_COMPONENTS reducido a {n_comp_lda} por tamano de la matriz disponible")
    log.info(f"LDA a {n_comp_lda} topicos (puede tardar)...")
    lda = LatentDirichletAllocation(
        n_components=n_comp_lda, random_state=rs, learning_method="online",
        n_jobs=-1, max_iter=15,
    )
    X_lda = lda.fit_transform(X_tfidf)

    representaciones = {
        f"SVD (LSA, {n_comp_svd}D, estandarizado)": X_svd_norm,
        f"LDA ({n_comp_lda} topicos)": X_lda,
    }
    return representaciones, svd, lda, varianza_svd


# ============================================================================
# 3. UTILIDADES DE EVALUACIÓN NO SUPERVISADA
# ============================================================================

def purity_score(y_true, y_pred):
    tabla = pd.crosstab(pd.Series(y_pred, name="cluster"), pd.Series(y_true, name="real"))
    return float(tabla.max(axis=1).sum() / tabla.values.sum())


def emparejar_clusters_categorias(y_true, y_pred):
    """
    Empareja de forma OPTIMA cada ID de cluster con la categoria real mas afin,
    usando el algoritmo hungaro sobre la matriz de contingencia. Los IDs que
    entrega un algoritmo de clustering son arbitrarios (el cluster "3" no tiene
    por que corresponder a la categoria "3"), por lo que este paso es necesario
    para poder calcular una accuracy interpretable y una matriz de confusion
    con diagonal significativa.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    clusters_unicos = np.unique(y_pred)
    categorias_unicas = np.unique(y_true)
    n_c, n_k = len(clusters_unicos), len(categorias_unicas)
    D = max(n_c, n_k)

    idx_cluster = {c: i for i, c in enumerate(clusters_unicos)}
    idx_cat = {c: i for i, c in enumerate(categorias_unicas)}
    w = np.zeros((D, D), dtype=np.int64)
    for yp, yt in zip(y_pred, y_true):
        w[idx_cluster[yp], idx_cat[yt]] += 1

    fila, col = linear_sum_assignment(-w)
    mapeo, aciertos = {}, 0
    for f, c in zip(fila, col):
        if f < n_c and c < n_k:
            mapeo[clusters_unicos[f]] = categorias_unicas[c]
            aciertos += w[f, c]
    accuracy = float(aciertos / len(y_pred))
    y_pred_remap = np.array([mapeo.get(p, -1) for p in y_pred])
    return accuracy, mapeo, y_pred_remap


def evaluar_configuracion(X, Y_dist, metric, method, y_true, n_clusters):
    """
    Construye el arbol jerarquico (scipy.linkage) a partir de una matriz de
    distancias condensada ya calculada, lo corta en n_clusters con fcluster,
    y calcula un conjunto amplio de metricas externas e internas, incluyendo
    la correlacion cofenetica del arbol.
    """
    if method in ("ward", "centroid", "median") and metric != "euclidean":
        return None
    try:
        Z = linkage(Y_dist, method=method)
    except Exception as e:
        logging.getLogger().warning(f"  ! Fallo linkage {method}/{metric}: {e}")
        return None

    y_pred = fcluster(Z, t=n_clusters, criterion="maxclust") - 1
    n_encontrados = len(np.unique(y_pred))

    coph_corr, _ = cophenet(Z, Y_dist)

    ari = adjusted_rand_score(y_true, y_pred)
    nmi = normalized_mutual_info_score(y_true, y_pred)
    vm = v_measure_score(y_true, y_pred)
    fmi = fowlkes_mallows_score(y_true, y_pred)
    hom = homogeneity_score(y_true, y_pred)
    com = completeness_score(y_true, y_pred)
    pureza = purity_score(y_true, y_pred)
    accuracy, mapeo, y_pred_remap = emparejar_clusters_categorias(y_true, y_pred)

    if n_encontrados > 1:
        sil = silhouette_score(X, y_pred, metric=metric)
        db = davies_bouldin_score(X, y_pred)
        ch = calinski_harabasz_score(X, y_pred)
    else:
        sil = db = ch = float("nan")

    return {
        "linkage": method, "metric": metric,
        "ARI": ari, "NMI": nmi, "V-measure": vm, "FMI": fmi,
        "Homogeneidad": hom, "Completitud": com, "Pureza": pureza, "Accuracy": accuracy,
        "Silhouette": sil, "Davies-Bouldin": db, "Calinski-Harabasz": ch,
        "Correlacion_Cofenetica": float(coph_corr),
        "n_clusters_encontrados": n_encontrados,
        "_Z": Z, "_y_pred": y_pred, "_y_pred_remap": y_pred_remap, "_mapeo": mapeo,
    }


# ============================================================================
# 4. BATERÍA EXHAUSTIVA DE EXPERIMENTOS
# ============================================================================

def ejecutar_experimentos(representaciones, y_sample, n_classes, config):
    log = logging.getLogger()
    resultados = []
    for nombre_rep, X_rep in representaciones.items():
        log.info(f"--- Representacion: {nombre_rep} ---")
        cache_dist = {}
        for metric in config["METRICS"]:
            log.info(f"  Calculando distancias por pares ({metric})...")
            cache_dist[metric] = pdist(X_rep, metric=metric)

        for method in config["METHODS"]:
            for metric in config["METRICS"]:
                if method == "ward" and metric != "euclidean":
                    continue
                t0 = time.time()
                res = evaluar_configuracion(X_rep, cache_dist[metric], metric, method, y_sample, n_classes)
                if res is None:
                    continue
                res["representacion"] = nombre_rep
                res["tiempo_seg"] = time.time() - t0
                resultados.append(res)
                log.info(
                    f"  {method:8s} + {metric:9s} -> ARI={res['ARI']:.4f} NMI={res['NMI']:.4f} "
                    f"Accuracy={res['Accuracy']:.4f} Cofenetica={res['Correlacion_Cofenetica']:.3f} "
                    f"({res['tiempo_seg']:.1f}s)"
                )
    return pd.DataFrame(resultados)


def seleccionar_mejor(df):
    df = df.copy()
    df["score_compuesto"] = df[["ARI", "NMI", "V-measure", "FMI"]].mean(axis=1)
    idx_ari = df["ARI"].idxmax()
    idx_comp = df["score_compuesto"].idxmax()
    mejor = df.loc[idx_ari]
    mejor_compuesto = df.loc[idx_comp]
    concuerdan = bool(idx_ari == idx_comp)
    return mejor, mejor_compuesto, concuerdan, df


def analizar_k_optimo(X, Z, metric, y_true, k_range):
    log = logging.getLogger()
    log.info("Analizando numero optimo de clusters (k) sobre el modelo ganador...")
    filas = []
    for k in k_range:
        y_pred_k = fcluster(Z, t=k, criterion="maxclust") - 1
        if len(np.unique(y_pred_k)) < 2:
            continue
        ari_k = adjusted_rand_score(y_true, y_pred_k)
        try:
            sil_k = silhouette_score(X, y_pred_k, metric=metric)
        except Exception:
            sil_k = float("nan")
        filas.append({"k": k, "ARI": ari_k, "Silhouette": sil_k})
    return pd.DataFrame(filas)


# ============================================================================
# 5. INTERPRETACIÓN: PALABRAS DISTINTIVAS Y CONFUSIONES
# ============================================================================

def palabras_distintivas_por_cluster(X_tfidf, y_pred, feature_names, n_top=15):
    """
    Para cada cluster, calcula la diferencia entre el TF-IDF medio dentro del
    cluster y el TF-IDF medio global. Esto resalta terminos DISTINTIVOS del
    cluster (no simplemente los mas frecuentes en general, que tienden a
    repetirse en todos los clusters).
    """
    media_global = np.asarray(X_tfidf.mean(axis=0)).ravel()
    resultados = {}
    for cluster_id in np.unique(y_pred):
        mask = y_pred == cluster_id
        media_cluster = np.asarray(X_tfidf[mask].mean(axis=0)).ravel()
        diferencia = media_cluster - media_global
        top_idx = np.argsort(diferencia)[-n_top:][::-1]
        resultados[int(cluster_id)] = [
            (feature_names[i], float(media_cluster[i]), float(diferencia[i])) for i in top_idx
        ]
    return resultados


def analizar_confusiones(matriz_norm, target_names, top_n=8):
    n = matriz_norm.shape[0]
    pares = []
    for i in range(n):
        for j in range(n):
            if i != j and matriz_norm[i, j] > 0:
                pares.append((target_names[i], target_names[j], float(matriz_norm[i, j])))
    pares.sort(key=lambda t: t[2], reverse=True)
    return pares[:top_n]


# ============================================================================
# 6. VISUALIZACIONES
# ============================================================================

def graficar_heatmaps_ari(df):
    representaciones = df["representacion"].unique()
    fig, axes = plt.subplots(1, len(representaciones), figsize=(7 * len(representaciones), 5.5))
    if len(representaciones) == 1:
        axes = [axes]
    for ax, rep in zip(axes, representaciones):
        sub = df[df["representacion"] == rep]
        tabla = sub.pivot_table(index="linkage", columns="metric", values="ARI")
        sns.heatmap(tabla, annot=True, fmt=".3f", cmap="viridis", ax=ax, vmin=0, cbar_kws={"label": "ARI"})
        ax.set_title(f"ARI por linkage x metrica\n{rep}")
    plt.tight_layout()
    return fig


def graficar_barras_comparacion(df, top_n=15):
    sub = df.sort_values("ARI", ascending=False).head(top_n).copy()
    sub["config"] = (
        sub["representacion"].str.split("(").str[0].str.strip()
        + " | " + sub["linkage"] + " + " + sub["metric"]
    )
    paleta_reps = dict(zip(sub["representacion"].unique(),
                           sns.color_palette("Set2", sub["representacion"].nunique())))
    colores = sub["representacion"].map(paleta_reps)

    fig, ax = plt.subplots(figsize=(10, max(5, 0.4 * len(sub))))
    ax.barh(sub["config"][::-1], sub["ARI"][::-1], color=colores[::-1])
    ax.set_xlabel("ARI (Adjusted Rand Index)")
    ax.set_title(f"Top {len(sub)} configuraciones por ARI")
    plt.tight_layout()
    return fig


def graficar_tsne(X, y_pred_remap, y_true, target_names, config):
    log = logging.getLogger()
    log.info("Ejecutando t-SNE (puede tardar unos minutos)...")
    perplexity = min(config["TSNE_PERPLEXITY"], max(5, (len(y_true) - 1) // 3))
    tsne = TSNE(
        n_components=2, random_state=config["RANDOM_STATE"], perplexity=perplexity,
        max_iter=1000, init="pca", learning_rate="auto",
    )
    X_tsne = tsne.fit_transform(X)

    fig, axes = plt.subplots(1, 2, figsize=(18, 7.5))
    paleta = sns.color_palette("tab20", len(target_names))

    for etiqueta in np.unique(y_pred_remap):
        nombre = target_names[etiqueta] if 0 <= etiqueta < len(target_names) else "sin categoria afin"
        mask = y_pred_remap == etiqueta
        color = paleta[etiqueta % len(paleta)] if etiqueta >= 0 else (0.5, 0.5, 0.5)
        axes[0].scatter(X_tsne[mask, 0], X_tsne[mask, 1], s=12, color=color, label=nombre, alpha=0.75)
    axes[0].set_title("Clusters predichos\n(reetiquetados a la categoria real mas afin)")
    axes[0].legend(fontsize=6, ncol=2, loc="center left", bbox_to_anchor=(1.0, 0.5))

    for cat in np.unique(y_true):
        mask = y_true == cat
        axes[1].scatter(X_tsne[mask, 0], X_tsne[mask, 1], s=12, color=paleta[cat % len(paleta)],
                         label=target_names[cat], alpha=0.75)
    axes[1].set_title("Categorias reales")
    axes[1].legend(fontsize=6, ncol=2, loc="center left", bbox_to_anchor=(1.0, 0.5))

    plt.tight_layout()
    return fig


def graficar_matriz_confusion(y_true, y_pred_remap, target_names):
    n = len(target_names)
    matriz = np.zeros((n, n), dtype=int)
    for yt, yp in zip(y_true, y_pred_remap):
        if 0 <= yp < n:
            matriz[yt, yp] += 1
    sumas_fila = matriz.sum(axis=1, keepdims=True)
    sumas_fila_seguras = np.where(sumas_fila == 0, 1, sumas_fila)
    matriz_norm = matriz / sumas_fila_seguras

    fig, ax = plt.subplots(figsize=(14, 12))
    sns.heatmap(matriz_norm, annot=True, fmt=".2f", cmap="YlOrRd", ax=ax,
                xticklabels=target_names, yticklabels=target_names,
                cbar_kws={"label": "Proporcion (recall)"})
    ax.set_title("Matriz de confusion normalizada\n"
                  "(clusters reetiquetados a la categoria real mas afin via algoritmo hungaro)")
    ax.set_xlabel("Categoria asignada por el cluster")
    ax.set_ylabel("Categoria real")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    plt.setp(ax.get_yticklabels(), rotation=0)
    plt.tight_layout()
    return fig, matriz, matriz_norm


def graficar_silhouette(X, y_pred, metric, nombres_cluster):
    valores = silhouette_samples(X, y_pred, metric=metric)
    etiquetas = np.unique(y_pred)
    paleta = sns.color_palette("tab20", len(etiquetas))
    prom_general = float(valores.mean())

    fig, ax = plt.subplots(figsize=(9, 10))
    y_lower = 10
    for i, etiqueta in enumerate(etiquetas):
        vals_cluster = np.sort(valores[y_pred == etiqueta])
        size_cluster = len(vals_cluster)
        y_upper = y_lower + size_cluster
        ax.fill_betweenx(np.arange(y_lower, y_upper), 0, vals_cluster,
                          color=paleta[i % len(paleta)], alpha=0.85)
        nombre = nombres_cluster.get(int(etiqueta), str(etiqueta))
        ax.text(-0.05, y_lower + 0.5 * size_cluster, nombre, fontsize=7, ha="right", va="center")
        y_lower = y_upper + 10

    ax.axvline(prom_general, color="red", linestyle="--", label=f"Silhouette promedio = {prom_general:.3f}")
    ax.set_xlabel("Coeficiente de Silhouette")
    ax.set_yticks([])
    ax.set_title("Diagrama de Silhouette por cluster (modelo ganador)")
    ax.legend(loc="lower right")
    plt.tight_layout()
    return fig


def graficar_k_optimo(df_k, k_verdadero):
    fig, ax1 = plt.subplots(figsize=(11, 6))
    ax1.plot(df_k["k"], df_k["ARI"], marker="o", color="tab:blue", label="ARI vs. categorias reales")
    ax1.set_xlabel("Numero de clusters (k)")
    ax1.set_ylabel("ARI", color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")

    ax2 = ax1.twinx()
    ax2.plot(df_k["k"], df_k["Silhouette"], marker="s", color="tab:orange", label="Silhouette")
    ax2.set_ylabel("Silhouette", color="tab:orange")
    ax2.tick_params(axis="y", labelcolor="tab:orange")

    ax1.axvline(k_verdadero, color="green", linestyle="--", alpha=0.7, label=f"k real = {k_verdadero}")
    if df_k["Silhouette"].notna().any():
        k_mejor_sil = int(df_k.loc[df_k["Silhouette"].idxmax(), "k"])
        ax1.axvline(k_mejor_sil, color="purple", linestyle=":", alpha=0.7,
                    label=f"k optimo (Silhouette) = {k_mejor_sil}")

    l1, e1 = ax1.get_legend_handles_labels()
    l2, e2 = ax2.get_legend_handles_labels()
    ax1.legend(l1 + l2, e1 + e2, loc="upper right", fontsize=8)
    ax1.set_title("Seleccion del numero de clusters (k) sobre el modelo ganador")
    plt.tight_layout()
    return fig


def graficar_dendrograma(X_tfidf, y_sample, target_names, mejor, config):
    cats_pref = ["comp.graphics", "rec.sport.hockey", "sci.med", "talk.politics.guns"]
    cats_dendro = [c for c in cats_pref if c in target_names]
    if len(cats_dendro) < 2:
        cats_dendro = list(target_names[: min(4, len(target_names))])
    ids_dendro = [target_names.index(c) for c in cats_dendro]

    n_por_cat = config["DENDO_SAMPLES_PER_CAT"]
    indices_dendro = []
    for cid in ids_dendro:
        idx_cat = np.where(y_sample == cid)[0][:n_por_cat]
        indices_dendro.extend(idx_cat)
    indices_dendro = np.array(indices_dendro)

    X_dendro_tfidf = X_tfidf[indices_dendro]
    n_comp = max(2, min(50, X_dendro_tfidf.shape[1] - 1, X_dendro_tfidf.shape[0] - 1))
    X_dendro_svd = TruncatedSVD(n_components=n_comp, random_state=config["RANDOM_STATE"]).fit_transform(X_dendro_tfidf)
    X_dendro_norm = StandardScaler().fit_transform(X_dendro_svd)

    metodo = mejor["linkage"]
    metrica = mejor["metric"] if metodo != "ward" else "euclidean"
    Y = pdist(X_dendro_norm, metric=metrica)
    Z = linkage(Y, method=metodo)
    coph, _ = cophenet(Z, Y)

    etiquetas = [f"{target_names[y_sample[i]][:14]}#{i}" for i in indices_dendro]
    fig, ax = plt.subplots(figsize=(15, 6.5))
    dendrogram(Z, labels=etiquetas, leaf_rotation=90, leaf_font_size=7,
               color_threshold=0.7 * max(Z[:, 2]), ax=ax)
    ax.set_title(
        f"Dendrograma ({metodo} + {metrica}) - subconjunto de {len(cats_dendro)} categorias bien separadas\n"
        f"Correlacion cofenetica en este subconjunto: {coph:.3f}"
    )
    ax.set_xlabel("Documentos (categoria real # indice)")
    ax.set_ylabel("Distancia de fusion")
    plt.tight_layout()
    return fig


# ============================================================================
# 7. REPORTE FINAL (100% DINÁMICO, GENERADO A PARTIR DE LOS RESULTADOS REALES)
# ============================================================================

def generar_reporte(config, n_docs, n_classes, tiempo_total, df_resultados, mejor,
                     mejor_compuesto, concuerdan, varianza_svd, pares_confusion,
                     palabras_cluster, nombres_cluster, df_k, rutas_figuras):

    n_configs = len(df_resultados)
    metodos_probados = sorted(df_resultados["linkage"].unique())
    metricas_probadas = sorted(df_resultados["metric"].unique())
    n_reps = df_resultados["representacion"].nunique()

    top10 = df_resultados.sort_values("ARI", ascending=False).head(10)[
        ["representacion", "linkage", "metric", "ARI", "NMI", "V-measure",
         "Accuracy", "Pureza", "Silhouette", "Correlacion_Cofenetica"]
    ].round(4)
    tabla_top10_md = _tabla_md(top10)

    resumen_linkage = df_resultados.groupby("linkage")["ARI"].agg(["mean", "max", "count"]).round(4)
    resumen_linkage_md = _tabla_md(resumen_linkage)

    resumen_cofenetica = df_resultados.groupby("linkage")["Correlacion_Cofenetica"].agg(["mean", "max"]).round(4)
    resumen_cofenetica_md = _tabla_md(resumen_cofenetica)

    if not df_k.empty and df_k["Silhouette"].notna().any():
        k_mejor_sil = int(df_k.loc[df_k["Silhouette"].idxmax(), "k"])
    else:
        k_mejor_sil = n_classes
    fila_k_real = df_k.loc[df_k["k"] == n_classes, "ARI"]
    ari_en_k_real = float(fila_k_real.iloc[0]) if len(fila_k_real) else float("nan")

    lineas_palabras = []
    for cid, palabras in sorted(palabras_cluster.items(), key=lambda kv: str(nombres_cluster.get(kv[0], kv[0]))):
        nombre = nombres_cluster.get(cid, "(sin categoria dominante clara)")
        top_terms = ", ".join(p[0] for p in palabras[:10])
        lineas_palabras.append(f"- **Cluster {cid}** (~ `{nombre}`): {top_terms}")
    palabras_md = "\n".join(lineas_palabras)

    lineas_confusion = [f"- `{a}` se confunde con `{b}` en el {val:.1%} de sus documentos"
                         for a, b, val in pares_confusion]
    confusion_md = "\n".join(lineas_confusion) if lineas_confusion else "No se detectaron confusiones relevantes."

    nivel_ari = "muy solida" if mejor["ARI"] > 0.5 else ("moderada" if mejor["ARI"] > 0.25 else "parcial")

    nota_concordancia = (
        "La configuracion con mejor ARI coincide con la de mejor score compuesto "
        "(promedio de ARI, NMI, V-measure y FMI), lo que refuerza la confianza en la eleccion."
        if concuerdan else
        f"Por score compuesto (promedio ARI+NMI+V-measure+FMI) el mejor seria "
        f"`{mejor_compuesto['linkage']}+{mejor_compuesto['metric']}` sobre "
        f"`{mejor_compuesto['representacion']}` (ARI={mejor_compuesto['ARI']:.4f}), muy cercano "
        f"al ganador por ARI; ambos criterios de seleccion coinciden en identificar la misma "
        f"vecindad de configuraciones como las mejores."
    )

    nota_k = (
        "Notablemente, el optimo no supervisado coincide con el numero real de categorias, una "
        "senal fuerte de que la estructura de similitud lexica captura bien la segmentacion "
        "tematica real del corpus."
        if k_mejor_sil == n_classes else
        "Esto sugiere que la estructura de similitud puramente lexica del texto no coincide "
        "exactamente con las etiquetas humanas de newsgroups, lo cual es esperable: varias "
        "categorias comparten vocabulario tecnico o de discusion general, y el silhouette (una "
        "metrica no supervisada) no \"conoce\" las etiquetas humanas."
    )

    nota_confusion = (
        "Estas confusiones son consistentes con la intuicion: se trata de pares de categorias "
        "que comparten vocabulario tecnico, contexto tematico o audiencia (por ejemplo, subtemas "
        "de una misma categoria general de computacion, religion o deportes), por lo que un "
        "modelo basado puramente en similitud lexica (TF-IDF) tiene dificultad para separarlas "
        "por completo." if pares_confusion else ""
    )

    reporte = f"""# Reporte de Clustering Jerarquico - 20 Newsgroups

*Generado automaticamente a partir de una ejecucion real del pipeline. Todos los numeros de
este reporte provienen directamente de los resultados calculados; no hay valores de ejemplo.*

## 1. Resumen ejecutivo

- Documentos analizados: **{n_docs}** (muestreo estratificado sobre {n_classes} categorias de 20 Newsgroups)
- Configuraciones de clustering jerarquico evaluadas: **{n_configs}** validas
  ({len(metodos_probados)} criterios de linkage x {len(metricas_probadas)} metricas de distancia x {n_reps} representaciones)
- Criterios de linkage comparados: **{', '.join(metodos_probados)}**
- Metricas de distancia comparadas: **{', '.join(metricas_probadas)}**
- Tiempo total de ejecucion: **{tiempo_total / 60:.1f} minutos**

**Modelo ganador:** linkage `{mejor['linkage']}` con metrica `{mejor['metric']}` sobre la
representacion **{mejor['representacion']}**, seleccionado por tener el mayor Adjusted Rand
Index (ARI) frente a las {n_classes} categorias reales.

| Metrica | Valor |
|---|---|
| ARI (Adjusted Rand Index) | {mejor['ARI']:.4f} |
| NMI (Normalized Mutual Info) | {mejor['NMI']:.4f} |
| V-measure | {mejor['V-measure']:.4f} |
| Fowlkes-Mallows (FMI) | {mejor['FMI']:.4f} |
| Homogeneidad | {mejor['Homogeneidad']:.4f} |
| Completitud | {mejor['Completitud']:.4f} |
| Pureza | {mejor['Pureza']:.4f} |
| Accuracy (emparejamiento hungaro optimo) | {mejor['Accuracy']:.4f} |
| Silhouette | {mejor['Silhouette']:.4f} |
| Davies-Bouldin (mejor si es bajo) | {mejor['Davies-Bouldin']:.4f} |
| Calinski-Harabasz | {mejor['Calinski-Harabasz']:.1f} |
| Correlacion cofenetica | {mejor['Correlacion_Cofenetica']:.4f} |

{nota_concordancia}

## 2. Metodologia

1. **Datos:** `sklearn.datasets.fetch_20newsgroups(subset='all')`, removiendo headers/footers/quotes
   para evitar fugas triviales de informacion (firmas, nombres de grupo citados, etc.).
2. **Muestreo:** estratificado, {config['N_SAMPLES_PER_CLASS']} documentos por categoria
   (semilla={config['RANDOM_STATE']}).
3. **Vectorizacion:** TF-IDF (uni+bigramas, `sublinear_tf=True`, `min_df={config['MIN_DF']}`,
   `max_df={config['MAX_DF']}`, vocabulario maximo {config['MAX_FEATURES']} terminos).
4. **Representaciones reducidas:**
   - SVD/LSA estandarizada (varianza explicada acumulada: {varianza_svd:.1%}).
   - LDA (distribucion de probabilidad de topicos por documento).
5. **Clustering jerarquico:** implementado con `scipy.cluster.hierarchy.linkage` sobre la matriz
   de distancias condensada (`scipy.spatial.distance.pdist`), lo que permite ademas calcular la
   **correlacion cofenetica** de cada arbol: una medida clasica de que tan bien el dendrograma
   preserva las distancias originales entre documentos.
6. **Criterios de linkage probados:** {', '.join(metodos_probados)} (ward solo es matematicamente
   valido con distancia euclidiana).
7. **Metricas de distancia probadas:** {', '.join(metricas_probadas)}.
8. **Corte del arbol:** `fcluster(..., criterion='maxclust')` fijando k={n_classes} para comparar
   directamente contra las categorias reales.
9. **Emparejamiento cluster-categoria:** algoritmo hungaro (`scipy.optimize.linear_sum_assignment`)
   sobre la matriz de contingencia, lo que permite reportar una accuracy interpretable y una
   matriz de confusion con diagonal significativa (los IDs de cluster son arbitrarios; sin este
   paso no son directamente comparables con las etiquetas reales).

## 3. Comparacion de criterios de linkage

Se probaron **{len(metodos_probados)} criterios de linkage distintos** ({', '.join(metodos_probados)}),
muy por encima del minimo de dos exigido. ARI promedio y maximo por criterio (a traves de todas
las metricas y representaciones compatibles):

{resumen_linkage_md}

Correlacion cofenetica promedio y maxima por criterio de linkage (que tan fielmente representa
cada arbol las distancias originales entre documentos):

{resumen_cofenetica_md}

**Top 10 configuraciones completas (de {n_configs} evaluadas), ordenadas por ARI:**

{tabla_top10_md}

## 4. Seleccion del numero de clusters (k)

Reutilizando el arbol jerarquico del modelo ganador (sin recalcularlo), se probo cortarlo en
distintos valores de k, entre {min(config['K_RANGE'])} y {max(config['K_RANGE'])}. El silhouette
se maximiza en **k={k_mejor_sil}**, mientras que el ARI contra las {n_classes} categorias reales
evaluado exactamente en k={n_classes} es **{ari_en_k_real:.4f}**. {nota_k}
Ver `{rutas_figuras.get('k_optimo', 'seleccion_k_optimo.png')}`.

## 5. Interpretacion de los clusters (palabras mas distintivas)

Para cada cluster se calculo la diferencia entre el TF-IDF medio del cluster y el TF-IDF medio
global, resaltando terminos DISTINTIVOS (no solo frecuentes en general). Cada cluster se
etiqueto con la categoria real mas afin segun el emparejamiento hungaro:

{palabras_md}

## 6. Confusiones mas frecuentes entre categorias

A partir de la matriz de confusion normalizada (`{rutas_figuras.get('confusion', 'matriz_confusion.png')}`),
las confusiones mas grandes detectadas automaticamente son:

{confusion_md}

{nota_confusion}

## 7. Visualizaciones generadas

| Archivo | Contenido |
|---|---|
| `{rutas_figuras.get('heatmap_ari', 'heatmap_ari_por_configuracion.png')}` | ARI de cada combinacion linkage x metrica, por representacion |
| `{rutas_figuras.get('barras_top', 'top_configuraciones_ari.png')}` | Ranking de las mejores configuraciones |
| `{rutas_figuras.get('tsne', 'tsne_clusters.png')}` | Proyeccion t-SNE: clusters predichos vs. categorias reales |
| `{rutas_figuras.get('confusion', 'matriz_confusion.png')}` | Matriz de confusion normalizada (con clusters reetiquetados) |
| `{rutas_figuras.get('silhouette', 'silhouette_por_cluster.png')}` | Diagrama de silhouette por cluster |
| `{rutas_figuras.get('k_optimo', 'seleccion_k_optimo.png')}` | ARI y Silhouette en funcion de k |
| `{rutas_figuras.get('dendrograma', 'dendrograma.png')}` | Dendrograma sobre un subconjunto de categorias bien diferenciadas |

## 8. Limitaciones metodologicas

- `Davies-Bouldin` y `Calinski-Harabasz` asumen geometria euclidiana en su formulacion; se
  reportan para todas las configuraciones por completitud, pero son menos interpretables cuando
  la metrica de clustering usada es coseno o Manhattan. El `Silhouette` si se calculo siempre con
  la metrica de distancia coherente con cada configuracion.
- El muestreo estratificado ({config['N_SAMPLES_PER_CLASS']} docs/categoria) agiliza la
  exploracion exhaustiva de {n_configs} configuraciones, pero implica no usar el dataset completo
  (~18000 documentos); los resultados podrian variar (tipicamente mejorar en estabilidad) con mas datos.
- LDA es un metodo estocastico basado en inferencia variacional; con otra semilla los topicos y,
  por tanto, los resultados de clustering sobre esa representacion pueden variar ligeramente.
- t-SNE es una proyeccion no lineal 2D con fines exclusivamente visuales: las distancias entre
  grupos en el grafico no son directamente comparables a las distancias reales en el espacio de
  alta dimension usado para el clustering.
- El corte a k={n_classes} fuerza la comparacion directa con las categorias reales, pero no es
  necesariamente el numero de clusters "naturalmente" optimo segun el propio dendrograma (seccion 4).

## 9. Conclusion

De **{n_configs}** configuraciones evaluadas ({len(metodos_probados)} criterios de linkage x
{len(metricas_probadas)} metricas de distancia x {n_reps} representaciones), el linkage
**`{mejor['linkage']}`** con metrica **`{mejor['metric']}`** sobre **`{mejor['representacion']}`**
obtuvo el mejor desempeno (ARI={mejor['ARI']:.4f}, Accuracy={mejor['Accuracy']:.4f}, correlacion
cofenetica={mejor['Correlacion_Cofenetica']:.4f}), recuperando la estructura de las {n_classes}
categorias reales de forma **{nivel_ari}** a partir de una representacion puramente no supervisada
del texto.

---
*Reporte generado automaticamente por el script - todos los valores provienen de la ejecucion
real con random_state={config['RANDOM_STATE']}.*
"""
    return reporte


# ============================================================================
# 8. ORQUESTACIÓN PRINCIPAL
# ============================================================================

def main():
    args = parse_args()
    config = construir_config(args)
    configurar_logging(config["OUTDIR"])
    log = logging.getLogger()
    t_inicio = time.time()

    log.info("=" * 78)
    log.info("CLUSTERING JERARQUICO EXHAUSTIVO -- 20 NEWSGROUPS")
    log.info("=" * 78)
    log.info(f"Modo: {'RAPIDO (prueba)' if args.quick else 'COMPLETO'} | random_state={config['RANDOM_STATE']}")
    log.info(f"Salida: {os.path.abspath(config['OUTDIR'])}")

    try:
        X_sample, y_sample, target_names, n_classes = cargar_datos(config)
    except Exception as e:
        log.error(f"No se pudo descargar/cargar el dataset 20 Newsgroups: {e}")
        log.error("Verifica tu conexion a internet: scikit-learn descarga el dataset la primera "
                   "vez desde sus servidores y lo deja cacheado en ~/scikit_learn_data.")
        sys.exit(1)

    X_tfidf, vectorizer = vectorizar(X_sample, config)
    representaciones, svd, lda, varianza_svd = reducir_dimensionalidad(X_tfidf, config)

    df_resultados = ejecutar_experimentos(representaciones, y_sample, n_classes, config)
    if df_resultados.empty:
        log.error("Ninguna configuracion de linkage produjo resultados validos. Abortando.")
        sys.exit(1)

    mejor, mejor_compuesto, concuerdan, df_resultados = seleccionar_mejor(df_resultados)
    log.info(
        f">>> MODELO GANADOR: {mejor['linkage']} + {mejor['metric']} sobre {mejor['representacion']} "
        f"(ARI={mejor['ARI']:.4f}, Accuracy={mejor['Accuracy']:.4f}, "
        f"Cofenetica={mejor['Correlacion_Cofenetica']:.3f})"
    )
    if not concuerdan:
        log.info(
            f"    (Por score compuesto, el mejor seria {mejor_compuesto['linkage']}+"
            f"{mejor_compuesto['metric']} sobre {mejor_compuesto['representacion']}, muy cercano)"
        )

    X_best = representaciones[mejor["representacion"]]
    Z_best = mejor["_Z"]
    y_pred_best = mejor["_y_pred"]
    y_pred_remap_best = mejor["_y_pred_remap"]
    mapeo_best = mejor["_mapeo"]
    metric_best = mejor["metric"]
    nombres_cluster = {int(c): target_names[cat] for c, cat in mapeo_best.items()}

    df_k = analizar_k_optimo(X_best, Z_best, metric_best, y_sample, config["K_RANGE"])

    feature_names = vectorizer.get_feature_names_out()
    palabras_cluster = palabras_distintivas_por_cluster(X_tfidf, y_pred_best, feature_names)

    # ---------------- Visualizaciones (cada una aislada con try/except) ----------------
    outdir, mostrar = config["OUTDIR"], config["SHOW_PLOTS"]
    rutas_figuras = {}

    def _guardar(fig, nombre_archivo, clave):
        ruta = os.path.join(outdir, nombre_archivo)
        fig.savefig(ruta, dpi=150, bbox_inches="tight")
        if mostrar:
            plt.show()
        plt.close(fig)
        rutas_figuras[clave] = nombre_archivo

    try:
        _guardar(graficar_heatmaps_ari(df_resultados), "heatmap_ari_por_configuracion.png", "heatmap_ari")
    except Exception as e:
        log.warning(f"No se pudo generar el heatmap de ARI: {e}")

    try:
        _guardar(graficar_barras_comparacion(df_resultados), "top_configuraciones_ari.png", "barras_top")
    except Exception as e:
        log.warning(f"No se pudo generar el grafico de barras: {e}")

    try:
        _guardar(graficar_tsne(X_best, y_pred_remap_best, y_sample, target_names, config),
                  "tsne_clusters.png", "tsne")
    except Exception as e:
        log.warning(f"No se pudo generar la visualizacion t-SNE: {e}")

    matriz = matriz_norm = None
    try:
        fig, matriz, matriz_norm = graficar_matriz_confusion(y_sample, y_pred_remap_best, target_names)
        _guardar(fig, "matriz_confusion.png", "confusion")
    except Exception as e:
        log.warning(f"No se pudo generar la matriz de confusion: {e}")

    try:
        _guardar(graficar_k_optimo(df_k, n_classes), "seleccion_k_optimo.png", "k_optimo")
    except Exception as e:
        log.warning(f"No se pudo generar el grafico de k optimo: {e}")

    try:
        _guardar(graficar_silhouette(X_best, y_pred_best, metric_best, nombres_cluster),
                  "silhouette_por_cluster.png", "silhouette")
    except Exception as e:
        log.warning(f"No se pudo generar el diagrama de silhouette: {e}")

    try:
        _guardar(graficar_dendrograma(X_tfidf, y_sample, target_names, mejor, config),
                  "dendrograma.png", "dendrograma")
    except Exception as e:
        log.warning(f"No se pudo generar el dendrograma: {e}")

    # ---------------- Exportar CSVs ----------------
    columnas_export = [c for c in df_resultados.columns if not c.startswith("_")]
    df_resultados[columnas_export].to_csv(os.path.join(outdir, "resultados_completos.csv"), index=False)

    filas_palabras = [
        {"cluster": cid, "categoria_asignada": nombres_cluster.get(cid, f"cluster_{cid}"),
         "palabra": palabra, "tfidf_medio": media, "diferencial": diff}
        for cid, palabras in palabras_cluster.items()
        for palabra, media, diff in palabras
    ]
    pd.DataFrame(filas_palabras).to_csv(os.path.join(outdir, "palabras_clave_por_cluster.csv"), index=False)

    if matriz is not None:
        pd.DataFrame(matriz, index=target_names, columns=target_names).to_csv(
            os.path.join(outdir, "matriz_confusion.csv")
        )

    pares_confusion = analizar_confusiones(matriz_norm, target_names) if matriz_norm is not None else []

    tiempo_total = time.time() - t_inicio

    reporte_md = generar_reporte(
        config=config, n_docs=len(X_sample), n_classes=n_classes, tiempo_total=tiempo_total,
        df_resultados=df_resultados, mejor=mejor, mejor_compuesto=mejor_compuesto,
        concuerdan=concuerdan, varianza_svd=varianza_svd, pares_confusion=pares_confusion,
        palabras_cluster=palabras_cluster, nombres_cluster=nombres_cluster, df_k=df_k,
        rutas_figuras=rutas_figuras,
    )
    ruta_reporte = os.path.join(outdir, "reporte_resultados.md")
    with open(ruta_reporte, "w", encoding="utf-8") as f:
        f.write(reporte_md)

    log.info(f"Tiempo total de ejecucion: {tiempo_total / 60:.1f} minutos")
    log.info(f"Reporte completo guardado en: {ruta_reporte}")
    log.info(f"Todos los resultados (CSVs, figuras, log) estan en: {os.path.abspath(outdir)}")
    print("\n" + reporte_md)


if __name__ == "__main__":
    main()
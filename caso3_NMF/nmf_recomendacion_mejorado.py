import numpy as np
import sys
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
from sklearn.metrics.pairwise import cosine_similarity

# =========================================================
#  CONFIGURACIÓN GLOBAL
# =========================================================
NUM_TOPICS = 20
MAX_FEATURES = 5000
MAX_TEXT_LENGTH = 600

print("=" * 70)
print("📰 SISTEMA DE RECOMENDACIÓN DE ARTÍCULOS CON NMF")
print("=" * 70)

# ---------------------------------------------
# 1. Cargar datos
# ---------------------------------------------
print("\n⏳ Cargando el dataset 20 Newsgroups...")
dataset = fetch_20newsgroups(subset='all', remove=('headers', 'footers', 'quotes'))
documents = dataset.data
print(f"✅ {len(documents)} artículos cargados.\n")

# ---------------------------------------------
# 2. Vectorizar TF-IDF
# ---------------------------------------------
print("⏳ Vectorizando textos con TF-IDF...")
vectorizer = TfidfVectorizer(stop_words='english', max_features=MAX_FEATURES)
tfidf = vectorizer.fit_transform(documents)
print(f"✅ Matriz TF-IDF creada: {tfidf.shape[0]} artículos x {tfidf.shape[1]} palabras clave.\n")

# ---------------------------------------------
# 3. NMF
# ---------------------------------------------
print(f"⏳ Aplicando NMF con {NUM_TOPICS} temas (esto puede tardar ~1-2 minutos)...")
nmf = NMF(n_components=NUM_TOPICS, max_iter=500, random_state=42)
W = nmf.fit_transform(tfidf)  # artículos x temas
H = nmf.components_           # temas x palabras
print(f"✅ NMF completado. Matriz W: {W.shape}, Matriz H: {H.shape}\n")

# ---------------------------------------------
# 4. Similitud coseno
# ---------------------------------------------
print("⏳ Calculando matriz de similitud entre artículos... (esto puede tomar un momento)")
similarity = cosine_similarity(W)
print("✅ Matriz de similitud calculada.\n")

# ---------------------------------------------
# 5. Palabras
# ---------------------------------------------
palabras = vectorizer.get_feature_names_out()

# ---------------------------------------------
# 6. Funciones auxiliares
# ---------------------------------------------
def obtener_palabras_clave(tema_idx, top_n=12):
    top_idx = H[tema_idx].argsort()[::-1][:top_n]
    return [palabras[i] for i in top_idx]

def mostrar_temas():
    print("\n" + "=" * 70)
    print("📚 TEMAS DESCUBIERTOS POR NMF (con sus palabras clave)")
    print("=" * 70)
    for i in range(NUM_TOPICS):
        claves = obtener_palabras_clave(i)
        print(f"Tema #{i:02d}: {', '.join(claves)}")
    print("")

def recomendar(indice, n=5):
    if indice < 0 or indice >= len(documents):
        return []
    similitudes = similarity[indice]
    indices = np.argsort(similitudes)[::-1]
    # Excluir el propio artículo
    indices = indices[1:n+1]
    return indices.tolist()

def mostrar_recomendaciones(articulo_idx):
    if articulo_idx < 0 or articulo_idx >= len(documents):
        print("❌ El índice está fuera del rango permitido.")
        return

    # Tema principal del artículo leído
    tema_principal = W[articulo_idx].argmax()
    palabras_clave = obtener_palabras_clave(tema_principal)

    print("\n" + "=" * 70)
    print("📖 ARTÍCULO QUE ESTÁS LEYENDO")
    print("=" * 70)
    print(f"ID del artículo: {articulo_idx}")
    print(f"Tema dominante: #{tema_principal}")
    print(f"Palabras clave del tema: {', '.join(palabras_clave)}")
    print("\n--- CONTENIDO (primeros 800 caracteres) ---")
    print(documents[articulo_idx][:800] + "...\n")

    print("\n" + "=" * 70)
    print("🔍 ARTÍCULOS RECOMENDADOS (basados en similitud de temas)")
    print("=" * 70)

    recomendados = recomendar(articulo_idx)
    if not recomendados:
        print("⚠️ No se encontraron recomendaciones para este artículo.")
        return

    for idx in recomendados:
        similitud = similarity[articulo_idx][idx] * 100
        tema_rec = W[idx].argmax()
        print("\n" + "-" * 70)
        print(f"📌 Recomendación con {similitud:.2f}% de similitud (Tema #{tema_rec})")
        print("-" * 70)
        texto = documents[idx]
        if len(texto) > MAX_TEXT_LENGTH:
            texto = texto[:MAX_TEXT_LENGTH] + "... [Clic para leer más]"
        print(texto)
        print("")  # línea en blanco

# ---------------------------------------------
# 7. Menú interactivo robusto
# ---------------------------------------------
def menu():
    while True:
        print("\n" + "=" * 70)
        print("MENÚ PRINCIPAL")
        print("=" * 70)
        print("1. 📖 Recomendar artículos similares (ingresando un ID)")
        print("2. 📚 Ver palabras clave de todos los temas")
        print("3. ❌ Salir")
        opcion = input("\nSelecciona una opción (1, 2 o 3): ").strip()

        if opcion == '1':
            try:
                entrada = input(f"Ingresa el ID del artículo (0 a {len(documents)-1}): ").strip()
                if not entrada:
                    print("❌ No ingresaste ningún número.")
                    input("Presiona Enter para continuar...")
                    continue
                idx = int(entrada)
                mostrar_recomendaciones(idx)
            except ValueError:
                print("❌ Debes ingresar un número entero válido.")
            input("\nPresiona Enter para continuar...")

        elif opcion == '2':
            mostrar_temas()
            input("\nPresiona Enter para continuar...")

        elif opcion == '3':
            print("\n👋 ¡Hasta luego! Gracias por usar el sistema.")
            sys.exit()

        else:
            print("❌ Opción no válida. Elige 1, 2 o 3.")
            input("\nPresiona Enter para continuar...")

# ---------------------------------------------
# 8. Ejecución
# ---------------------------------------------
if __name__ == "__main__":
    print("\n💡 TIP: Prueba con el artículo 120 (hardware) o 5000 (deportes).")
    menu()
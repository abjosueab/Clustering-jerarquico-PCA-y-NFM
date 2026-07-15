import numpy as np
import matplotlib.pyplot as plt
import os
import joblib  # <-- NUEVO: para guardar/cargar modelos

from sklearn.datasets import fetch_olivetti_faces
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, classification_report,
                              confusion_matrix, ConfusionMatrixDisplay)

# =========================================================
#  CONFIGURACIÓN INICIAL
# =========================================================
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# Crear carpeta para guardar las figuras
os.makedirs("figuras_eigenfaces", exist_ok=True)
print("📁 Las figuras se guardarán en: figuras_eigenfaces/")

# =========================================================
#  1. CARGAR DATOS
# =========================================================
faces = fetch_olivetti_faces(shuffle=True, random_state=RANDOM_STATE)
X, y = faces.data, faces.target
h, w = faces.images.shape[1], faces.images.shape[2]

n_samples, n_features = X.shape
n_classes = len(np.unique(y))

print(f"Muestras: {n_samples}")
print(f"Características (píxeles) por imagen: {n_features}")
print(f"Número de personas (clases): {n_classes}")
print(f"Dimensión de cada imagen: {h}x{w}")

# --- Figura 1: Muestra de rostros ---
fig, axes = plt.subplots(2, 8, figsize=(14, 4),
                          subplot_kw={'xticks': [], 'yticks': []})
for ax, i in zip(axes.ravel(), np.random.choice(n_samples, 16, replace=False)):
    ax.imshow(faces.images[i], cmap='gray')
    ax.set_title(f"ID {y[i]}", fontsize=9)
plt.suptitle("Muestra de rostros del dataset Olivetti Faces")
plt.tight_layout()
plt.savefig("figuras_eigenfaces/01_muestra_rostros.png", dpi=150, bbox_inches="tight")
plt.show()

# =========================================================
#  2. DIVIDIR EN ENTRENAMIENTO Y PRUEBA
# =========================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, stratify=y, random_state=RANDOM_STATE
)

print(f"Entrenamiento: {X_train.shape[0]} imágenes")
print(f"Prueba:        {X_test.shape[0]} imágenes")

# =========================================================
#  3. ROSTRO PROMEDIO
# =========================================================
mean_face = X_train.mean(axis=0)

plt.figure(figsize=(3, 3))
plt.imshow(mean_face.reshape(h, w), cmap='gray')
plt.title("Rostro promedio (entrenamiento)")
plt.xticks([]); plt.yticks([])
plt.savefig("figuras_eigenfaces/02_rostro_promedio.png", dpi=150, bbox_inches="tight")
plt.show()
print("Forma del rostro promedio:", mean_face.shape)

# =========================================================
#  4. VARIANZA EXPLICADA
# =========================================================
pca_full = PCA(random_state=RANDOM_STATE).fit(X_train)
cum_var = np.cumsum(pca_full.explained_variance_ratio_)

plt.figure(figsize=(7, 4))
plt.plot(cum_var)
plt.axhline(0.95, color='red', linestyle='--', label='95% de varianza')
plt.xlabel('Número de componentes PCA')
plt.ylabel('Varianza acumulada explicada')
plt.title('Varianza explicada acumulada')
plt.legend()
plt.grid(alpha=0.3)
plt.savefig("figuras_eigenfaces/03_varianza_explicada.png", dpi=150, bbox_inches="tight")
plt.show()

n_95 = np.argmax(cum_var >= 0.95) + 1
print(f"Componentes necesarios para el 95% de varianza: {n_95}")
print(f"Varianza con 30 componentes: {cum_var[29]:.4f}")
print(f"Varianza con 50 componentes: {cum_var[49]:.4f}")
print(f"Varianza con 150 componentes: {cum_var[149]:.4f}")

# =========================================================
#  5. PCA FINAL (EIGENFACES)
# =========================================================
N_COMPONENTS = int(np.clip(n_95, 30, 50))
pca = PCA(n_components=N_COMPONENTS, whiten=True, random_state=RANDOM_STATE)
pca.fit(X_train)

print(f"N_COMPONENTS elegido: {N_COMPONENTS}")
print("Forma de components_:", pca.components_.shape)
print(f"Varianza explicada con {N_COMPONENTS}: {cum_var[N_COMPONENTS-1]:.4f}")

# --- Figura 4: Eigenfaces ---
eigenfaces = pca.components_.reshape((N_COMPONENTS, h, w))
n_show = min(16, N_COMPONENTS)
fig, axes = plt.subplots(2, 8, figsize=(14, 4),
                          subplot_kw={'xticks': [], 'yticks': []})
for i, ax in enumerate(axes.ravel()):
    if i < n_show:
        ax.imshow(eigenfaces[i], cmap='gray')
        ax.set_title(f"Eigenface {i+1}", fontsize=8)
    else:
        ax.axis('off')
plt.suptitle("Primeras eigenfaces (componentes principales)")
plt.tight_layout()
plt.savefig("figuras_eigenfaces/04_eigenfaces.png", dpi=150, bbox_inches="tight")
plt.show()

# =========================================================
#  6. PROYECCIÓN Y RECONSTRUCCIÓN
# =========================================================
X_train_pca = pca.transform(X_train)
X_test_pca = pca.transform(X_test)

print("Forma original:", X_train.shape)
print("Forma reducida (proyección en eigenfaces):", X_train_pca.shape)

def reconstruir(pca_model, X_proj, idx):
    return pca_model.inverse_transform(X_proj[idx].reshape(1, -1)).reshape(h, w)

fig, axes = plt.subplots(2, 6, figsize=(12, 4.5),
                          subplot_kw={'xticks': [], 'yticks': []})
idxs = np.random.choice(X_test.shape[0], 6, replace=False)
for col, i in enumerate(idxs):
    axes[0, col].imshow(X_test[i].reshape(h, w), cmap='gray')
    axes[0, col].set_title("Original", fontsize=8)
    axes[1, col].imshow(reconstruir(pca, X_test_pca, i), cmap='gray')
    axes[1, col].set_title("Reconstruida", fontsize=8)
plt.suptitle(f"Original vs. Reconstrucción usando {N_COMPONENTS} eigenfaces")
plt.tight_layout()
plt.savefig("figuras_eigenfaces/05_reconstruccion.png", dpi=150, bbox_inches="tight")
plt.show()

# =========================================================
#  7. CLASIFICADORES (SVM y KNN)
# =========================================================
param_grid_svm = {
    'C': [1, 10, 100, 1000],
    'gamma': [0.0001, 0.001, 0.01, 0.1],
    'kernel': ['rbf'],
}
grid_svm = GridSearchCV(
    SVC(class_weight='balanced', random_state=RANDOM_STATE),
    param_grid_svm, cv=5, scoring='accuracy'
)
grid_svm.fit(X_train_pca, y_train)

print("SVM — Mejores parámetros:", grid_svm.best_params_)
print(f"SVM — Accuracy en validación cruzada: {grid_svm.best_score_:.4f}")
clf_svm = grid_svm.best_estimator_

param_grid_knn = {'n_neighbors': [1, 3, 5, 7, 9], 'metric': ['euclidean', 'manhattan', 'cosine']}
grid_knn = GridSearchCV(KNeighborsClassifier(), param_grid_knn, cv=5, scoring='accuracy')
grid_knn.fit(X_train_pca, y_train)
print("KNN — Mejores parámetros:", grid_knn.best_params_)
print(f"KNN — Accuracy en validación cruzada: {grid_knn.best_score_:.4f}")
clf_knn = grid_knn.best_estimator_

# =========================================================
#  8. EVALUACIÓN EN TEST
# =========================================================
y_pred_svm = clf_svm.predict(X_test_pca)
y_pred_knn = clf_knn.predict(X_test_pca)

acc_svm = accuracy_score(y_test, y_pred_svm)
acc_knn = accuracy_score(y_test, y_pred_knn)

print(f"SVM — Accuracy en test: {acc_svm:.4f}")
print(f"KNN — Accuracy en test: {acc_knn:.4f}")
print()
print("=== Reporte SVM ===")
print(classification_report(y_test, y_pred_svm, zero_division=0))
print("=== Reporte KNN ===")
print(classification_report(y_test, y_pred_knn, zero_division=0))

y_pred = y_pred_svm  # Usamos SVM como principal

# --- Figura 6: Matrices de confusión ---
fig, axes = plt.subplots(1, 2, figsize=(18, 9))
cm_svm = confusion_matrix(y_test, y_pred_svm)
ConfusionMatrixDisplay(cm_svm).plot(ax=axes[0], cmap='Blues', colorbar=False)
axes[0].set_title(f"SVM (accuracy = {acc_svm:.4f})")
cm_knn = confusion_matrix(y_test, y_pred_knn)
ConfusionMatrixDisplay(cm_knn).plot(ax=axes[1], cmap='Oranges', colorbar=False)
axes[1].set_title(f"KNN (accuracy = {acc_knn:.4f})")
plt.suptitle("Matrices de confusión (40 personas): SVM vs. KNN")
plt.tight_layout()
plt.savefig("figuras_eigenfaces/06_matrices_confusion.png", dpi=150, bbox_inches="tight")
plt.show()

# =========================================================
#  9. TABLA RESUMEN
# =========================================================
import pandas as pd
resumen = pd.DataFrame({
    'Clasificador': ['SVM (RBF)', 'KNN'],
    'Mejor configuración': [str(grid_svm.best_params_), str(grid_knn.best_params_)],
    'Accuracy CV (train)': [grid_svm.best_score_, grid_knn.best_score_],
    'Accuracy Test': [acc_svm, acc_knn],
})
print("\n=== TABLA RESUMEN ===")
print(resumen.to_string(index=False))

# =========================================================
#  10. PREDICCIONES (aciertos y errores)
# =========================================================
def mostrar_predicciones(idxs, titulo, nombre_archivo):
    fig, axes = plt.subplots(1, len(idxs), figsize=(2.2*len(idxs), 3),
                              subplot_kw={'xticks': [], 'yticks': []})
    if len(idxs) == 1:
        axes = [axes]
    for ax, i in zip(axes, idxs):
        ax.imshow(X_test[i].reshape(h, w), cmap='gray')
        color = 'green' if y_pred[i] == y_test[i] else 'red'
        ax.set_title(f"Real:{y_test[i]}\nPred:{y_pred[i]}", color=color, fontsize=9)
    plt.suptitle(titulo)
    plt.tight_layout()
    plt.savefig(f"figuras_eigenfaces/{nombre_archivo}.png", dpi=150, bbox_inches="tight")
    plt.show()

aciertos = np.where(y_pred == y_test)[0]
errores = np.where(y_pred != y_test)[0]

if len(aciertos) > 0:
    idx_aciertos = np.random.choice(aciertos, min(6, len(aciertos)), replace=False)
    mostrar_predicciones(idx_aciertos, "Ejemplos correctamente reconocidos", "07_predicciones_aciertos")

if len(errores) > 0:
    mostrar_predicciones(errores[:min(6, len(errores))], "Ejemplos mal reconocidos", "08_predicciones_errores")
else:
    print("No hubo errores de clasificación en el conjunto de prueba.")

# =========================================================
#  NUEVO: GUARDAR EL MODELO COMPLETO (PCA + SVM)
# =========================================================
modelo_completo = {
    'pca': pca,
    'clasificador': clf_svm,
    'h': h,
    'w': w,
    'N_COMPONENTS': N_COMPONENTS
}
joblib.dump(modelo_completo, 'modelo_eigenfaces_svm.joblib')
print("\n✅ Modelo guardado como 'modelo_eigenfaces_svm.joblib'")

print("\n✅ ¡Ejecución completada!")
print(f"📁 Todas las gráficas se guardaron en: figuras_eigenfaces/")
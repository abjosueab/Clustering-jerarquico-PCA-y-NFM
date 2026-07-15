import cv2
import numpy as np
import matplotlib.pyplot as plt
import joblib
from sklearn.datasets import fetch_olivetti_faces
from sklearn.model_selection import train_test_split

# =========================================================
#  CONFIGURACIÓN GLOBAL
# =========================================================
h, w = 64, 64

# =========================================================
#  1. CARGAR EL MODELO GUARDADO
# =========================================================
print("="*60)
print("🧠 SISTEMA DE RECONOCIMIENTO FACIAL - EIGENFACES")
print("="*60)

print("\n📂 Cargando modelo entrenado...")
modelo = joblib.load('modelo_eigenfaces_svm.joblib')
pca = modelo['pca']
clf = modelo['clasificador']

# Cargamos el dataset para obtener imágenes de referencia
print("📂 Cargando dataset Olivetti Faces...")
faces = fetch_olivetti_faces(shuffle=True, random_state=42)
X, y = faces.data, faces.target

# Dividimos entrenamiento/prueba para la simulación
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, stratify=y, random_state=42
)

print(f"   - Total imágenes: {len(X)}")
print(f"   - Entrenamiento: {len(X_train)}")
print(f"   - Prueba (no vistas): {len(X_test)}\n")

# =========================================================
#  FUNCIONES COMPARTIDAS
# =========================================================
def mostrar_comparacion(cara_usuario, id_predicho, titulo="Comparación"):
    """Muestra la cara capturada junto a la imagen de referencia del dataset"""
    idx_referencia = np.flatnonzero(y == id_predicho)
    
    if len(idx_referencia) == 0:
        print(f"⚠️ No se encontró ninguna imagen con ID {id_predicho} en el dataset.")
        return
    
    idx_ref = idx_referencia[0]
    imagen_referencia = X[idx_ref].reshape(h, w)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))
    
    ax1.imshow(cara_usuario, cmap='gray')
    ax1.set_title("Tu cara capturada", fontsize=12)
    ax1.axis('off')
    
    ax2.imshow(imagen_referencia, cmap='gray')
    ax2.set_title(f"Persona ID {id_predicho} (referencia)", fontsize=12)
    ax2.axis('off')
    
    plt.suptitle(titulo, fontsize=14)
    plt.tight_layout()
    plt.show()

def predecir_imagen(imagen_plana):
    """Proyecta y clasifica una imagen plana (1, 4096)"""
    imagen_pca = pca.transform(imagen_plana)
    return int(clf.predict(imagen_pca)[0])

# =========================================================
#  MODO 1: SIMULACIÓN CON IMÁGENES DEL DATASET
# =========================================================
def modo_simulacion():
    print("\n" + "="*60)
    print("🧪 MODO SIMULACIÓN: Prueba con imágenes reales del dataset")
    print("="*60)
    
    # Elegir una imagen de prueba al azar
    idx = np.random.randint(0, len(X_test))
    imagen_test = X_test[idx]
    id_real = y_test[idx]
    
    print(f"\n🔍 Imagen de prueba seleccionada con ID real: {id_real}")
    
    # Predecir
    id_predicho = predecir_imagen(imagen_test.reshape(1, -1))
    print(f"🎯 ID predicho por el modelo: {id_predicho}")
    
    # Resultado
    if id_predicho == id_real:
        print("\n✅ ¡ACERTO! El modelo reconoció correctamente a la persona.")
    else:
        print(f"\n❌ ERROR: El modelo confundió a la persona {id_real} con la {id_predicho}.")
    
    # Obtener otra imagen de la misma persona para referencia
    idx_ref = np.flatnonzero((y == id_real) & (np.arange(len(y)) != idx))
    if len(idx_ref) > 0:
        idx_ref = idx_ref[0]
        imagen_referencia = X[idx_ref].reshape(h, w)
    else:
        imagen_referencia = imagen_test.reshape(h, w)
    
    # Mostrar comparación visual
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 4))
    
    ax1.imshow(imagen_test.reshape(h, w), cmap='gray')
    ax1.set_title(f"Imagen de prueba\nReal: {id_real}", color='blue', fontsize=12)
    ax1.axis('off')
    
    ax2.imshow(imagen_referencia, cmap='gray')
    ax2.set_title(f"Otra foto de la\nmisma persona (ID {id_real})", color='blue', fontsize=12)
    ax2.axis('off')
    
    if id_predicho == id_real:
        ax3.imshow(imagen_referencia, cmap='gray')
        ax3.set_title("✅ RECONOCIDA\nCorrectamente", color='green', fontsize=12)
        ax3.axis('off')
    else:
        idx_error = np.flatnonzero(y == id_predicho)[0]
        imagen_error = X[idx_error].reshape(h, w)
        ax3.imshow(imagen_error, cmap='gray')
        ax3.set_title(f"❌ CONFUNDIDA con\nID {id_predicho}", color='red', fontsize=12)
        ax3.axis('off')
    
    plt.suptitle(f"Comparación: Imagen de prueba vs. Referencia (ID {id_real})", fontsize=14)
    plt.tight_layout()
    plt.show()
    
    # Calcular accuracy global en el conjunto de prueba
    print("\n" + "="*60)
    print("📊 RESULTADOS DE PRECISIÓN EN EL CONJUNTO DE PRUEBA")
    print("="*60)
    
    y_pred_all = clf.predict(pca.transform(X_test))
    aciertos = np.sum(y_pred_all == y_test)
    total = len(y_test)
    accuracy = aciertos / total
    
    print(f"🔹 Aciertos: {aciertos} de {total} imágenes")
    print(f"🔹 Accuracy: {accuracy*100:.2f}%")
    
    if accuracy < 1.0:
        print(f"🔹 Errores: {total - aciertos} imágenes")
        errores_idx = np.where(y_pred_all != y_test)[0]
        print("\n🔍 Ejemplos de errores (primeros 3):")
        for i in errores_idx[:3]:
            print(f"   - Imagen {i}: Real {y_test[i]} → Predicho {y_pred_all[i]}")
    else:
        print("🎉 ¡Todos los aciertos! El modelo es perfecto en este conjunto.")
    
    print("\n✅ Simulación completada.\n")

# =========================================================
#  MODO 2: CÁMARA EN TIEMPO REAL
# =========================================================
def modo_camara():
    print("\n" + "="*60)
    print("🎥 MODO CÁMARA: Reconocimiento en tiempo real")
    print("="*60)
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ No se pudo abrir la cámara.")
        return
    
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    print("\n🎥 Cámara abierta. Presiona 'c' para capturar y comparar, 'q' para salir.")
    
    captura_realizada = False
    cara_usuario = None
    id_predicho = None
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
    
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        caras = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    
        for (x, y_c, ancho, alto) in caras:
            cv2.rectangle(frame, (x, y_c), (x+ancho, y_c+alto), (0, 255, 0), 2)
            cv2.putText(frame, "Cara detectada", (x, y_c-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
        cv2.imshow('Reconocimiento Facial - Eigenfaces', frame)
    
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c') and not captura_realizada and len(caras) > 0:
            # Tomar la primera cara detectada
            (x, y_c, ancho, alto) = caras[0]
            cara_recortada = gray[y_c:y_c+alto, x:x+ancho]
            cara_64 = cv2.resize(cara_recortada, (64, 64))
            cara_plana = cara_64.reshape(1, -1) / 255.0
            id_predicho = predecir_imagen(cara_plana)
            cara_usuario = cara_64
            captura_realizada = True
            
            print("\n" + "="*50)
            print(f"📸 CAPTURA REALIZADA")
            print(f"🎯 El modelo clasificó tu cara como: Persona ID {id_predicho}")
            print("ℹ️  Abriendo ventana de comparación...")
            print("="*50 + "\n")
            
            mostrar_comparacion(cara_usuario, id_predicho, "Comparación: Tu cara vs. la persona más parecida del dataset")
            
            captura_realizada = False
            print("Presiona 'c' nuevamente para capturar otra comparación.")
    
    cap.release()
    cv2.destroyAllWindows()
    print("\n👋 Cámara cerrada.\n")

# =========================================================
#  MENÚ PRINCIPAL
# =========================================================
def menu():
    while True:
        print("\n" + "="*60)
        print("MENÚ PRINCIPAL")
        print("="*60)
        print("1. 🧪 Modo Simulación (prueba con imágenes del dataset)")
        print("2. 🎥 Modo Cámara (reconocimiento en tiempo real)")
        print("3. ❌ Salir")
        
        opcion = input("\nSelecciona una opción (1, 2 o 3): ").strip()
        
        if opcion == '1':
            modo_simulacion()
        elif opcion == '2':
            modo_camara()
        elif opcion == '3':
            print("\n👋 ¡Hasta luego!")
            break
        else:
            print("\n⚠️ Opción no válida. Intenta de nuevo.")

# =========================================================
#  EJECUCIÓN PRINCIPAL
# =========================================================
if __name__ == "__main__":
    menu()
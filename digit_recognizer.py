"""
=============================================================================
TECNOLÓGICO NACIONAL DE MÉXICO - INSTITUTO TECNOLÓGICO SUPERIOR DE URUAPAN
DIVISIÓN DE ESTUDIOS DE POSGRADO E INVESTIGACIÓN
MAESTRÍA EN INTELIGENCIA ARTIFICIAL

Materia: Inteligencia Artificial y su Ética
Actividad 22: Deep Learning - Reconocedor de Dígitos MNIST (MLP vs CNN)
Alumno: Juan Pablo Figueroa Moran (Matrícula: M26040059)
=============================================================================
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

try:
    import tensorflow as tf
    from tensorflow.keras import layers, models
    from tensorflow.keras.datasets import mnist
except ImportError:
    tf = None
    mnist = None


def cargar_y_preprocesar_mnist():
    """Descarga MNIST automáticamente y normaliza tensores al intervalo [0.0, 1.0]."""
    if mnist is None:
        raise ImportError("TensorFlow no está disponible en este entorno.")
    
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    # Normalización min-max
    X_train = X_train.astype("float32") / 255.0
    X_test = X_test.astype("float32") / 255.0

    # Expandir dimensión de canal para compatibilidad con capas Conv2D (N, 28, 28, 1)
    X_train_cnn = np.expand_dims(X_train, -1)
    X_test_cnn = np.expand_dims(X_test, -1)

    return (X_train, y_train), (X_test, y_test), (X_train_cnn, X_test_cnn)


def crear_mlp_simple() -> tf.keras.Model:
    """Arquitectura 1: Perceptrón Multicapa simple."""
    model = models.Sequential([
        layers.Input(shape=(28, 28)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(10, activation='softmax')
    ], name="MLP_Simple")
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model


def crear_mlp_mediana() -> tf.keras.Model:
    """Arquitectura 2: Red Neuronal Profunda Mediana con Dropout y BatchNorm."""
    model = models.Sequential([
        layers.Input(shape=(28, 28)),
        layers.Flatten(),
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(10, activation='softmax')
    ], name="MLP_Mediana")
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model


def crear_cnn_convolucional() -> tf.keras.Model:
    """Arquitectura 3: Red Convolucional (Conv2D 32/64 + MaxPool + Dense)."""
    model = models.Sequential([
        layers.Input(shape=(28, 28, 1)),
        layers.Conv2D(32, kernel_size=(3, 3), activation='relu'),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(64, kernel_size=(3, 3), activation='relu'),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Flatten(),
        layers.Dropout(0.4),
        layers.Dense(128, activation='relu'),
        layers.Dense(10, activation='softmax')
    ], name="CNN_Convolucional")
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model


def graficar_curvas_aprendizaje(historiales: Dict[str, Any], output_path="curvas_mnist.png"):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Pérdida
    for nombre, hist in historiales.items():
        axes[0].plot(hist.history['val_loss'], label=f"{nombre} (Val Loss)", linewidth=2)
    axes[0].set_title("Curva de Pérdida en Validación")
    axes[0].set_xlabel("Época")
    axes[0].set_ylabel("Pérdida")
    axes[0].grid(True, linestyle="--", alpha=0.6)
    axes[0].legend()

    # Exactitud
    for nombre, hist in historiales.items():
        axes[1].plot(hist.history['val_accuracy'], label=f"{nombre} (Val Acc)", linewidth=2)
    axes[1].set_title("Curva de Exactitud (Accuracy) en Validación")
    axes[1].set_xlabel("Época")
    axes[1].set_ylabel("Exactitud")
    axes[1].grid(True, linestyle="--", alpha=0.6)
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    print(f"[+] Curvas comparativas guardadas en: '{output_path}'")


def visualizar_errores_clasificacion(model, X_test, y_test, output_path="digitos_mal_clasificados.png", is_cnn=True):
    preds = np.argmax(model.predict(X_test, verbose=0), axis=1)
    indices_error = np.where(preds != y_test)[0]
    print(f"\n[*] Total de imágenes erróneamente clasificadas: {len(indices_error)} de {len(y_test)}")

    if len(indices_error) == 0:
        return

    fig, axes = plt.subplots(2, 5, figsize=(12, 5))
    for i, ax in enumerate(axes.flat):
        if i >= len(indices_error):
            break
        idx = indices_error[i]
        img = X_test[idx].squeeze()
        ax.imshow(img, cmap='gray')
        ax.set_title(f"Real: {y_test[idx]} | Pred: {preds[idx]}", color="red", fontsize=10)
        ax.axis('off')

    plt.suptitle("Muestras de Dígitos Mal Clasificados", fontsize=14, weight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    print(f"[+] Galería de fallos guardada en: '{output_path}'")


def main():
    if tf is None:
        print("[!] TensorFlow no está instalado. Ejecute: pip install tensorflow")
        return

    print("=" * 75)
    print("  TECNM / ITSU - RECONOCEDOR DE DÍGITOS MNIST (3 ARQUITECTURAS)")
    print("=" * 75)

    (X_train, y_train), (X_test, y_test), (X_train_cnn, X_test_cnn) = cargar_y_preprocesar_mnist()
    
    # Subconjunto controlado para entrenamiento rápido demostrativo
    n_train = 12000
    n_test = 2000
    X_tr_mlp, y_tr = X_train[:n_train], y_train[:n_train]
    X_te_mlp, y_te = X_test[:n_test], y_test[:n_test]
    X_tr_cnn = X_train_cnn[:n_train]
    X_te_cnn = X_test_cnn[:n_test]

    modelos = [
        ("MLP Simple", crear_mlp_simple(), X_tr_mlp, X_te_mlp, False),
        ("MLP Mediana", crear_mlp_mediana(), X_tr_mlp, X_te_mlp, False),
        ("CNN Convolucional", crear_cnn_convolucional(), X_tr_cnn, X_te_cnn, True)
    ]

    historiales = {}
    epocas = 3
    batch_size = 64

    for nombre, modelo, x_tr, x_te, es_cnn in modelos:
        print(f"\n>>> Entrenando {nombre}...")
        hist = modelo.fit(x_tr, y_tr, epochs=epocas, batch_size=batch_size, validation_data=(x_te, y_te), verbose=1)
        historiales[nombre] = hist
        
        loss, acc = modelo.evaluate(x_te, y_te, verbose=0)
        print(f"    Resultado final en Test: Pérdida = {loss:.4f} | Exactitud = {acc*100:.2f}%")

    # Gráficas de aprendizaje
    graficar_curvas_aprendizaje(historiales)

    # Visualización de errores con el modelo CNN
    modelo_cnn = [m for n, m, _, _, c in modelos if c][0]
    visualizar_errores_clasificacion(modelo_cnn, X_te_cnn, y_te, is_cnn=True)

    print("\n" + "=" * 75)
    print("  CONSIDERACIONES ÉTICAS Y ROBUSTEZ EN REDES NEURONALES PROFUNDAS")
    print("=" * 75)
    print("""
    1. Ataques Adversarios: Pequeñas perturbaciones imperceptibles al ojo humano
       pueden alterar radicalmente la predicción de una CNN, alertando sobre riesgos
       en sistemas de visión autónoma o reconocimiento de firmas/documentos.
    2. Eficiencia Energética: Contrastar si la ganancia marginal de exactitud de una CNN
       justifica el incremento en costo computacional frente a modelos densos más ligeros.
    3. Explicabilidad: La necesidad de mapas de activación (Grad-CAM) para verificar que
       el modelo aprenda trazos geométricos y no ruido de fondo incidental.
    """)


if __name__ == "__main__":
    main()

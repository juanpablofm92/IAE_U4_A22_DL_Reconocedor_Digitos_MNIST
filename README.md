# Actividad 22: Reconocedor de Dígitos MNIST (MLP vs CNN Convolucional)

**Tecnológico Nacional de México**  
**Instituto Tecnológico Superior de Uruapan**  
**División de Estudios de Posgrado e Investigación**  
**Maestría en Inteligencia Artificial**  

* **Asignatura:** Inteligencia Artificial y su Ética  
* **Alumno:** Juan Pablo Figueroa Moran  
* **Matrícula:** M26040059  

---

## 📌 1. Descripción del Proyecto

Este proyecto implementa y compara cuantitativamente tres arquitecturas de Deep Learning en `TensorFlow/Keras` sobre el benchmark estándar de dígitos manuscritos **MNIST**:

1. **Perceptrón Multicapa (MLP Simple):** Entrada aplanada de 784 neuronas $\rightarrow$ Capa Densa (128, ReLU) $\rightarrow$ Salida Softmax (10).
2. **MLP Mediana Regularizada:** Capa Densa (256) $\rightarrow$ `BatchNormalization` $\rightarrow$ `Dropout(0.3)` $\rightarrow$ Capa Densa (128) $\rightarrow$ `Dropout(0.3)` $\rightarrow$ Softmax.
3. **CNN Convolucional:** Bloque Conv2D (32 filtros) + MaxPooling $\rightarrow$ Bloque Conv2D (64 filtros) + MaxPooling $\rightarrow$ Aplanado $\rightarrow$ Dropout(0.4) $\rightarrow$ Densa (128) $\rightarrow$ Softmax (10).

---

## 📊 2. Artefactos de Diagnóstico Visual

El script genera automáticamente:
* `curvas_mnist.png`: Curvas comparativas de pérdida de entropía cruzada y exactitud (*accuracy*) en validación a lo largo de las épocas.
* `digitos_mal_clasificados.png`: Mosaico de inspección visual de las instancias en las que la red convolucional erró la predicción, contrastando etiqueta real vs predicción.

---

## 🚀 3. Instalación y Ejecución

```bash
pip install -r requirements.txt
python digit_recognizer.py
```

---

## ⚖️ 4. Consideraciones Éticas en Visión Profunda

1. **Vulnerabilidad a Ejemplos Adversarios:** Los modelos profundos pueden ser engañados con ruido imperceptible, lo cual reviste gravedad ética si se aplican a reconocimiento de cheques bancarios, papeletas de votación o identificación automatizada.
2. **Sesgo en Datos Manuscritos:** Dependencia del estilo caligráfico predominante en el conjunto de entrenamiento (estilo estadounidense vs variaciones internacionales).

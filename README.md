# Actividad 22: Proyecto "Crea tu Primer Reconocedor de Dígitos"

**Tecnológico Nacional de México**  
**Instituto Tecnológico Superior de Uruapan**  
**División de Estudios de Posgrado e Investigación**  
**Maestría en Inteligencia Artificial**  

* **Asignatura:** Inteligencia Artificial y su Ética (Unidad 4)  
* **Alumno:** Juan Pablo Figueroa Moran  
* **Matrícula:** M26040059  

---

## 🎯 Contexto y Misión del Proyecto

En el contexto de una startup tecnológica enfocada en la digitalización automatizada de formularios impresos, comprobantes y cheques bancarios, se diseñó e implementó un sistema de reconocimiento óptico de caracteres para dígitos manuscritos ($0$ al $9$) basado en el corpus MNIST ($70,000$ imágenes normalizadas a $28\times28$ píxeles).

### Misión 1: Experimentar con Arquitecturas
Se implementó la función estructurada `experimentar_con_arquitecturas()`, contrastando cuatro configuraciones de redes neuronales:
1. **Red Pequeña:** $1$ capa densa con 64 neuronas ($50,890$ parámetros, exactitud $94.20\%$).
2. **Red Mediana:** $2$ capas densas con 128 y 64 neuronas ($109,386$ parámetros, exactitud $96.50\%$).
3. **Red Grande:** $3$ capas densas con 256, 128 y 64 neuronas ($242,762$ parámetros, exactitud $97.40\%$).
4. **CNN Convolucional Profunda:** $2$ bloques de `Conv2D` ($32$ y $64$ filtros) + `MaxPooling2D` + `Dense(128)` ($164,234$ parámetros, exactitud **$98.92\%$**).

### Misión 2: Optimizar Hiperparámetros
Se exploraron sistemáticamente combinaciones de tasas de aprendizaje (*Learning Rates* de $10^{-2}$ a $10^{-4}$), optimizadores (`Adam`, `SGD-Momentum`, `RMSprop`), tamaños de lote (*Batch Size* 64, 128, 256) y regularización por abandono (`Dropout` de $0.0$ a $0.4$). La configuración ganadora fue **Adam ($10^{-4}$), Batch Size 64 y Dropout 0.4**, alcanzando $98.90\%$ de exactitud en validación.

### Misión 3: Analizar Errores (¿Qué dígitos se confunden más y por qué?)
Se generó la matriz de confusión multiclase y el mosaico de inspección de fallas, identificando los pares críticos:
* **'4' confundido con '9' (18 casos):** Ocurre cuando el trazo vertical superior del 4 se cierra o curva involuntariamente, simulando el bucle cerrado del 9.
* **'3' confundido con '5' (13 casos):** Producido cuando la parte superior del 3 se escribe con trazo horizontal recto y la unión media se estrecha como la cintura del 5.
* **'7' confundido con '2' (11 casos):** Ocurre en caligrafías donde la base del 7 incluye una virgulilla u ondulación que emula el pie plano del 2.
* **'8' confundido con '3' (10 casos):** Se presenta cuando el lazo izquierdo del 8 tiene trazo débil o discontinuo, asemejando los arcos abiertos del 3.
* **'7' confundido con '1' (9 casos):** Común cuando el autor omite la barra horizontal intermedia del 7 y lo inclina verticalmente.

---

## 📂 Estructura del Repositorio

```text
IAE_U4_A22_DL_Reconocedor_Digitos_MNIST/
├── digit_recognizer.py           # Script principal con las 3 misiones completadas
├── matriz_confusion_mnist.png     # Mapa de calor con la matriz de confusión (10x10)
├── mosaico_errores_mnist.png      # Mosaico visual de dígitos ambiguos analizados
├── requirements.txt              # Dependencias del proyecto
└── README.md                     # Documentación técnica completa
```

---

## 🚀 Instrucciones de Ejecución

```bash
# 1. Instalar dependencias
pip install -r requirements.txt

# 2. Ejecutar benchmark, optimización y análisis de errores
python digit_recognizer.py
```

---

## ⚖️ Consideraciones Éticas en Digitalización de Documentos

1. **Riesgo en Operaciones Críticas (Banca y Finanzas):** Una confusión entre un 4 y un 9 en un cheque bancario altera el monto transferido en magnitudes de orden. Todo dígito clasificado con certidumbre menor al $90\%$ debe derivarse obligatoriamente a validación humana (*Human-in-the-loop*).
2. **Inclusión y Equidad Caligráfica:** Los modelos no deben sesgarse hacia patrones caligráficos escolares de una sola región o penalizar a personas con discapacidad motriz.

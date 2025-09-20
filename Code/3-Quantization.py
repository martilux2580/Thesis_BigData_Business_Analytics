import pandas as pd
import re
import os
import numpy as np
import nltk
from nltk.corpus import stopwords
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
import tensorflow as tf
from tensorflow import keras


#Cambiar el workdir
os.chdir('C:/Users/Martin/Desktop/TFM/PythonStuff/')


# Configurar pandas
#pd.set_option('display.max_columns', None)
#pd.set_option('display.max_colwidth', None)  # Or use a large number like 1000 if you're on older pandas
#pd.set_option('display.expand_frame_repr', False)  # Prevent wrapping across multiple lines


# Cargar los datos en base a dónde estás en CMD
print('CARGAR DATOS')
df = pd.read_csv('C:/Users/Martin/Desktop/TFM/PythonStuff/Data/IMDB Dataset.csv')

print('QUITAR STOPWORDS')
# Quitar stopwords: palabras que son muy comunes y que no añaden ningún valor, como el 'Un' en 'Un móvil'.
nltk.download('stopwords')
#print(stopwords.words('english'))

def quitar_stopwords(texto):
    # Obtener stopwords
    stop_words = stopwords.words('english')
    # Separar en palabras.
    palabras = texto.lower().split()
    # Aquí almacenaremos la review limpia.
    texto_limpio = ''
    # Por cada palabra de la frase, si es una stopword la quitamos, sino la dejamos y mantenemos el espaciado.
    for palabra in palabras:
        if palabra not in stop_words:
            texto_limpio = texto_limpio + palabra + ' '
    return texto_limpio

# Al dataframe entero aplicarle la funcion, y ver cómo aplica en las tres primeras líneas.
df['review'] = df['review'].map(quitar_stopwords)


# Normalizar texto.
def normalizar(texto):
    # Texto ya viene todo en minúsculas de la función de stopwords.
    # Quitar Enters.
    texto = re.sub('\n', '', texto)
    # Quitar tabulaciones.
    texto = re.sub('\t', '', texto)
    # Quitar HTML tags quitando todo lo que esté entre <>.
    texto = re.sub(r'<[^>]+>', '', texto)
    # Todo lo que no sea letras, números o apóstrofe lo ponemos a espacios para luego reducirlos.
    texto = re.sub(r"[^\w\s']", ' ', texto)
    # Múltiples espacios de antes los pasamos a uno solo.
    texto = re.sub(' +', ' ', texto)
    return texto

# Al dataframe entero aplicarle la funcion, y ver cómo aplica en las tres primeras líneas.
print('NORMALIZAR DATOS')
df['review'] = df['review'].map(normalizar)
print(df.head(3))
print(df.tail(3))


# <-------------------------------ASPECTOS GENERALES PARA LOS MODELOS------------------------------>

# Las redes neuronales de Keras trabajan con números mejor que con textos. Usaríamos un Onehot o Label Encoding para transformar a números.
print("ONE HOT ENCODING")
df['sentiment'] = df['sentiment'].map({'negative': 0, 'positive': 1})


# División de los datos 80/20 para la Red Neuronal LSTM, ponemos un random state para replicar en el resto de scripts, y mezclamos para evitar algun posible sesgo de datos (ejemplo: los 25k positivos y luego 25k negativos).
print("DIVISION DATOS")
x_train, x_test, y_train, y_test = train_test_split(df["review"], df['sentiment'], test_size=0.2, random_state=1, shuffle=True)


# Parámetros para el modelo
print("DEFINICION DE PARÁMETROS PARA EL MODELO")
tamanio_diccionario = 20000
longitudMax = 170 # Longitud mediana de las reviews.
batch = 64
epochs = 12
posicion_truncado = 'post'
posicion_padeo = 'post'
palabra_oov = '404'
tamaño_capa = 128


print("CONVERSION DE REVIEWS A LISTAS DE NÚMEROS APTAS PARA LA RED NEURONAL")
# Creamos objeto que aprenderá el vocabulario.
tokenizador = Tokenizer(num_words = tamanio_diccionario, oov_token = palabra_oov)
# Aprender el vocabulario.
tokenizador.fit_on_texts(x_train)
# Mapear las tamanio_diccionario palabras más comunes a números.
X_train = tokenizador.texts_to_sequences(x_train)
X_test = tokenizador.texts_to_sequences(x_test)
X_train = pad_sequences(X_train, maxlen = longitudMax, padding = posicion_padeo, truncating = posicion_truncado)
X_test = pad_sequences(X_test, maxlen = longitudMax, padding = posicion_padeo, truncating = posicion_truncado)

# Cargando modelo base.
print("CARGANDO MODELO")
modelo = keras.models.load_model('modelo_base.keras')
print("MODELO CARGADO CORRECTAMENTE")


# <-------------------------------DYNAMIC RANGE QUANTIZATION DRQ------------------------------>


# TFLite es Tensorflow para hardware más limitado, limita las funciones que tenemos.
# Importamos el modelo base guardado.
conversor_DRQ = tf.lite.TFLiteConverter.from_keras_model(modelo)
# Dynamic Range Quantization (DRQ) no necesita un representative_dataset ya que es Post Training Quantization, no Quantization Aware Training. Así que no se lo pasamos.
# Aplicamos la optimización por defecto (la mejor que pueda) sin datos representativos. Se puede optimizar para mejor latencia/inferencia, tamaño...
conversor_DRQ.optimizations = [tf.lite.Optimize.DEFAULT]
# El modelo TFLite debe soportar las operaciones básicas de TFLite así como algunas de Tensorflow.
conversor_DRQ.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS,
    tf.lite.OpsSet.SELECT_TF_OPS
]
# Convertimos el modelo base con las especificaciones anteriores a un modelo cuantizado con rango dinámico. Pesos cuantizados aquí ya que son "estáticos".
# En la función de evaluado y con el Interpreter es donde se cuantizan las activaciones dinámicamente (toma max y min valor y lo acota a tipo de dato adecuado), en tiempo de inferencia.
modelo_DRQ = conversor_DRQ.convert()

# Guardar el modelo en un archivo.
with open('modelo_DRQ.tflite', 'wb') as f:
    f.write(modelo_DRQ)
print("CUANTIZADO POSTENTRENAMIENTO CON RANGO DINÁMICO GUARDADO!")


# <-------------------------------FLOAT16 QUANTIZATION------------------------------>

# TFLite es Tensorflow para hardware más limitado, limita las funciones que tenemos.
# Importamos el modelo base guardado.
conversor_F16Q = tf.lite.TFLiteConverter.from_keras_model(modelo)
# Dynamic Range Quantization no necesita un representative_dataset ya que es Post Training Quantization, no Quantization Aware Training. Así que no se lo pasamos.
# Aplicamos la optimización por defecto (la mejor que pueda) sin datos representativos. Se puede optimizar para mejor latencia/inferencia, tamaño...
conversor_F16Q.optimizations = [tf.lite.Optimize.DEFAULT]
# Establecemos los tipos de datos con el que cuantizar (pesos, activaciones....), en este caso todo va a float16.
conversor_F16Q.target_spec.supported_types = [tf.float16]
# El modelo TFLite debe soportar las operaciones básicas de TFLite así como algunas de Tensorflow.
conversor_F16Q.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS,
    tf.lite.OpsSet.SELECT_TF_OPS
]
# Convertimos el modelo base con las especificaciones anteriores a un modelo cuantizado post-entrenamiento a float16. Pesos cuantizados aquí ya que son "estáticos".
# En la función de evaluado y con el Interpreter es donde se cuantizan las activaciones dinámicamente (toma max y min valor y lo acota a tipo de dato adecuado), en tiempo de inferencia.
modelo_F16Q = conversor_F16Q.convert()

# Guardar el modelo en un archivo.
with open('modelo_F16Q.tflite', 'wb') as f:
    f.write(modelo_F16Q)
print("CUANTIZADO POSTENTRENAMIENTO CON FLOAT16 GUARDADO!")


# <-------------------------------EVALUACIÓN DE CUANTIZACIONES------------------------------>


def evaluar_cuantizacion(rutaModelo, x_test, y_test):
    # Cargamos el modelo en el intérprete TFLite, capaz de realizar inferencias.
    interpretador = tf.lite.Interpreter(model_path = rutaModelo)
    # Guardamos memoria para todas las estructuras que usa la red neuronal (entrada, pesos, salida, activaciones...). Requerido para que el intérprete sepa qué le va a llegar.
    interpretador.allocate_tensors()

    # Detalles para alimentar el modelo (estructura/forma, índice para alimentar el modelo, datatype).
    detalle_entrada = interpretador.get_input_details()
    detalle_salida = interpretador.get_output_details()

    # Al convertirlo en algo iterable evitamos errores si nos llega otro tipo de estructura...
    y_test = np.array(y_test)
    # Contador de aciertos y de críticas totales.
    correctas = 0
    total = len(x_test)

    # Bucle para inferir sobre la muestra de test y comprobar cuanta Accuracy presenta.
    for i in range(total):
        # Lista con una sola review transformada en números, 170 números.
        entrada = x_test[i:i+1]
        # Requerido por el modelo TFLite.
        entrada = entrada.astype(np.float32)

        # Le damos al modelo la entrada procesada para que infiera.
        interpretador.set_tensor(detalle_entrada[0]['index'], entrada)
        interpretador.invoke()
        # Obtenemos la inferencia.
        salida = interpretador.get_tensor(detalle_salida[0]['index'])[0][0]

        # 0 es negativo, 1 es positivo
        prob = int(salida > 0.5)  # Clasificación binaria. np.argmax(salida) para varias clases.
        if prob == y_test[i]:
            correctas += 1

    # Cálculo de precision e impresión de la misma y de tamaño de modelo a nivel archivo.
    precision = correctas / total
    tamaño_modelo_kb = os.path.getsize(rutaModelo) / 1024
    print(f"MODELO: {os.path.basename(rutaModelo)} | ACCURACY: {precision:.4f} ({correctas}/{total}) | TAMAÑO: {tamaño_modelo_kb:.1f} KB")


print(f"EVALUANDO MODELOS QUANTIZADOS")
evaluacion = modelo.evaluate(np.asarray(X_test), np.asarray(y_test))
print(f"MODELO: {os.path.basename('modelo_base.keras')} | ACCURACY: {round(evaluacion[1], 4)} | TAMAÑO: {os.path.getsize('modelo_base.keras') / 1024:.1f} KB")
evaluar_cuantizacion('modelo_DRQ.tflite', X_test, y_test)
evaluar_cuantizacion('modelo_F16Q.tflite', X_test, y_test)


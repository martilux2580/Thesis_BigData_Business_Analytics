import pandas as pd
import re
import os
import numpy as np
import math
import nltk
from nltk.corpus import stopwords
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
import tensorflow as tf
from tensorflow import keras

import tensorflow_model_optimization as tfmot
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

# <-------------------------------PRUNING------------------------------>

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

print("RESUMEN ESTRUCTURA MODELO")
modelo.summary()


print("INICIO DE TECNICAS DE PRUNING")
# APLICAR PRUNING SOBRE MODELO YA ENTRENADO
print("APLICANDO PRUNING SOBRE EL MODELO ENTRENADO")

# Rango de proporción de pesos a cero al principio y final. Por lo general, mejor entre 0.2 y 0.8.
# 25% de pesos podados inicialmente, no subirlo mucho ya que lo hacemos más pequeño y menos preciso.
pct_inicio = 0.2
# 85% de pesos podados al final.
pct_final = 0.8
# Cuántos epochs reentrenamos el modelo con pruning. Es para que se adapte el modelo cuando le has quitado pesos, no necesita muchos ya que ya está entrenado.
pruning_epochs = 3
# Paso en el que comienza el podado. Aplicamos podado después de que haya sido ya entrenado, en verdad estamos aplicándolo en los últimos 3 epochs.
begin_step = 0
# Pasos por epochs son cuanta muestra tienes entre cuantas muestras procesas a la vez. Es un batch procesado, cada vez que pasa se actualiza el modelo.
pasos_por_epochs = math.ceil(len(X_train) / batch)
# Paso en el que se para el podado, en este caso al final del último epoch de fine-tuning. Si hubiesen más pasos, los pesos podados se quedan como están, el resto se actualizan.
end_step = pasos_por_epochs * pruning_epochs
# Cuando se entrene el modelo se actualiza el paso en cada batch para saber cuándo aplicar o no la poda. Si no lo utilizamos no actualizará bien el sparsity y el podado no avanza.
# Por cada batch se actualiza el paso y se calcula el nuevo sparsity.
callbacks = [tfmot.sparsity.keras.UpdatePruningStep()]

# Necesario para aplicar Magnitude-Based Pruning de la libreria Tensorflow Model Optimization Toolkit.
# Magnitude Based Pruning considera que los pesos cercanos a cero no influyen lo suficiente en la salida del modelo como para que estén ahí aumentando la complejidad del mismo.
# Por este motivo, este tipo de Pruning desactiva (pone a cero a través de máscara) esos pesos con valores cercanos a cero y así reduce tamaño sin afectar a la precisión.
prune_low_magnitude = tfmot.sparsity.keras.prune_low_magnitude

# Aplicar podado al modelo base, con función polinómica para que se vaya aplicando con cierta progresión, versus el ConstantSparsity que lo aplica a partir de cierto step.
modelo_podado = prune_low_magnitude(to_prune = modelo, pruning_schedule = tfmot.sparsity.keras.PolynomialDecay(initial_sparsity = pct_inicio, final_sparsity = pct_final, begin_step = 0, end_step = end_step))

# Recompilar el modelo podado.
modelo_podado.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics=['accuracy'])

# Reentrenar el modelo con podado.
print("REENTRENAMIENTO DEL MODELO PODADO")
modelo_podado.fit(X_train, y_train, batch_size = batch, epochs = pruning_epochs, validation_split = 0.1, callbacks = callbacks)

# Comprobar el modelo con los datos de Test, asegurando que sean vectores.
# Obtenemos en una lista la pérdida y la accuracy, como se especificó en Compile().
print("COMPROBAR MODELO CON DATOS DE TEST")
evaluacion = modelo.evaluate(np.asarray(X_test), np.asarray(y_test))
print(f"Loss: {round(evaluacion[0], 4)}, accuracy: {round(evaluacion[1],2)*100}%")
evaluacion = modelo_podado.evaluate(np.asarray(X_test), np.asarray(y_test))
print(f"Loss: {round(evaluacion[0], 4)}, accuracy: {round(evaluacion[1],2)*100}%")

# Quitamos las estructuras de poda auxiliares, dejando el modelo listo para inferencia.
modelo_limpio_podado = tfmot.sparsity.keras.strip_pruning(modelo_podado)

# Comprobar modelos distintos
print("ID del modelo base:", id(modelo))
print("ID del modelo podado:", id(modelo_podado))

# Calcular parámetros en los modelos.
print("CALCULANDO COMPARACIÓN DE PARÁMETROS EN MODELOS")

# Función para contar los parámetros útiles y totales en el modelo podado, basado en el modelo base.
def contar_parametros(modelo):
    params_no_podados = 0
    params_totales = 0
    for capa in modelo.layers:
        for peso in capa.weights:
            # Si es un peso podado, en su nombre llevará mask, que indica que la máscara de podado está aplicada sobre el peso.
            if 'mask' not in peso.name:
                # Obtenemos el peso convertido en un array.
                valor = peso.numpy()
                # Todos los valores que tenga se cuentan como parámetros totales.
                params_totales += valor.size
                # Todos los valores que no sean ceros (si lo son entonces son podados) se tienen como parámetros útiles o no podados.
                params_no_podados += np.count_nonzero(valor)
    return params_no_podados, params_totales

# Contar parámetros para comparar podado.
params_utiles_podado, params_totales_base = contar_parametros(modelo_limpio_podado)
print(f"Parámetros útiles (no son cero) en modelo podado: {params_utiles_podado}")
print(f"Parámetros totales en modelo base: {params_totales_base}")
print(f"Proporción de podado (sparsity): {round(1 - params_utiles_podado / params_totales_base, 2) * 100}%.")

# Guardar modelo final podado, pesa menos que el original ya que hemos perdido detalle al pasar pesos cercanos a cero a cero.
print("GUARDANDO MODELO PODADO")
modelo_limpio_podado.save('modelo_podado.keras')


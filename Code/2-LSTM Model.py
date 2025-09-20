import pandas as pd
import re
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from nltk.corpus import stopwords
from tensorflow.keras.preprocessing.sequence import pad_sequences
#from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
#from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

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

# <-------------------------------MODELO<------------------------------>
# Creando modelo

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


# Construir el modelo
def crear_modelo_LSTM():
    # El modelo será una sucesión de capas de neuronas.
    modelo = Sequential()

    # Primer set de capas.
    # La primera capa construye los embeddings, genera matriz que contiene filas (vectores) para cada palabra.
    modelo.add(Embedding(tamanio_diccionario, tamaño_capa, input_length = longitudMax))
    # La segunda capa tendrá las neuronas de memoria LSTM.
    modelo.add(LSTM(tamaño_capa))
    # La tercera capa es de Regularización por Dropout, el 20% de las neuronas los ponemos a cero para evitar sobreaprendizaje en el entrenamiento.
    modelo.add(Dropout(0.2))  
    
    # La última capa sigmoide nos dará probabilidad, si cerca de 1 será positivo, si cerca de 0 será negativo.
    modelo.add(Dense(1, activation = 'sigmoid'))

    # Aglutinamos todas las capas y definimos cómo se evalúa el modelo.
    # Función de pérdida entropía cruzada binaria, funciona bien con clasificaciones binarias frente a MSE.
    # Optimizador ADAM, robusto.
    # Queremos que cuando se entrene nos vaya indicando la precision/accuracy sobre train y sobre validación.
    modelo.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics=['accuracy'])

    return modelo

# Obtener nuestro modelo.
print("CONSTRUIR EL MODELO")
modelo = crear_modelo_LSTM()


# Resumen de la estructura de nuestro modelo, nos da información de capas y parámetros.
print("RESUMEN ESTRUCTURA MODELO")
modelo.summary()


# Entrenar nuestro modelo compilado con datos de train, 10 vueltas a todos los datos y procesa batch muestras a la vez mientras entrena.
# Muchos epochs pueden sobreentrenar, si más batch es más rapido pero necesita memoria/procesamiento y si menos batch va más lento.
# Guardamos los resultados del entrenamiento en modelo_entrenado.
print("ENTRENAR MODELO")
modelo_entrenado = modelo.fit(X_train, y_train, batch_size = batch, epochs = epochs, validation_split = 0.1)


"""
epochs10, dim64, maxLen70, batch32  --> Test accuracy 8020: 0.8263, 0.99
epochs10, dim64, maxLen100, batch32 --> Test accuracy 8020: 0.8489, 0.99
epochs10, dim64, maxLen130, batch32 --> Test accuracy 8020: 0.8650, 0.99
epochs10, dim64, maxLen170, batch32 --> Test accuracy 8020: 0.8707, 0.99
epochs10, dim64, maxLen230, batch32 --> Test accuracy 8020: 0.8715, 0.99
epochs10, dim64, maxLen250, batch32 --> Test accuracy 8020: 0.8772, 0.98
epochs10, dim64, maxLen310, batch32 --> Test accuracy 8020: 0.8816, 0.95

epochs10, dim64, maxLen70, batch64  --> Test accuracy 8020: 0.8368, 0.99
epochs10, dim64, maxLen100, batch64 --> Test accuracy 8020: 0.8531, 0.99
epochs10, dim64, maxLen130, batch64 --> Test accuracy 8020: 0.8618, 0.99
epochs10, dim64, maxLen170, batch64 --> Test accuracy 8020: 0.8647, 0.99
epochs10, dim64, maxLen230, batch64 --> Test accuracy 8020: 0.8767, 0.99
epochs10, dim64, maxLen250, batch64 --> Test accuracy 8020: 0.8627, 0.98
epochs10, dim64, maxLen310, batch64 --> Test accuracy 8020: 0.8739, 0.97

epochs10, dim128, maxLen70, batch32  --> Test accuracy 8020: 0.8338, 0.99
epochs10, dim128, maxLen100, batch32 --> Test accuracy 8020: 0.8530, 0.99
epochs10, dim128, maxLen130, batch32 --> Test accuracy 8020: 0.8641, 0.99
epochs10, dim128, maxLen170, batch32 --> Test accuracy 8020: 0.8693, 0.99
epochs10, dim128, maxLen230, batch32 --> Test accuracy 8020: 0.8770, 0.99
epochs10, dim128, maxLen250, batch32 --> Test accuracy 8020: 0.8681, 0.99
epochs10, dim128, maxLen310, batch32 --> Test accuracy 8020: 0.8728, 0.99

epochs10, dim128, maxLen70, batch64  --> Test accuracy 8020: 0.8297, 0.99
epochs10, dim128, maxLen100, batch64 --> Test accuracy 8020: 0.8541, 0.99
epochs10, dim128, maxLen130, batch64 --> Test accuracy 8020: 0.8658, 0.99
epochs10, dim128, maxLen170, batch64 --> Test accuracy 8020: 0.8723, 0.99
epochs10, dim128, maxLen230, batch64 --> Test accuracy 8020: 0.8690, 0.99
epochs10, dim128, maxLen250, batch64 --> Test accuracy 8020: 0.8699, 0.99
epochs10, dim128, maxLen310, batch64 --> Test accuracy 8020: 0.8735, 0.98 
"""


# Gráfico de líneas con la evolución de la precision/accuracy.
print("GRÁFICO ACCURACY TRAIN VALIDATION")
plt.plot(modelo_entrenado.history['accuracy'])
plt.plot(modelo_entrenado.history['val_accuracy'])
plt.title('Precisión del modelo.')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()


# Comprobar el modelo con los datos de Test, asegurando que sean vectores.
# Obtenemos en una lista la pérdida y la accuracy, como se especificó en Compile().
print("COMPROBAR MODELO CON DATOS DE TEST")
evaluacion = modelo.evaluate(np.asarray(X_test), np.asarray(y_test))
print(f"Loss: {round(evaluacion[0], 4)}, accuracy: {round(evaluacion[1],2)*100}%")


# La predicción nos dará valores entre 0 y 1 por la salida sigmoide, lo redondeamos a 0 negativo o 1 positivo.
print("OBTENIENDO PREDICCIONES")
prediccion_probabilidades = modelo.predict(X_test)
# Del bool de si es > 0.5 nos dará true/false, que es 1/0 con el astype(), y que con flatten convetimos la lista de listas a lista.
prediccion = (prediccion_probabilidades > 0.5).astype(int).flatten()
realidad = y_test


print("MATRIZ DE CONFUSIÓN")
matriz = confusion_matrix(realidad, prediccion, labels = [0, 1]) # Añadimos labels para saberlas clases que se clasifican.
plt.figure(figsize = (12, 7))
graf = plt.subplot()
# Generamos la matriz con el valor en cada celda de la matriz (anot) en formato entero (fmt), en el plot lienzo (ax) y no otro, y sustituir el 0, 1 con los labels.
sns.heatmap(matriz, annot = True, ax = graf, fmt = 'g', xticklabels=["Negative", "Positive"], yticklabels=["Negative", "Positive"])
# Establecer los labels para la información del gráfico, y guardar el gráfico en el workdir antes de mostrarlo.
graf.set_xlabel('Predicción')
graf.set_ylabel('Realidad', size = 12)
graf.set_title('Matriz de confusión', size = 18) 
graf.xaxis.set_ticklabels(["negative","positive"], size = 7)
graf.yaxis.set_ticklabels(["negative","positive"], size = 7)
plt.savefig('matriz_confusion.png')
plt.show()


print("INFORME DE CLASIFICACIÓN FINAL")
informe = classification_report(realidad, prediccion)
print(informe)


print("GUARDANDO MODELO")
modelo.save('modelo_base.keras')
print("MODELO GUARDADO CORRECTAMENTE")


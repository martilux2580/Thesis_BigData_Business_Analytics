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

from tensorflow.keras import layers, losses, metrics, Model
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
import tensorflow as tf
import numpy as np


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
tamanio_capa = 128


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

# Cargando modelo profesor.
print("CARGANDO MODELO")
modelo = keras.models.load_model('modelo_base.keras')
print("MODELO CARGADO CORRECTAMENTE")
# Congelar pesos del modelo profesor
modelo.trainable = False

# <-------------------------------INICIO DEL BLOQUE DE KNOWLEDGE DISTILLATION------------------------------>
print("INICIO KNOWLEDGE DISTILLATION")
# Hyperparams de Destilación
valor_Alpha = 0.85
valor_Temperatura = 1.5


# Crear modelo estudiante más pequeño
# Misma estructura de capas que el modelo maestro, pero reducido.
print("CREANDO MODELO ESTUDIANTE")
def crear_modelo_estudiante():
    modelo = keras.Sequential()
    modelo.add(Embedding(tamanio_diccionario, int(tamanio_capa / 2), input_length = longitudMax))
    modelo.add(LSTM(int(tamanio_capa / 2)))
    modelo.add(Dropout(0.2))
    modelo.add(Dense(1, activation = 'sigmoid'))
    return modelo
estudiante = crear_modelo_estudiante()


# Subclase de la clase Model para entrenamiento con distilación.
# La idea es que al llamar a ciertas funciones de la clase Model las personalicemos para la destilación.
print("DEFINIENDO CLASE DE DISTILACION")
class Distiller(Model):
    # Constructor de la clase, requerirá como atributos de la clase un modelo maestro ya entreando y un modelo estudiante básico....
    def __init__(self, estudiante, maestro, alpha, temperatura):
        # Para que esta clase Distiller funcione, necesita el constructor de su clase padre (Model) inicializado con sus diferentes funciones, por eso lo llamamos.
        # Es como si declarasemos un objeto Model pero diciendo que "vamos a definir un objeto personalizado", al inicializar ese objeto Model se ponen a disposición sus distintas funciones.
        # De esta manera, declaramos un objeto Modelo que se asigna a esta clase Distiller, y que algunas de sus funciones las sobreescribiremos con nuestras definiciones de las mismas.
        super(Distiller, self).__init__()
        # Asignamos los atributos maestro y estudiante que nos mandan a la clase.
        self.maestro = maestro
        self.estudiante = estudiante
        # Asignamos hiperparámetros de la Destilación:
        # Asignar a Distiller el balance entre ponderar más Student Loss (aprender de los datos, valor 1) o Distillation Loss (aprender de la manera que trabaja el profesor, valor 0).
        self.alpha = alpha
        # Asignar a Distiller el valor de temperatura utilizados en la función softmax() para los logits (raw output del modelo), valor por defecto 1, con menos una respuesta sobresaldrá mucho, con más aplanan la curva de resultados.
        self.temperatura = temperatura

    # Sobreescribimos la funcion Model.compile() de Keras con la definición de esta función, a la que le pasamos diferentes parámetros de la destilación....
    def compile(self, optimizador, metrica, fnc_loss_rendimiento, fnc_loss_destilacion):
        # Asigna la función compile() de la clase padre a Distiller, asignando el optimizador y la metrica que hayan indicado, permite otros métodos como fit() y evaluate()
        super(Distiller, self).compile(optimizer = optimizador, metrics = metrica)
        # Asignar a Distiller el optimizador de rendimiento utilizado para los gradientes y la retropropagación.
        self.optimizador = optimizador
        # Asignar a Distiller la función de pérdida para comparar predicción del estudiante con la realidad.
        self.fnc_loss_rendimiento = fnc_loss_rendimiento
        # Asignar a Distiller la función de pérdida para comparar las predicciones del profesor con las del estudiante.
        self.fnc_loss_destilacion = fnc_loss_destilacion

    # Sobreescribimos la función de entrenamiento Model.train_step() de Keras utilizada para los batches (todos los datos que pasan a la vez antes de que se actualicen pesos).
    def train_step(self, batch_datos):
        # Divide el batch de datos en la entrada X y los resultados Y.
        x, y = batch_datos
        # Forward pass de la entrada al maestro sin que calcule gradientes, ya que el modelo está ya entrenado.
        preds_maestro = self.maestro(x, training = False)

        # Manejamos el contexto de predicción con GradientTape(), el cual graba las operaciones realizadas para poder recalcular los pesos más tarde.
        with tf.GradientTape() as tape:
            # Forward pass de la entrada al estudiante calculando gradientes en base a su salida.
            preds_estudiante = self.estudiante(x, training = True)
            # Calcular función de pérdida entre predicción del estudiante y realidad.
            loss_estudiante = self.fnc_loss_rendimiento(y, preds_estudiante)
            # Calcular función de pérdida de aprendizaje del profesor frente al estudiante, utilizando la temperatura y función softmax() aplicado a las clases (1, binaria en este caso), en vez de batches (0).
            loss_destilacion = self.fnc_loss_destilacion(tf.nn.softmax(preds_maestro / self.temperatura, axis=1), tf.nn.softmax(preds_estudiante / self.temperatura, axis = 1))
            # Calculamos la pérdida total combinando las dos pérdidas.
            loss_total = self.alpha * loss_estudiante + (1 - self.alpha) * loss_destilacion

        # Obtenemos los parámetros entrenables del estudiante para calcular los gradientes y actualizar estos parámetros.
        params_entrenables = self.estudiante.trainable_variables
        # Calculamos los gradientes teniendo en cuenta la pérdida y los parámetros entrenables.
        gradientes = tape.gradient(loss_total, params_entrenables)
        # Aplicamos los gradientes a esos parámetros entrenables, actualizando y haciendo que el modelo aprenda con la retropropagación.
        self.optimizador.apply_gradients(zip(gradientes, params_entrenables))
        # Actualizamos metricas según las prediciones realizadas, para saber el rendimiento del modelo.
        self.compiled_metrics.update_state(y, preds_estudiante)
        # Guardamos en un diccionario todas las métricas para ir actualizándolas según los epochs.
        metricas = {m.name: m.result() for m in self.metrics}
        # Añadimos a esos resultados la pérdida de rendimiento y de destilación.
        metricas.update({"student_loss": loss_estudiante, "distillation_loss": loss_destilacion})
        # Devuelve los resultados de las métricas actualizadas.
        return metricas

    # Sobreescribimos la función de entrenamiento Model.test_step() de Keras utilizada cuando se realiza la evaluación del modelo.
    def test_step(self, batch_data):
        # Divide el batch de datos en la entrada X y los resultados Y.
        x, y = batch_data
        # Predicción del estudiante sin que calcule gradientes, ya que el modelo está ya entrenado.
        pred_estudiante = self.estudiante(x, training = False)
        # Calcular función de pérdida de precisión.
        loss_estudiante = self.fnc_loss_rendimiento(y, pred_estudiante)
        # Actualización de métricas del estudiantes en base a sus predicciones.
        self.compiled_metrics.update_state(y, pred_estudiante)
        # Guardamos las métricas del estudiante con sus valores actuales.
        metricas = {m.name: m.result() for m in self.metrics}
        # Actualizamos las métricas con su nuevo valor según predicciones realizadas.
        metricas.update({"student_loss": loss_estudiante})
        # Devolvemos métricas actualizadas.
        return metricas


print("COMPILANDO Y ENTRENANDO DISTILLER")
# Crear clase Distiller con el estudiante y el profesor.
distiller = Distiller(estudiante, modelo, valor_Alpha, valor_Temperatura)
# Compilamos con mismas métricas que el profesor, para que sean comparables.
distiller.compile(optimizador = keras.optimizers.Adam(), metrica = [metrics.BinaryAccuracy()], fnc_loss_destilacion = losses.KLDivergence(), fnc_loss_rendimiento = losses.BinaryCrossentropy())
# Mostrar estructura del modelo estudiante.
print(estudiante.summary())
# Imprimir configuración de hiperparámetros.
print(f"\nEntrenando con alpha = {valor_Alpha}, temperatura = {valor_Temperatura}.")
# Entrenamiento del modelo distilado, se guarda en variable para ver más adelante las métricas en cada momento.
history = distiller.fit(X_train, y_train, epochs = epochs, batch_size = batch, validation_data = (X_test, y_test))


# Evaluaciones del modelo estudiante y profesor.
print("\nEVALUACION FINAL")
print("Modelo Profesor:")
teacher_eval = modelo.evaluate(X_test, y_test)

print("\nModelo Estudiante:")
student_eval = distiller.evaluate(X_test, y_test)

print(f"\nAccuracy Profesor: {teacher_eval[1]:.4f}.")
print(f"Accuracy Estudiante: {student_eval[0]:.4f}.")

# Guardar el modelo destilado.
print("GUARDANDO MODELO")
estudiante.save('modelo_destilado.keras')
print("MODELO GUARDADO CORRECTAMENTE")


"""
# PRUEBAS AUTOMATICAS CON DIFERENTES ALPHA Y TEMPERATURE
print("\nPRUEBAS AUTOMÁTICAS: ")
# Rango de valores para probar todos con todos, así como la lista que almacenará resultados.
alphas = [0.05, 0.15, 0.30, 0.50, 0.70, 0.85, 0.95]
temperaturas = [0.75, 1.00, 1.25, 1.50, 2.00, 3.00, 5.00]
resultados = []

# Doble bucle para probar combinaciones.
for alpha in alphas:
    for temperatura in temperaturas:
        print(f"\nEntrenando con alpha = {alpha}, temperatura = {temperatura}.")
        # Crear nuevo modelo estudiante para cada prueba.
        estudiante = crear_modelo_estudiante()
        # Crear distiller.
        distiller = Distiller(estudiante, modelo, alpha, temperatura)
        distiller.compile(optimizador = keras.optimizers.Adam(), metrica = [metrics.BinaryAccuracy()], fnc_loss_rendimiento = losses.BinaryCrossentropy(), fnc_loss_destilacion = losses.KLDivergence())
        # Entrenamiento.
        distiller.fit(X_train, y_train, epochs=epochs, batch_size=batch, validation_data=(X_test, y_test), verbose = 0)
        # Evaluación.
        eval_result = distiller.evaluate(X_test, y_test, verbose = 0)
        binary_accuracy = eval_result[0]
        student_loss = eval_result[1]
        resultados.append((alpha, temperatura, student_loss, binary_accuracy))
        # Imprimir resultados.
        print(f"Alpha: {alpha}, Temperatura: {temperatura}, Loss: {student_loss:.4f}, Accuracy: {binary_accuracy:.4f}.")


# Ordenar resultados por accuracy descendente, guardándolo en la misma variable.
resultados.sort(key=lambda x: x[3], reverse=True)


# Mostrar resumen final ordenado de mayor a menor accuracy.
print("\nRESUMEN DE PRUEBAS ORDENADO POR ACCURACY")
for alpha, temperatura, loss, accuracy in resultados:
    print(f"alpha = {alpha}, temperatura = {temperatura} --> Loss = {loss:.4f}, Accuracy = {accuracy:.4f}.")


# Mostrar setup con mejor accuracy.
final_alpha, final_temp, final_loss, final_accuracy = resultados[0]
print(f"\nMEJOR CONFIGURACION ({epochs} epochs): alpha = {final_alpha}, temperatura = {final_temp} --> Accuracy = {final_accuracy:.4f}.")
"""


"""
RESUMEN DE PRUEBAS ORDENADO POR ACCURACY (16 horas de entreno): 
alpha = 0.95, temperatura = 1.5 --> Loss = 0.2323, Accuracy = 0.8840.
alpha = 0.5, temperatura = 1.25 --> Loss = 0.0986, Accuracy = 0.8823.
alpha = 0.15, temperatura = 5.0 --> Loss = 0.4537, Accuracy = 0.8789.

alpha = 0.7, temperatura = 1.0 --> Loss = 0.1642, Accuracy = 0.8758.
alpha = 0.7, temperatura = 1.0 --> Loss = 0.1642, Accuracy = 0.8758.
alpha = 0.15, temperatura = 1.5 --> Loss = 0.2828, Accuracy = 0.8748.
alpha = 0.85, temperatura = 3.0 --> Loss = 0.3330, Accuracy = 0.8737.
alpha = 0.3, temperatura = 1.5 --> Loss = 0.2235, Accuracy = 0.8728.
alpha = 0.15, temperatura = 1.0 --> Loss = 0.6445, Accuracy = 0.8700.
alpha = 0.95, temperatura = 1.25 --> Loss = 0.1788, Accuracy = 0.8696.
alpha = 0.3, temperatura = 3.0 --> Loss = 0.3343, Accuracy = 0.8691.
alpha = 0.5, temperatura = 3.0 --> Loss = 0.6469, Accuracy = 0.8678.
alpha = 0.7, temperatura = 2.0 --> Loss = 0.1473, Accuracy = 0.8678.
alpha = 0.85, temperatura = 2.0 --> Loss = 0.2610, Accuracy = 0.8624.
alpha = 0.05, temperatura = 0.75 --> Loss = 0.3363, Accuracy = 0.8616.
alpha = 0.3, temperatura = 5.0 --> Loss = 0.2701, Accuracy = 0.8610.
alpha = 0.95, temperatura = 2.0 --> Loss = 0.5474, Accuracy = 0.8609.
alpha = 0.3, temperatura = 2.0 --> Loss = 0.5058, Accuracy = 0.8606.
alpha = 0.5, temperatura = 1.5 --> Loss = 0.1725, Accuracy = 0.8592.
alpha = 0.85, temperatura = 1.25 --> Loss = 0.2598, Accuracy = 0.8545.
alpha = 0.05, temperatura = 1.0 --> Loss = 0.3358, Accuracy = 0.8521.
alpha = 0.85, temperatura = 1.0 --> Loss = 0.2163, Accuracy = 0.8519.
alpha = 0.3, temperatura = 1.0 --> Loss = 0.3881, Accuracy = 0.8492.
alpha = 0.5, temperatura = 0.75 --> Loss = 0.3637, Accuracy = 0.8485.
alpha = 0.15, temperatura = 2.0 --> Loss = 0.5343, Accuracy = 0.8480.
alpha = 0.95, temperatura = 1.0 --> Loss = 0.3972, Accuracy = 0.8460.
alpha = 0.7, temperatura = 1.25 --> Loss = 0.3119, Accuracy = 0.8455.
alpha = 0.05, temperatura = 5.0 --> Loss = 0.2755, Accuracy = 0.8454.
alpha = 0.05, temperatura = 3.0 --> Loss = 0.2046, Accuracy = 0.8447.
alpha = 0.7, temperatura = 5.0 --> Loss = 0.3337, Accuracy = 0.8442.
alpha = 0.5, temperatura = 5.0 --> Loss = 0.6770, Accuracy = 0.8436.
alpha = 0.3, temperatura = 0.75 --> Loss = 0.3735, Accuracy = 0.8415.
alpha = 0.15, temperatura = 3.0 --> Loss = 0.3898, Accuracy = 0.8400.
alpha = 0.3, temperatura = 1.25 --> Loss = 0.3194, Accuracy = 0.8320.
alpha = 0.7, temperatura = 3.0 --> Loss = 0.3794, Accuracy = 0.8216.
alpha = 0.95, temperatura = 0.75 --> Loss = 0.2533, Accuracy = 0.8152.
alpha = 0.85, temperatura = 0.75 --> Loss = 0.3073, Accuracy = 0.8120.
alpha = 0.5, temperatura = 1.0 --> Loss = 0.4859, Accuracy = 0.8065.
alpha = 0.05, temperatura = 2.0 --> Loss = 0.4344, Accuracy = 0.8042.
alpha = 0.85, temperatura = 5.0 --> Loss = 0.3662, Accuracy = 0.8002.
alpha = 0.7, temperatura = 0.75 --> Loss = 0.2802, Accuracy = 0.7905.
alpha = 0.05, temperatura = 1.5 --> Loss = 0.3990, Accuracy = 0.7894.
alpha = 0.05, temperatura = 1.25 --> Loss = 0.2949, Accuracy = 0.7848.
alpha = 0.5, temperatura = 2.0 --> Loss = 0.3154, Accuracy = 0.7693.
alpha = 0.95, temperatura = 5.0 --> Loss = 0.5802, Accuracy = 0.6507.
alpha = 0.95, temperatura = 3.0 --> Loss = 0.6328, Accuracy = 0.5385.
alpha = 0.15, temperatura = 1.25 --> Loss = 0.6089, Accuracy = 0.5351.
alpha = 0.15, temperatura = 0.75 --> Loss = 0.6199, Accuracy = 0.5319.
alpha = 0.85, temperatura = 1.5 --> Loss = 0.6887, Accuracy = 0.5025.

MEJOR CONFIGURACION (12 epochs): alpha = 0.95, temperatura = 1.5 --> Accuracy = 0.8840.
"""


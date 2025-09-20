import pandas as pd
import re
import os
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from wordcloud import WordCloud
import matplotlib.pyplot as plt


#Cambiar el workdir
os.chdir('C:/Users/Martin/Desktop/TFM/PythonStuff/')


# Configurar pandas
#pd.set_option('display.max_columns', None)
#pd.set_option('display.max_colwidth', None)  # Or use a large number like 1000 if you're on older pandas
#pd.set_option('display.expand_frame_repr', False)  # Prevent wrapping across multiple lines


# Cargar los datos en base a dónde estás en CMD
print('CARGAR DATOS')
df = pd.read_csv('C:/Users/Martin/Desktop/TFM/PythonStuff/Data/IMDB Dataset.csv')
print(df.head(3))

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
print(df.head(3))


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


print('A BIT OF EDA')
# Un poco de EDA.
# Contar nulos en el dataframe.
print('CUANTOS NULOS')
print(df.isnull().sum())
# Datos genéricos del dataframe.
print('DESCRIPTIVO DATOS')
print(df.describe())
# Palabras por review (nos vendrá bien para el diagrama Wordcloud)
print('CONTEO PALABRAS')
df['word_count'] = df['review'].apply(lambda x: len(x.split()))  
print(df.head(3))
# Para cada sentimiento (positive, negative), sacar la media, mínimo y máximo de palabras.
stats_Palabras = df.groupby('sentiment')['word_count'].agg(['mean', 'min', 'max'])  
print(stats_Palabras)
# Añadir en una nueva columna la longitud en caracteres de la review.
print('LONGITUD CARACTERES REVIEW')
df['review_length'] = df['review'].apply(len)
print(df.head(3))
# Para cada sentimiento (positive, negative), sacar la media, mínimo y máximo de caracteres.
stats_Longitud = df.groupby('sentiment')['review_length'].agg(['mean', 'min', 'max'])  
print(stats_Longitud)
# Percentiles de 5 en 5, obtenemos las palabras y las contamos, y con quantile sacamos los percentiles que queremos de ese grupo de documentos.
print("Percentiles del 1 al 100 en 5")
print(df["review"].str.split().str.len().quantile([i/100 for i in range(0, 101, 5)]))
# Cantidad de caracteres únicos, juntamos todas las reviews en un texto largo sin espacios, y lo transformamos en un conjunto (no admite dupps).
caracteres_unicos = set(''.join(df['review'].astype(str)))
print(f"Cantidad de caracteres únicos: {len(caracteres_unicos)}")


# Obtener TFIDF (Term Frequency qué tan frecuente, IDF qué tan única es con respecto a todos los documentos, más alto significa más importante o que contiene/aporta mayor significado)
# Inicializamos el objeto que luego convertirá nuestros documentos en un vector de números.
vectorizador = TfidfVectorizer(stop_words='english', max_features=1000)
# Matriz de filas los docs y columnas las palabras, valores son el valor TFIDF.
matriz = vectorizador.fit_transform(df['review'])
# Para ponerle a las columnas el nombre de las palabras.
nombre_columnas_tfidf = vectorizador.get_feature_names_out()
# Pasar la matriz a un dataframe donde de columnas tenemos los features. Pasamos las filas del doc a array ya aue la matriz es sparse (no guarda ceros por eficiencia), así podemos convertir.
matrizDf = pd.DataFrame(matriz.toarray(), columns=nombre_columnas_tfidf)
# De cada feature sacamos el TFIDF medio, computandolo en base a su valor en cada doc.
puntuaciones_medias = matrizDf.mean().sort_values(ascending=False)
# Obtenemos los 10 features con mayor TFIDF
print('FEATURES CON MAYOR SCORE DE TFIDF')
print(puntuaciones_medias.head(10))


# Gráficos para longitud de las reviews y cantidad de palabras
# Para cada sentimiento
for sentiment in ['positive', 'negative']:
    # Obtenemos los datos para ese sentimiento en un nuevo df.
    df_sent = df[df['sentiment'] == sentiment]
    
    print('HISTOGRAMA LONGITUD REVIEW: ' + sentiment)
    # Histograma para longitud de la review.
    df_sent['review_length'].hist()
    plt.title(f'Review Length - {sentiment.capitalize()}') # positive --> Positive
    plt.xlabel('Length')
    plt.ylabel('Frequency')
    plt.show()
    
    print('HISTOGRAMA CANTIDAD DE PALABRAS REVIEW: ' + sentiment)
    # Histograma para cantidad de palabras de la review.
    df_sent['word_count'].hist()
    plt.title(f'Word Count - {sentiment.capitalize()}') # positive --> Positive
    plt.xlabel('Word Count')
    plt.ylabel('Frequency')
    plt.show()


# Wordcloud según sentiment.
# Obtener un string con todas las palabras de ese sentimiento separado por espaacios.
palabras_negativo = ' '.join(df[df['sentiment'] == 'negative']['review'])
palabras_positivo = ' '.join(df[df['sentiment'] == 'positive']['review'])

# Generar gráficos Wordcloud
print('WORDCLOUD: NEGATIVE')
WordCloud(width=800, height=400, background_color='white').generate(palabras_negativo).to_image().show()
print('WORDCLOUD: POSITIVE')
WordCloud(width=800, height=400, background_color='white').generate(palabras_positivo).to_image().show()


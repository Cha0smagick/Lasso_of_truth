import os
import nltk
from textblob import TextBlob
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.probability import FreqDist
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import seaborn as sns

# Descargar recursos de NLTK para el idioma español
nltk.download('punkt')
nltk.download('stopwords')

# Solicitar al usuario la ruta completa del archivo .txt
ruta_archivo = input("Por favor, ingresa la ruta completa del archivo .txt: ")

# Verificar si el archivo existe
if not os.path.isfile(ruta_archivo):
    print("El archivo especificado no existe.")
    exit()

# Leer el contenido del archivo
with open(ruta_archivo, 'r', encoding='utf-8') as archivo:
    texto = archivo.read()

# Preprocesamiento de texto
texto = texto.lower()
palabras = word_tokenize(texto, language='spanish')
palabras = [palabra for palabra in palabras if palabra.isalnum()]  # Eliminar caracteres no alfanuméricos

# Eliminar stopwords
stop_words = set(stopwords.words('spanish'))
palabras = [palabra for palabra in palabras if palabra not in stop_words]

# Análisis de sentimiento
analisis = TextBlob(texto)

# Extracción de términos clave
frecuencia = FreqDist(palabras)
palabras_clave = frecuencia.most_common(10)  # Tomar las 10 palabras más comunes

# Análisis de estructura
frases = sent_tokenize(texto)
num_frases = len(frases)

# Calcular la polaridad promedio de las oraciones
polaridades = [sent.sentiment.polarity for sent in analisis.sentences]
polaridad_promedio = sum(polaridades) / len(polaridades)

# Calcular la subjetividad promedio de las oraciones
subjetividades = [sent.sentiment.subjectivity for sent in analisis.sentences]
subjetividad_promedio = sum(subjetividades) / len(subjetividades)

# Calcular la longitud promedio de los párrafos
longitudes_parrafos = [len(frase.split()) for frase in frases]
longitud_promedio = sum(longitudes_parrafos) / len(longitudes_parrafos)

# Calcular un score de tono basado en las métricas
score_tono = (polaridad_promedio + 1) * (1 - subjetividad_promedio) * (1 / (longitud_promedio + 1))

# Visualización de resultados
plt.figure(figsize=(12, 12))

# Gráfico de barras para palabras clave
plt.subplot(2, 2, 1)
plt.bar(*zip(*palabras_clave))
plt.title('Palabras Clave')
plt.xticks(rotation=45)
plt.xlabel('Palabra')
plt.ylabel('Frecuencia')

# Nube de palabras
wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(frecuencia)
plt.subplot(2, 2, 2)
plt.imshow(wordcloud, interpolation='bilinear')
plt.title('Nube de Palabras')
plt.axis('off')

# Gráfico de barras para longitud de párrafos
plt.subplot(2, 2, 3)
plt.axhline(y=longitud_promedio, color='red', linestyle='--', label='Longitud Promedio')
sns.barplot(x=list(range(1, num_frases+1)), y=longitudes_parrafos, palette="viridis")
plt.title('Longitud de Párrafos')
plt.xlabel('Párrafo')
plt.ylabel('Longitud de Palabras')
plt.legend()

# Histograma de polaridad de oraciones
plt.subplot(2, 2, 4)
sns.histplot(polaridades, kde=True, color="skyblue")
plt.title('Distribución de Polaridad de Oraciones')
plt.xlabel('Polaridad')
plt.ylabel('Frecuencia')

# Ajustar el espaciado entre subplots
plt.tight_layout()

# Mostrar análisis de sentimiento
descripcion_polaridad = "La polaridad mide el tono del texto, donde -1 es muy negativo, 0 es neutro y 1 es muy positivo."
descripcion_subjetividad = "La subjetividad mide la objetividad del texto, donde 0 es muy objetivo y 1 es muy subjetivo."
descripcion_longitud = "La longitud promedio de los párrafos se compara con el tamaño promedio en un texto típico."

print("\nAnálisis de Sentimiento:")
print("Polaridad Promedio de Oraciones:", polaridad_promedio)
print("Subjetividad Promedio de Oraciones:", subjetividad_promedio)
print("Longitud Promedio de Párrafos:", longitud_promedio)
print("\nDescripciones:")
print(descripcion_polaridad)
print(descripcion_subjetividad)
print(descripcion_longitud)

# Definir el tono basado en el score
if score_tono > 0.7:
    tono = "Muy Positivo"
elif score_tono > 0.4:
    tono = "Positivo"
elif score_tono > 0.2:
    tono = "Neutro Positivo"
elif score_tono > -0.2:
    tono = "Neutro"
elif score_tono > -0.4:
    tono = "Neutro Negativo"
elif score_tono > -0.7:
    tono = "Negativo"
else:
    tono = "Muy Negativo"

# Conclusión con precisión de 5 dígitos en el score
conclusion = f"El texto tiene un tono {tono} con un score de tono de {score_tono:.5f}."

print("\nConclusión:")
print(conclusion)

# Mostrar la figura
plt.show()

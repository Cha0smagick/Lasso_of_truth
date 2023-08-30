import os
import nltk
from textblob import TextBlob
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.probability import FreqDist
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import seaborn as sns
import numpy as np

# Crear la carpeta "resultados" si no existe
if not os.path.exists("resultados"):
    os.mkdir("resultados")

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
polaridad_promedio = round(sum(polaridades) / len(polaridades), 5)  # Redondear a 5 decimales

# Calcular la subjetividad promedio de las oraciones
subjetividades = [sent.sentiment.subjectivity for sent in analisis.sentences]
subjetividad_promedio = round(sum(subjetividades) / len(subjetividades), 5)  # Redondear a 5 decimales

# Calcular la longitud promedio de los párrafos
longitudes_parrafos = [len(frase.split()) for frase in frases]
longitud_promedio = round(sum(longitudes_parrafos) / len(longitudes_parrafos), 5)  # Redondear a 5 decimales

# Calcular un score de tono basado en las métricas
score_tono = round((polaridad_promedio + 1) * (1 - subjetividad_promedio) * (1 / (longitud_promedio + 1)), 5)  # Redondear a 5 decimales

# Carpeta para los resultados
if not os.path.exists("resultados/imagenes"):
    os.mkdir("resultados/imagenes")

# Gráfico de barras para palabras clave y guardar como imagen independiente
plt.figure(figsize=(8, 6))
sns.barplot(x=[word for word, freq in palabras_clave], y=[freq for word, freq in palabras_clave], palette="muted")
plt.title('Palabras Clave')
plt.xticks(rotation=45)
plt.xlabel('Palabra')
plt.ylabel('Frecuencia')
plt.savefig('resultados/imagenes/palabras_clave.png', bbox_inches='tight')  # Guardar la imagen

# Nube de palabras y guardar como imagen independiente
wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(frecuencia)
plt.figure(figsize=(10, 6))
plt.imshow(wordcloud, interpolation='bilinear')
plt.title('Nube de Palabras')
plt.axis('off')
plt.savefig('resultados/imagenes/nube_palabras.png', bbox_inches='tight')  # Guardar la imagen

# Gráfico de barras para longitud de párrafos y guardar como imagen independiente
plt.figure(figsize=(8, 6))
sns.barplot(x=list(range(1, num_frases+1)), y=longitudes_parrafos, palette="Blues_d")
plt.axhline(y=longitud_promedio, color='red', linestyle='--', label='Longitud Promedio')
plt.title('Longitud de Párrafos')
plt.xlabel('Párrafo')
plt.ylabel('Longitud de Palabras')
plt.legend()
plt.savefig('resultados/imagenes/longitud_parrafos.png', bbox_inches='tight')  # Guardar la imagen

# Histograma de polaridad de oraciones y guardar como imagen independiente
plt.figure(figsize=(8, 6))
sns.histplot(polaridades, kde=True, color="skyblue")
plt.axvline(x=polaridad_promedio, color='green', linestyle='-', label='Media')
plt.title('Distribución de Polaridad de Oraciones')
plt.xlabel('Polaridad')
plt.ylabel('Frecuencia')
plt.legend()
plt.savefig('resultados/imagenes/histograma_polaridad.png', bbox_inches='tight')  # Guardar la imagen

# Diagrama de dispersión de palabras clave
# Mostrar la ubicación de las palabras clave en el texto
plt.figure(figsize=(10, 6))
for palabra, _ in palabras_clave:
    posiciones = [i for i, x in enumerate(palabras) if x == palabra]
    plt.scatter(posiciones, [frecuencia[palabra]] * len(posiciones), label=palabra, alpha=0.5, marker='o')
plt.title('Diagrama de Dispersión de Palabras Clave')
plt.xlabel('Posición en el Texto')
plt.ylabel('Frecuencia')
plt.legend()
plt.savefig('resultados/imagenes/diagrama_dispersión_palabras_clave.png', bbox_inches='tight')

# Gráfico de líneas de frecuencia de palabras a lo largo del texto
frecuencia_palabras = [frecuencia[palabra] for palabra, _ in palabras_clave]
posiciones_palabras = [pos[0] for pos in palabras_clave]
plt.figure(figsize=(10, 6))
plt.plot(posiciones_palabras, frecuencia_palabras, marker='o', linestyle='-', color='green')
plt.title('Gráfico de Frecuencia de Palabras Clave a lo Largo del Texto')
plt.xlabel('Posición en el Texto')
plt.ylabel('Frecuencia')
plt.xticks(posiciones_palabras, [palabra for palabra, _ in palabras_clave], rotation=45)
plt.tight_layout()
plt.savefig('resultados/imagenes/grafico_frecuencia_palabras.png', bbox_inches='tight')

# Mapa de calor de la matriz de co-ocurrencia
# Crear una matriz de co-ocurrencia
co_ocurrencia_matrix = np.zeros((len(palabras_clave), len(palabras_clave)))
for i, (palabra1, _) in enumerate(palabras_clave):
    for j, (palabra2, _) in enumerate(palabras_clave):
        co_ocurrencia = sum(1 for k in range(len(palabras) - 1) if palabras[k] == palabra1 and palabras[k + 1] == palabra2)
        co_ocurrencia_matrix[i][j] = co_ocurrencia

# Crear el mapa de calor
plt.figure(figsize=(8, 8))
sns.heatmap(co_ocurrencia_matrix, annot=True, fmt='.0f', xticklabels=[palabra for palabra, _ in palabras_clave],
            yticklabels=[palabra for palabra, _ in palabras_clave], cmap="YlGnBu")
plt.title('Mapa de Calor de Co-ocurrencia de Palabras Clave')
plt.savefig('resultados/imagenes/mapa_calor_coocurrencia.png', bbox_inches='tight')

# Gráfico de densidad de polaridad
plt.figure(figsize=(8, 6))
sns.kdeplot(polaridades, shade=True, color="orange")
plt.axvline(x=polaridad_promedio, color='green', linestyle='-', label='Media')
sns.kdeplot(polaridades, shade=True, color="red", linestyle='--', label='Tendencia')
plt.title('Gráfico de Densidad de Polaridad')
plt.xlabel('Polaridad')
plt.ylabel('Densidad')
plt.legend()
plt.savefig('resultados/imagenes/grafico_densidad_polaridad.png', bbox_inches='tight')

# Ajustar el espaciado entre subplots
plt.tight_layout()

# Calcular la conclusión
if score_tono > 0.7:
    tono = "Muy Positivo"
elif score_tono > 0.4:
    tono = "Positivo"
elif score_tono > 0.2:
    tono = "Levemente Positivo"
elif score_tono > -0.2:
    tono = "Neutro"
elif score_tono > -0.4:
    tono = "Levemente Negativo"
elif score_tono > -0.7:
    tono = "Negativo"
else:
    tono = "Muy Negativo"

conclusion = f"El tono del texto es {tono} con un score de tono de {score_tono}."

# Guardar análisis de sentimiento en un archivo .txt
with open('resultados/analisis_sentimiento.txt', 'w', encoding='utf-8') as archivo_txt:
    archivo_txt.write("Análisis de Sentimiento:\n")
    archivo_txt.write(f"Polaridad Promedio de Oraciones: {polaridad_promedio}\n")
    archivo_txt.write(f"Subjetividad Promedio de Oraciones: {subjetividad_promedio}\n")
    archivo_txt.write(f"Longitud Promedio de Párrafos: {longitud_promedio}\n\n")
    archivo_txt.write("Descripciones:\n")
    archivo_txt.write("La polaridad mide el tono del texto, donde -1 es muy negativo, 0 es neutro y 1 es muy positivo.\n")
    archivo_txt.write("La subjetividad mide la objetividad del texto, donde 0 es muy objetivo y 1 es muy subjetivo.\n")
    archivo_txt.write("La longitud promedio de los párrafos se compara con el tamaño promedio en un texto típico.\n\n")
    archivo_txt.write("Conclusión:\n")
    archivo_txt.write(conclusion)

# Mostrar la figura
plt.show()

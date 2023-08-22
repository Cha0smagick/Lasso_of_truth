# Lasso_of_truth

# Herramienta de Análisis Cualitativo de Texto en Python para Investigadores Sociales

Esta herramienta de análisis cualitativo de texto en Python ha sido diseñada específicamente para investigadores sociales y antropólogos interesados en evaluar y comprender el contenido textual de manera profunda. Proporciona una serie de capacidades analíticas, incluyendo la medición de polaridad y subjetividad, la identificación de palabras clave y la generación de un tono general basado en métricas de análisis de texto. Este recurso resulta valioso para el análisis de discursos, entrevistas, transcripciones de grupos focales y otros datos cualitativos en investigación social y antropológica.
Características Destacadas

1. Análisis de Sentimiento

La herramienta calcula la polaridad y subjetividad del texto mediante la biblioteca NLTK y TextBlob. La polaridad mide el tono del texto, donde valores negativos indican negatividad, cero denota neutralidad y valores positivos señalan positividad. La subjetividad mide el grado de subjetividad u objetividad del texto, con 0 representando máxima objetividad y 1 máxima subjetividad.

2. Extracción de Palabras Clave

Identifica y presenta las 10 palabras más frecuentes en el texto analizado. Esta característica es particularmente útil para identificar temas clave y conceptos predominantes en el corpus textual.

3. Visualización de Datos

La herramienta utiliza las bibliotecas Matplotlib y Seaborn para crear gráficos que facilitan la comprensión de los datos. Incluye un gráfico de barras para palabras clave, una nube de palabras y un gráfico de barras para la longitud de los párrafos.

# Instalación

Para utilizar esta herramienta, sigue estos pasos:

Clona este repositorio en tu máquina local usando git clone o descarga el archivo ZIP.
Descomprime el archivo de ser necesario y navega al directorio del repositorio:

    cd Lasso_of_truth-main

Instala las dependencias utilizando pip y el archivo requirements.txt:

    pip install -r requirements.txt

Uso

Una vez instaladas las dependencias, sigue estos pasos:

  Asegúrate de estar en el directorio del repositorio.

  Ejecuta el programa con el comando:

    python lasso_of_truth.py

  La herramienta te solicitará la ruta completa del archivo .txt que deseas analizar. Proporciona la ruta y espera a que se complete el análisis.

Resultados

La herramienta mostrará gráficos y un resumen detallado del análisis de texto. Además, genera un tono general del texto, lo que facilita la evaluación cualitativa del contenido como positivo, neutral o negativo. Esto resulta particularmente útil para los investigadores sociales que desean categorizar y evaluar el tono de sus datos cualitativos.



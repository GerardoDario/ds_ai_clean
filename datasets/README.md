# Datasets para IA y Machine Learning

Este directorio contiene enlaces y referencias a datasets populares para proyectos de IA.

## 🌐 Repositorios de Datasets

### Plataformas Generales
- **[Kaggle Datasets](https://www.kaggle.com/datasets)**: Miles de datasets, notebooks y competiciones
- **[UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/)**: Clásico repositorio académico
- **[Google Dataset Search](https://datasetsearch.research.google.com/)**: Buscador de datasets
- **[AWS Open Data](https://registry.opendata.aws/)**: Datasets en la nube de AWS
- **[Papers With Code Datasets](https://paperswithcode.com/datasets)**: Datasets de papers de investigación
- **[Hugging Face Datasets](https://huggingface.co/datasets)**: Especializado en NLP y ML

### Específicos de Gobierno
- **[Data.gov](https://www.data.gov/)**: Datos del gobierno de EE.UU.
- **[EU Open Data Portal](https://data.europa.eu/)**: Datos de la Unión Europea
- **[World Bank Open Data](https://data.worldbank.org/)**: Datos económicos globales

## 📊 Datasets por Categoría

### Computer Vision

#### Clasificación de Imágenes
- **ImageNet**: 14M imágenes, 1000 clases
  - [Website](http://www.image-net.org/)
  - Uso: Transfer learning, benchmarking
  
- **CIFAR-10/CIFAR-100**: 60K imágenes pequeñas
  - 10 o 100 clases
  - Uso: Proyectos educativos, benchmarking

- **MNIST**: 70K dígitos escritos a mano
  - [Website](http://yann.lecun.com/exdb/mnist/)
  - Uso: Introducción a clasificación

- **Fashion-MNIST**: 70K imágenes de ropa
  - Alternativa más desafiante a MNIST
  
- **Tiny ImageNet**: Subset de ImageNet
  - 200 clases, imágenes más pequeñas

#### Detección de Objetos
- **MS COCO** (Common Objects in Context)
  - [Website](https://cocodataset.org/)
  - 330K imágenes, 80 categorías de objetos
  - Annotations: bounding boxes, segmentación

- **Pascal VOC**: 20 clases de objetos
  - [Website](http://host.robots.ox.ac.uk/pascal/VOC/)
  - Benchmark clásico

- **Open Images Dataset**: 9M imágenes
  - [Website](https://storage.googleapis.com/openimages/web/index.html)
  - 600 categorías de objetos

#### Rostros
- **CelebA**: 200K imágenes de celebridades
  - 40 atributos por imagen
  - Uso: Face recognition, attribute prediction

- **LFW** (Labeled Faces in the Wild): 13K imágenes
  - Benchmark para face verification

- **WIDER FACE**: 32K imágenes para detección de rostros
  - Diferentes escalas y oclusiones

#### Segmentación
- **Cityscapes**: 5K imágenes de conducción urbana
  - [Website](https://www.cityscapes-dataset.com/)
  - Segmentación semántica de alta calidad

- **ADE20K**: 25K imágenes de escenas
  - 150 categorías semánticas

- **Mapillary Vistas**: Street-level imagery
  - 25K imágenes de alta resolución

#### Médicas
- **ChestX-ray14**: 112K imágenes de rayos X
  - 14 enfermedades
  - [Website](https://nihcc.app.box.com/v/ChestXray-NIHCC)

- **ISIC**: Imágenes de lesiones de piel
  - [Website](https://www.isic-archive.com/)
  - Detección de melanoma

- **BraTS**: Imágenes de resonancia magnética cerebral
  - Segmentación de tumores

### Natural Language Processing

#### Corpus Generales
- **Common Crawl**: Petabytes de datos web
  - [Website](https://commoncrawl.org/)
  - Uso: Pre-entrenamiento de LLMs

- **Wikipedia Dumps**: Artículos completos
  - Múltiples idiomas
  - [Website](https://dumps.wikimedia.org/)

- **BookCorpus**: 11K libros
  - Uso: Pre-entrenamiento (BERT, GPT)

- **OpenWebText**: Reproducción de WebText
  - [GitHub](https://github.com/jcpeterson/openwebtext)

#### Clasificación y Sentimientos
- **IMDB Movie Reviews**: 50K reviews de películas
  - Sentimiento binario (positivo/negativo)
  - [Website](https://ai.stanford.edu/~amaas/data/sentiment/)

- **SST** (Stanford Sentiment Treebank): 11K frases
  - 5 niveles de sentimiento
  - Análisis fino de sentimiento

- **Yelp Reviews**: Millones de reviews de negocios
  - Ratings de 1-5 estrellas

- **AG News**: 120K artículos de noticias
  - 4 categorías

- **20 Newsgroups**: 20K documentos
  - 20 categorías de temas

#### Question Answering
- **SQuAD** (Stanford QA Dataset): 100K pares Q&A
  - [Website](https://rajpurkar.github.io/SQuAD-explorer/)
  - Versiones 1.1 y 2.0

- **Natural Questions**: 300K preguntas de Google
  - [Website](https://ai.google.com/research/NaturalQuestions)

- **MS MARCO**: 1M queries de Bing
  - Passage ranking y QA

- **TriviaQA**: 95K pares de trivia

#### Traducción
- **WMT** (Workshop on Machine Translation)
  - [Website](http://www.statmt.org/wmt/)
  - Múltiples pares de idiomas

- **Europarl**: Corpus paralelo del Parlamento Europeo
  - 21 idiomas europeos

- **OpenSubtitles**: Subtítulos de películas
  - 60+ idiomas

#### Named Entity Recognition
- **CoNLL-2003**: Benchmark estándar de NER
  - Inglés y alemán
  - 4 tipos de entidades

- **OntoNotes**: Corpus multi-dominio
  - 18 tipos de entidades

#### Diálogo y Conversación
- **Ubuntu Dialogue Corpus**: 1M conversaciones
  - Support técnico

- **PersonaChat**: Conversaciones con personalidades
  - Chitchat

- **MultiWOZ**: 10K diálogos multi-dominio
  - Task-oriented dialogues

### Datos Tabulares / Estructurados

#### Clasificación
- **Titanic**: Supervivencia en el Titanic
  - [Kaggle](https://www.kaggle.com/c/titanic)
  - Proyecto introductorio clásico

- **Credit Card Fraud**: Transacciones fraudulentas
  - Clases muy desbalanceadas
  - [Kaggle](https://www.kaggle.com/mlg-ulb/creditcardfraud)

- **Adult Income**: Predicción de ingresos
  - Census data

#### Regresión
- **Boston Housing**: Precios de casas
  - 506 muestras, 13 features
  - Clásico para enseñanza

- **California Housing**: Precios de casas en California
  - 20K muestras

- **Ames Housing**: Alternativa moderna a Boston
  - 79 features, más realista

#### Series Temporales
- **Stock Market Data**: Precios de acciones
  - Yahoo Finance, Google Finance
  - Múltiples fuentes

- **Energy Consumption**: Consumo eléctrico
  - Household electric power consumption

- **Weather Data**: Datos meteorológicos
  - NOAA, Weather Underground

### Audio

#### Reconocimiento de Voz
- **LibriSpeech**: 1000 horas de audiolibros
  - [Website](http://www.openslr.org/12/)
  - ASR benchmark

- **Common Voice**: Dataset multilingüe de Mozilla
  - [Website](https://commonvoice.mozilla.org/)
  - 60+ idiomas

- **TIMIT**: Corpus fonético
  - Benchmark clásico

#### Música
- **GTZAN**: 1000 clips de música
  - 10 géneros musicales

- **Million Song Dataset**: Metadatos de 1M canciones
  - [Website](http://millionsongdataset.com/)

### Reinforcement Learning

- **Atari 2600 Games**: Suite de juegos de Atari
  - Incluido en OpenAI Gym
  - Benchmark estándar de RL

- **MuJoCo Environments**: Física para robótica
  - Continuous control tasks

- **StarCraft II**: Juego de estrategia
  - [PySC2](https://github.com/deepmind/pysc2)

### Multimodal

- **MS COCO**: Imágenes con captions
  - Múltiples tareas: detection, captioning, VQA

- **VQA** (Visual Question Answering)
  - Preguntas sobre imágenes
  - [Website](https://visualqa.org/)

- **Flickr30K**: 31K imágenes con 5 captions cada una

- **Conceptual Captions**: 3.3M pares imagen-texto

## 🌍 Datasets en Español

### NLP en Español
- **TASS**: Análisis de sentimientos en español
  - [Website](http://www.sepln.org/workshops/tass/)
  - Tweets en español

- **CoNLL-2002**: NER en español y holandés

- **MLSUM**: Summarization multilingüe
  - Incluye español

- **PAN-CLEF**: Varios tasks en español

- **SBW** (Spanish Billion Words Corpus)
  - Corpus grande de español

### Datasets Latinoamericanos
- **HAHA**: Humor en español
  - Tweets humorísticos

- **EmoEvent**: Detección de emociones
  - Noticias en español

## 💡 Tips para Trabajar con Datasets

### Búsqueda
1. **Define tu tarea primero**: ¿Qué quieres predecir?
2. **Considera el tamaño**: ¿Tienes recursos computacionales?
3. **Revisa licencias**: ¿Puedes usar el dataset?
4. **Checa calidad**: ¿Están bien anotados?

### Descarga
1. **Usa APIs cuando disponibles**: Más fácil que descargas manuales
2. **Considera versiones**: Algunos datasets tienen múltiples versiones
3. **Lee documentación**: Entiende el formato y estructura
4. **Verifica integridad**: Checksums, tamaño de archivos

### Exploración
1. **EDA exhaustivo**: Estadísticas, distribuciones, visualizaciones
2. **Checa valores faltantes**: ¿Cómo manejarlos?
3. **Identifica sesgos**: ¿Es representativo?
4. **Valida calidad**: ¿Errores en anotaciones?

### Uso Ético
1. **Lee términos de uso**: Respeta licencias
2. **Considera privacidad**: Datos sensibles
3. **Identifica sesgos**: No perpetúes discriminación
4. **Cita apropiadamente**: Da crédito a creadores

## 📦 Herramientas para Datasets

### Descarga y Gestión
- **kaggle**: CLI para descargar de Kaggle
```bash
kaggle datasets download -d dataset-name
```

- **HuggingFace Datasets**: Fácil acceso a datasets
```python
from datasets import load_dataset
dataset = load_dataset("squad")
```

- **TensorFlow Datasets**: Datasets listos para usar
```python
import tensorflow_datasets as tfds
ds = tfds.load('mnist', split='train')
```

### Versionado
- **DVC** (Data Version Control): Git para datos
- **Git LFS**: Large File Storage
- **Pachyderm**: Data versioning at scale

### Anotación
- **Label Studio**: Multi-purpose annotation
- **CVAT**: Video/image annotation
- **Prodigy**: Active learning annotation
- **Labelbox**: Enterprise annotation platform

## 🔍 Cómo Crear tu Propio Dataset

### Pasos
1. **Define el objetivo**: ¿Qué quieres predecir?
2. **Identifica fuentes**: Web scraping, APIs, sensors
3. **Recolecta datos**: Automatiza cuando sea posible
4. **Limpia y procesa**: Calidad es crucial
5. **Anota si es necesario**: Crowdsourcing, expertos
6. **Valida**: Checa consistencia
7. **Documenta**: Datasheet, README completo
8. **Comparte**: GitHub, Kaggle, Zenodo

### Consideraciones Éticas
- Obtén consentimiento si aplica
- Anonimiza información personal
- Considera sesgos en recolección
- Documenta limitaciones

## 📚 Recursos Adicionales

### Papers sobre Datasets
- "Datasheets for Datasets" (Gebru et al., 2018)
- "Data Statements for NLP" (Bender & Friedman, 2018)
- "The Dataset Nutrition Label" (Holland et al., 2018)

### Guías
- [Guide to Open Data Publishing](https://data.europa.eu/)
- [Data Packaging Guide](https://frictionlessdata.io/)

## 🎯 Datasets Recomendados por Nivel

### Principiantes
- MNIST / Fashion-MNIST
- Iris
- Titanic
- IMDB Reviews

### Intermedios
- CIFAR-10
- SQuAD
- MS COCO (subconjunto)
- Credit Card Fraud

### Avanzados
- ImageNet
- Common Crawl
- Full MS COCO
- LibriSpeech

## ⚠️ Advertencias

1. **Sesgos**: Muchos datasets tienen sesgos inherentes
   - *Ejemplo*: ImageNet tiene subrepresentación de culturas no occidentales
   - *Acción*: Audita tu dataset, verifica distribuciones, considera datos de múltiples fuentes

2. **Privacidad**: Algunos contienen información sensible
   - *Ejemplo*: Datasets de rostros pueden violar privacidad si se usan sin consentimiento
   - *Acción*: Verifica términos de uso, anonimiza datos personales, cumple GDPR/CCPA

3. **Licencias**: Respeta términos de uso
   - *Ejemplo*: Algunos datasets solo permiten uso académico, no comercial
   - *Acción*: Lee LICENSE.txt, verifica restricciones, documenta fuentes

4. **Actualización**: Datasets pueden quedar obsoletos
   - *Ejemplo*: Datos de redes sociales de 2015 pueden no reflejar comportamiento actual
   - *Acción*: Verifica fecha de recolección, considera drift temporal

5. **Calidad**: Siempre valida calidad de anotaciones
   - *Ejemplo*: Crowdsourced labels pueden tener errores (5-10% típico)
   - *Acción*: Spot-check manual, métricas de inter-annotator agreement, valida outliers

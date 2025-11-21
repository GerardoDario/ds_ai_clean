# Natural Language Processing (NLP)

Este directorio contiene recursos y trabajos relacionados con el Procesamiento de Lenguaje Natural.

## 📚 Temas Principales

### 1. Fundamentos de NLP
- **Preprocesamiento de Texto**
  - Tokenización
  - Stemming y Lemmatización
  - Stop words removal
  - Normalización de texto
  - Regular expressions

- **Representación de Texto**
  - Bag of Words (BoW)
  - TF-IDF
  - N-grams
  - Word Embeddings (Word2Vec, GloVe, FastText)
  - Contextualized embeddings (ELMo, BERT)

### 2. Tareas Clásicas de NLP
- **Clasificación de Texto**
  - Análisis de sentimientos
  - Detección de spam
  - Categorización de documentos
  - Intent classification

- **Named Entity Recognition (NER)**
  - Identificación de entidades
  - Modelos CRF
  - Deep learning para NER

- **Part-of-Speech (POS) Tagging**
  - Etiquetado gramatical
  - Hidden Markov Models
  - Neural POS tagging

- **Parsing Sintáctico**
  - Dependency parsing
  - Constituency parsing

### 3. Modelos de Lenguaje
- **Modelos N-gram**
  - Probabilidades de secuencias
  - Smoothing techniques

- **Modelos Neuronales**
  - RNN Language Models
  - LSTM Language Models
  - Transformer Language Models

- **Modelos Pre-entrenados**
  - **BERT** (Bidirectional Encoder Representations from Transformers)
  - **GPT** (Generative Pre-trained Transformer)
  - **RoBERTa, ALBERT, DistilBERT**
  - **T5** (Text-to-Text Transfer Transformer)
  - **XLNet, ELECTRA**

### 4. Tareas Avanzadas
- **Traducción Automática**
  - Seq2Seq models
  - Attention mechanism
  - Transformer para traducción

- **Question Answering**
  - Extractive QA
  - Generative QA
  - Reading comprehension

- **Text Summarization**
  - Extractive summarization
  - Abstractive summarization
  - Modelos neuronales para resumen

- **Dialogue Systems y Chatbots**
  - Rule-based systems
  - Retrieval-based models
  - Generative models

- **Text Generation**
  - Generación creativa
  - GPT y variantes
  - Control de generación

### 5. NLP Multilingüe
- Cross-lingual embeddings
- Multilingual BERT (mBERT)
- XLM, XLM-R
- Traducción zero-shot

## 🔧 Herramientas y Bibliotecas

### Bibliotecas de Python
- **NLTK**: Natural Language Toolkit (biblioteca clásica)
- **spaCy**: Procesamiento rápido e industrial
- **Transformers (HuggingFace)**: Modelos pre-entrenados de última generación
- **Gensim**: Topic modeling y word embeddings
- **TextBlob**: API simple para NLP
- **Stanford CoreNLP**: Suite completa de herramientas NLP

### Frameworks de Deep Learning
- **PyTorch**: Popular en investigación NLP
- **TensorFlow**: Framework completo
- **Fairseq**: Seq2Seq de Facebook
- **AllenNLP**: Framework de investigación

### Plataformas y Servicios
- **HuggingFace Hub**: Modelos y datasets pre-entrenados
- **Google Cloud Natural Language API**
- **AWS Comprehend**
- **Azure Text Analytics**

## 📖 Recursos Recomendados

### Cursos
- [CS224n - Stanford - Natural Language Processing with Deep Learning](http://web.stanford.edu/class/cs224n/)
- [Fast.ai - NLP Course](https://www.fast.ai/)
- [Coursera - Natural Language Processing Specialization](https://www.coursera.org/specializations/natural-language-processing)

### Libros
- "Speech and Language Processing" - Dan Jurafsky & James H. Martin
- "Natural Language Processing with Python" - Steven Bird, Ewan Klein, Edward Loper
- "Natural Language Processing in Action" - Hobson Lane et al.
- "Transformers for Natural Language Processing" - Denis Rothman

### Papers Fundamentales
- "Attention Is All You Need" - Vaswani et al., 2017 (Transformer)
- "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding" - Devlin et al., 2018
- "Language Models are Few-Shot Learners" - Brown et al., 2020 (GPT-3)
- "Neural Machine Translation by Jointly Learning to Align and Translate" - Bahdanau et al., 2014 (Attention mechanism)

### Blogs y Recursos
- [Jay Alammar's Blog](https://jalammar.github.io/) - Visualizaciones excelentes
- [Sebastian Ruder's Blog](https://ruder.io/)
- [The Gradient](https://thegradient.pub/)

## 🚀 Proyectos Sugeridos

1. **Análisis de Sentimientos en Twitter**: Clasificación de tweets
2. **Chatbot con BERT**: Sistema de Q&A
3. **Generador de Texto con GPT-2**: Fine-tuning para tu dominio
4. **Named Entity Recognition**: Extraer entidades de noticias
5. **Resumen Automático de Artículos**: Summarization con T5
6. **Clasificador de Intenciones**: Para asistentes virtuales
7. **Traducción Automática**: Modelo Seq2Seq
8. **Topic Modeling**: LDA o BERTopic para análisis de documentos

## 📊 Datasets Populares

### Clasificación y Sentimientos
- IMDB Movie Reviews
- Stanford Sentiment Treebank (SST)
- Yelp Reviews
- AG News

### Question Answering
- SQuAD (Stanford Question Answering Dataset)
- Natural Questions
- MS MARCO

### Traducción
- WMT (Workshop on Machine Translation)
- Multi30k
- OpenSubtitles

### Corpus Generales
- Wikipedia dumps
- Common Crawl
- BookCorpus
- C4 (Colossal Clean Crawled Corpus)

### Español
- TASS (análisis de sentimientos)
- CoNLL-2002 (NER)
- MLSUM (summarization)

## 💡 Best Practices

1. **Limpieza de datos**: El preprocesamiento es crucial
2. **Usa modelos pre-entrenados**: Transfer learning es estándar en NLP
3. **Fine-tuning cuidadoso**: Evita overfitting con datasets pequeños
4. **Atención al tokenizer**: Usa el mismo tokenizer del modelo pre-entrenado
5. **Evaluación apropiada**: Usa métricas específicas (BLEU, ROUGE, perplexity)
6. **Considera el contexto**: Los modelos bidireccionales (BERT) suelen superar a unidireccionales
7. **Data augmentation**: Técnicas como back-translation, synonym replacement
8. **Validación en tu dominio**: Los modelos generales pueden no funcionar bien en dominios específicos

## 🌐 Recursos Multilingües

### Modelos
- mBERT (Multilingual BERT)
- XLM-R (Cross-lingual RoBERTa)
- mT5 (Multilingual T5)

### Datasets para Español
- [OPUS - Parallel Corpus](https://opus.nlpl.eu/)
- [Europarl](https://www.statmt.org/europarl/)
- [SBWCE (Spanish Billion Words Corpus)](https://crscardellino.github.io/SBWCE/)

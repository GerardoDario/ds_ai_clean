# Deep Learning - Redes Neuronales Profundas

Este directorio contiene recursos y trabajos relacionados con Deep Learning.

## 📚 Temas Principales

### 1. Fundamentos de Redes Neuronales
- **Perceptrón y Multi-Layer Perceptron (MLP)**
  - Funciones de activación (ReLU, Sigmoid, Tanh, Softmax)
  - Forward propagation
  - Backpropagation
  - Gradient descent y variantes (SGD, Adam, RMSprop)

- **Regularización**
  - Dropout
  - Batch Normalization
  - Layer Normalization
  - Weight Decay

- **Optimización**
  - Learning rate scheduling
  - Gradient clipping
  - Early stopping
  - Inicialización de pesos (Xavier, He)

### 2. Redes Convolucionales (CNN)
- **Arquitecturas Clásicas**
  - LeNet
  - AlexNet
  - VGG
  - ResNet
  - Inception
  - EfficientNet

- **Componentes**
  - Convolutional layers
  - Pooling layers
  - Fully connected layers
  - Skip connections

- **Aplicaciones**
  - Clasificación de imágenes
  - Detección de objetos
  - Segmentación semántica
  - Transferencia de estilo

### 3. Redes Recurrentes (RNN)
- **Arquitecturas**
  - Vanilla RNN
  - LSTM (Long Short-Term Memory)
  - GRU (Gated Recurrent Unit)
  - Bidirectional RNN

- **Aplicaciones**
  - Predicción de series temporales
  - Generación de texto
  - Traducción automática
  - Análisis de sentimientos

### 4. Transformers
- **Arquitectura Transformer**
  - Self-attention mechanism
  - Multi-head attention
  - Positional encoding
  - Feed-forward networks

- **Modelos Populares**
  - BERT
  - GPT (GPT-2, GPT-3, GPT-4)
  - T5
  - Vision Transformers (ViT)

### 5. Autoencoders y GANs
- **Autoencoders**
  - Vanilla Autoencoders
  - Variational Autoencoders (VAE)
  - Denoising Autoencoders
  - Sparse Autoencoders

- **GANs (Generative Adversarial Networks)**
  - Vanilla GAN
  - DCGAN
  - StyleGAN
  - CycleGAN
  - Conditional GAN

## 🔧 Frameworks y Herramientas

### Principales Frameworks
- **TensorFlow/Keras**: Framework completo de Google
- **PyTorch**: Framework flexible y popular en investigación
- **JAX**: Computación numérica de alto rendimiento

### Herramientas de Soporte
- **TensorBoard**: Visualización de entrenamiento
- **Weights & Biases (W&B)**: Tracking de experimentos
- **MLflow**: Gestión del ciclo de vida de ML
- **ONNX**: Interoperabilidad entre frameworks

## 📖 Recursos Recomendados

### Cursos
- [Deep Learning Specialization - Andrew Ng](https://www.coursera.org/specializations/deep-learning)
- [Fast.ai - Practical Deep Learning for Coders](https://course.fast.ai/)
- [CS231n - Stanford - Convolutional Neural Networks](http://cs231n.stanford.edu/)
- [CS224n - Stanford - Natural Language Processing with Deep Learning](http://web.stanford.edu/class/cs224n/)

### Libros
- "Deep Learning" - Ian Goodfellow, Yoshua Bengio, Aaron Courville
- "Neural Networks and Deep Learning" - Michael Nielsen
- "Dive into Deep Learning" - Aston Zhang et al.

### Papers Fundamentales
- "ImageNet Classification with Deep Convolutional Neural Networks" (AlexNet)
- "Very Deep Convolutional Networks for Large-Scale Image Recognition" (VGG)
- "Deep Residual Learning for Image Recognition" (ResNet)
- "Attention Is All You Need" (Transformer)
- "Generative Adversarial Networks" (GAN)

## 🚀 Proyectos Sugeridos

1. **Clasificación de MNIST/CIFAR-10**: Introducción a CNNs
2. **Transfer Learning con ImageNet**: Usar modelos pre-entrenados
3. **Generación de Texto con RNN/LSTM**: Crear un generador de texto
4. **Chatbot con Seq2Seq**: Modelo encoder-decoder
5. **Style Transfer**: Transferencia de estilo artístico
6. **Face Generation con GAN**: Generar rostros sintéticos
7. **Object Detection con YOLO**: Detección de objetos en tiempo real
8. **Fine-tuning de BERT**: Clasificación de texto

## 💻 Recursos Computacionales

### Cloud Platforms
- **Google Colab**: GPUs gratuitas para prototipado
- **Kaggle Kernels**: GPUs gratuitas con límites
- **AWS SageMaker**: Infraestructura profesional
- **Google Cloud AI Platform**: Servicios de ML escalables
- **Azure Machine Learning**: Plataforma empresarial

### Hardware Recomendado
- GPU NVIDIA (GTX 1080 Ti, RTX 3090, A100 para entrenamiento serio)
- RAM: Mínimo 16GB, recomendado 32GB+
- Almacenamiento SSD para datasets grandes

## 💡 Best Practices

1. **Comienza simple**: Prueba primero con modelos pequeños
2. **Data augmentation**: Aumenta tu dataset para mejor generalización
3. **Transfer learning**: No reinventes la rueda, usa modelos pre-entrenados
4. **Monitorea overfitting**: Usa validation set y early stopping
5. **Experimenta con hiperparámetros**: Learning rate, batch size, arquitectura
6. **Visualiza tu red**: Entiende qué aprende cada capa
7. **Usa checkpoints**: Guarda modelos periódicamente durante el entrenamiento
8. **Mixed precision training**: Acelera el entrenamiento con FP16

## 📊 Datasets Populares

### Imágenes
- ImageNet
- COCO (Common Objects in Context)
- CIFAR-10/CIFAR-100
- MNIST/Fashion-MNIST

### Texto
- Wikipedia dump
- Common Crawl
- BookCorpus
- OpenWebText

### Multimodal
- MS COCO (imágenes con captions)
- Visual Question Answering (VQA)
- Conceptual Captions

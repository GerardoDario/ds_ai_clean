# Computer Vision - Visión por Computadora

Este directorio contiene recursos y trabajos relacionados con Computer Vision.

## 📚 Temas Principales

### 1. Fundamentos de Computer Vision
- **Procesamiento de Imágenes**
  - Filtros y convoluciones
  - Detección de bordes (Sobel, Canny)
  - Transformaciones (rotación, escalado, traslación)
  - Histogramas y ecualización
  - Morfología matemática

- **Características y Descriptores**
  - SIFT (Scale-Invariant Feature Transform)
  - SURF (Speeded Up Robust Features)
  - HOG (Histogram of Oriented Gradients)
  - ORB (Oriented FAST and Rotated BRIEF)

### 2. Clasificación de Imágenes
- **Arquitecturas de CNN**
  - LeNet-5
  - AlexNet
  - VGGNet (VGG16, VGG19)
  - ResNet (ResNet50, ResNet101)
  - Inception (GoogLeNet)
  - MobileNet
  - EfficientNet

- **Transfer Learning**
  - Fine-tuning de modelos pre-entrenados
  - Feature extraction
  - Domain adaptation

### 3. Detección de Objetos
- **Arquitecturas Clásicas**
  - R-CNN (Region-based CNN)
  - Fast R-CNN
  - Faster R-CNN
  - Mask R-CNN

- **Arquitecturas Modernas**
  - **YOLO** (You Only Look Once) - v1 a v8
  - **SSD** (Single Shot MultiBox Detector)
  - **RetinaNet**
  - **DETR** (Detection Transformer)

- **Conceptos**
  - Region proposals
  - Anchor boxes
  - Non-maximum suppression (NMS)
  - IoU (Intersection over Union)
  - mAP (mean Average Precision)

### 4. Segmentación de Imágenes
- **Segmentación Semántica**
  - FCN (Fully Convolutional Networks)
  - U-Net
  - SegNet
  - DeepLab (v1, v2, v3, v3+)
  - PSPNet

- **Segmentación de Instancias**
  - Mask R-CNN
  - YOLACT
  - PANet

- **Segmentación Panóptica**
  - Combinación de semántica e instancias
  - Panoptic FPN

### 5. Tareas Avanzadas
- **Pose Estimation**
  - OpenPose
  - AlphaPose
  - MediaPipe

- **Face Recognition y Verification**
  - FaceNet
  - ArcFace
  - Detección de landmarks faciales

- **Optical Flow y Tracking**
  - Lucas-Kanade
  - SORT, DeepSORT
  - Tracking de objetos múltiples

- **Image Generation**
  - GANs para imágenes
  - StyleGAN, StyleGAN2
  - Diffusion Models (Stable Diffusion, DALL-E)

- **3D Vision**
  - Depth estimation
  - 3D reconstruction
  - SLAM (Simultaneous Localization and Mapping)

### 6. Vision Transformers
- **ViT** (Vision Transformer)
- **SWIN** Transformer
- **DeiT** (Data-efficient Image Transformers)
- **CLIP** (Contrastive Language-Image Pre-training)

## 🔧 Herramientas y Bibliotecas

### Bibliotecas Principales
- **OpenCV**: Biblioteca clásica de computer vision
- **PIL/Pillow**: Manipulación básica de imágenes
- **scikit-image**: Procesamiento de imágenes en Python
- **albumentations**: Data augmentation avanzado
- **imgaug**: Augmentation de imágenes

### Frameworks de Deep Learning
- **TensorFlow/Keras**: Framework completo
- **PyTorch**: Popular en investigación
- **Detectron2**: Detección de objetos (Facebook)
- **MMDetection**: Suite completa de detección
- **YOLO oficial**: Implementaciones de YOLO

### Herramientas Especializadas
- **Roboflow**: Anotación y gestión de datasets
- **LabelImg**: Anotación de bounding boxes
- **CVAT**: Anotación de video e imágenes
- **VGG Image Annotator (VIA)**: Anotación web

## 📖 Recursos Recomendados

### Cursos
- [CS231n - Stanford - Convolutional Neural Networks for Visual Recognition](http://cs231n.stanford.edu/)
- [Deep Learning for Computer Vision - Michigan](https://web.eecs.umich.edu/~justincj/teaching/eecs498/FA2020/)
- [First Principles of Computer Vision - Columbia](https://fpcv.cs.columbia.edu/)

### Libros
- "Computer Vision: Algorithms and Applications" - Richard Szeliski
- "Deep Learning for Computer Vision" - Rajalingappaa Shanmugamani
- "Modern Computer Vision with PyTorch" - V Kishore Ayyadevara, Yeshwanth Reddy
- "Multiple View Geometry in Computer Vision" - Richard Hartley, Andrew Zisserman

### Papers Fundamentales
- "ImageNet Classification with Deep CNNs" (AlexNet, 2012)
- "Very Deep CNNs for Large-Scale Image Recognition" (VGG, 2014)
- "Deep Residual Learning for Image Recognition" (ResNet, 2015)
- "You Only Look Once: Unified, Real-Time Object Detection" (YOLO, 2016)
- "Mask R-CNN" (2017)
- "An Image is Worth 16x16 Words: Transformers for Image Recognition" (ViT, 2020)

### Comunidades y Recursos
- [Papers With Code - Computer Vision](https://paperswithcode.com/area/computer-vision)
- [Awesome Computer Vision](https://github.com/jbhuang0604/awesome-computer-vision)
- [PyImageSearch](https://www.pyimagesearch.com/)

## 🚀 Proyectos Sugeridos

1. **Clasificador de Imágenes**: CIFAR-10 o ImageNet subset
2. **Detector de Objetos en Tiempo Real**: Implementar YOLO
3. **Segmentación de Imágenes Médicas**: U-Net para segmentación
4. **Reconocimiento Facial**: Sistema de verificación de identidad
5. **Contador de Personas**: Detección y tracking
6. **OCR (Optical Character Recognition)**: Lectura de texto en imágenes
7. **Clasificación de Defectos**: Control de calidad industrial
8. **Análisis de Tráfico**: Detección y conteo de vehículos
9. **Pose Estimation para Fitness**: Análisis de ejercicios
10. **Generación de Imágenes**: GAN o Diffusion Model

## 📊 Datasets Populares

### Clasificación
- ImageNet
- CIFAR-10/CIFAR-100
- MNIST/Fashion-MNIST
- Tiny ImageNet

### Detección de Objetos
- MS COCO (Common Objects in Context)
- Pascal VOC
- Open Images Dataset
- LVIS (Large Vocabulary Instance Segmentation)

### Segmentación
- Cityscapes (conducción autónoma)
- ADE20K (escenas)
- Mapillary Vistas

### Específicos
- CelebA (rostros)
- WIDER FACE (detección de rostros)
- KITTI (conducción autónoma)
- LFW (Labeled Faces in the Wild)

## 💡 Best Practices

1. **Data Augmentation**: Esencial para evitar overfitting
   - Flips, rotations, crops
   - Color jittering
   - Mixup, CutMix
2. **Transfer Learning**: Empieza con modelos pre-entrenados en ImageNet
3. **Input normalization**: Usa las mismas estadísticas del pre-entrenamiento
4. **Resolución adecuada**: Balance entre precisión y velocidad
5. **Test-Time Augmentation (TTA)**: Mejora resultados en inferencia
6. **Ensemble methods**: Combina múltiples modelos
7. **Visualiza predicciones**: Entiende errores del modelo
8. **Considera restricciones**: Latencia, memoria, hardware disponible

## 🎯 Aplicaciones del Mundo Real

### Industria
- Control de calidad automatizado
- Clasificación de productos
- Robótica industrial

### Medicina
- Detección de tumores
- Segmentación de órganos
- Diagnóstico asistido

### Automoción
- Conducción autónoma
- ADAS (Advanced Driver Assistance Systems)
- Monitorización del conductor

### Retail
- Checkout sin cajero
- Análisis de comportamiento de clientes
- Gestión de inventario

### Seguridad
- Reconocimiento facial
- Detección de intrusos
- Análisis de video vigilancia

## 🔬 Tendencias Actuales

- **Vision Transformers**: Superando CNNs en muchas tareas
- **Self-supervised Learning**: Aprendizaje sin etiquetas
- **Few-shot Learning**: Aprender con pocos ejemplos
- **Neural Architecture Search**: Automatización del diseño de redes
- **Edge AI**: Modelos eficientes para dispositivos móviles
- **Multimodal Learning**: Combinando visión con lenguaje (CLIP, DALL-E)

# Infant-Cry-Detection-using-CNN

# 👶 Infant Cry Classification Using CNN

## 📌 Overview

This project focuses on classifying different types of infant cries using Convolutional Neural Networks (CNNs). By analyzing audio recordings from the [Infant Cry Audio Corpus](https://www.kaggle.com/datasets/warcoder/infant-cry), the model aims to distinguish between various cry types, facilitating better understanding and response to infants' needs.

---

## ✨ Features

- 🎧 **Audio Data Preprocessing**: Converts raw audio files into Mel-frequency cepstral coefficients (MFCCs) for effective feature extraction.  
- 🧠 **Deep Learning Model**: Implements a CNN architecture optimized for audio classification tasks.  
- 📊 **Evaluation Metrics**: Assesses model performance using accuracy, precision, recall, and F1-score.  

---

## ⚙️ Implementation

### 📁 Dataset
- Uses the [Infant Cry Audio Corpus](https://www.kaggle.com/datasets/warcoder/infant-cry), consisting of labeled infant cry recordings.

### 🔍 Preprocessing
- **Feature Extraction**: Computes MFCCs from audio signals.
- **Data Augmentation**: Applies techniques such as:
  - Noise addition
  - Time shifting

### 🧱 Model Architecture
- **Convolutional Layers**: Capture patterns in MFCCs.
- **Pooling Layers**: Reduce dimensionality.
- **Fully Connected Layers**: Perform final classification.

### 🏋️ Training
- **Loss Function**: Categorical Cross-Entropy
- **Optimizer**: Adam
- **Validation**: Uses a validation set to monitor overfitting

### 📈 Evaluation
- **Confusion Matrix**: Shows true vs. predicted labels
- **Classification Report**: Includes precision, recall, and F1-score

---




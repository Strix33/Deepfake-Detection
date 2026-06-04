# Audio Deepfake Detection

This repository contains the machine learning-based system to detect deepfake (synthetic) voices. The system utilizes audio feature extraction techniques such as YAMNet and deep learning models (ANN, CNN, RNN) to differentiate between real and fake audio.

## 🚀 Features
- **Audio Preprocessing**: Converts raw audio into meaningful features.
- **Feature Extraction**: Uses YAMNet to extract embeddings from audio signals.
- **Dataset**: Uses the Kaggle "In The Wild (audio Deepfake)" dataset for training the model.
- **Deep Learning Models**: Implements ANN, CNN, and RNN architectures for classification.
- **Training & Evaluation**: Trains models with labeled datasets and evaluates accuracy.
- **Inference API**: Provides an API to classify input audio as real or deepfake.

## Model Performance
- **ANN Model**: Achieves 97% accuracy (Best Model)
- **CNN Model**: Achieves 97% accuracy
- **RNN Model**: Achieves 94% accuracy

## 🛠️ Setup & Installation

1. Navigate to the directory:
   ```bash
   cd Audio_Deepfake_Detection
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run inference / API:
   ```bash
   python main.py
   ```

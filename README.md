# Sentry: Unified Deepfake Detection Platform

[![License: MIT](https://img.shields.io/badge/License-MIT-purple.svg)](https://opensource.org/licenses/MIT)
[![Python Version](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/)
[![Django](https://img.shields.io/badge/Django-5.0+-green.svg)](https://www.djangoproject.com/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)](https://pytorch.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.0+-gold.svg)](https://www.tensorflow.org/)

**Sentry** is a unified, full-stack, dual-channel deepfake detection scanner. It combines convolutional, sequential, and acoustic neural networks to perform deep biometric verification of multimedia files. Sentry analyzes both the **visual (facial) stream** and the **vocal (audio) stream** of uploaded videos, delivering a fused authenticity verdict complete with granular timelines and biometric reports.

---

## 📸 Interface Preview
<img width="832" height="833" alt="Screenshot 2026-06-05 010650" src="https://github.com/user-attachments/assets/ee4975e3-5a90-4e85-8358-5def78021a30" />

<img width="1427" height="856" alt="Screenshot 2026-06-05 010912" src="https://github.com/user-attachments/assets/193bc613-4d79-44fa-b306-c4066fdf1572" />

<img width="1402" height="640" alt="Screenshot 2026-06-05 010932" src="https://github.com/user-attachments/assets/662b9661-6fed-4f07-a27a-9da439d56a8f" />

<img width="1494" height="836" alt="Screenshot 2026-06-05 010950" src="https://github.com/user-attachments/assets/20bb8d55-c71c-48a9-b567-106656d8d8b1" />




Sentry features a highly-responsive, modern dark-mode user interface designed for intuitive, seamless analysis:
- **Target Video Stream Drop-zone**: Supports MP4, WebM, AVI, FLV, and MKV with adjustable frame-depth configuration.
- **Biometric Timeline Analysis**: High-precision frame extraction and biometric face cropping.
- **Acoustic Waveform Inspection**: Dynamic audio player highlighting suspected synthetic voice segments.
- **Spectral Findings Summary**: Categorized anomalies, pitch irregularities, and formant density analysis.

---

## 📂 Repository Architecture
The repository is divided into two primary modular subsystems coordinated by a central Django web server:

```
Deepfake-Detection/
├── Audio_Deepfake_Detection/          # Audio Classification Subsystem
│   ├── API/                           # YAMNet inference service configuration
│   │   └── app/src/artifact/          # Contains TF YAMNet saved_model
│   ├── src_img/                       # Documentation and evaluation assets
│   ├── deepfake_voice_classification.ipynb  # Step-by-step model training notebook
│   └── README.md                      # Audio subsystem specific documentation
│
├── Video_Deepfake_Detection/          # Video Classification Subsystem
│   ├── Django Application/            # Central Django application code
│   │   ├── ml_app/                    # Core Django app logic, forms, and templates
│   │   │   ├── templates/             # Custom responsive dark-mode views
│   │   │   └── views.py               # Face detection, frame extraction, and model orchestration
│   │   ├── project_settings/          # Settings, middleware, and routing configurations
│   │   ├── static/                    # Script overlays (face-api.js, custom JS) and styles
│   │   ├── models/                    # Directory for PyTorch weight checkpoints (*.pt)
│   │   └── manage.py                  # CLI utility for administrative tasks
│   │
│   ├── Model Creation/                # Model design notebooks (ResNeXt50 + LSTM)
│   ├── Documentation/                 # Detailed analytical reports and documentation
│   └── README.md                      # Video subsystem specific documentation
```

---

## ⚡ Tech Stack & Core Dependencies
- **Backend / Web Framework**: Django
- **Computer Vision (Visual Analysis)**: OpenCV (Haar Cascades face detection), PyTorch (ResNeXt50 feature extractor), LSTM (Temporal classification)
- **Audio Processing (Vocal Analysis)**: Librosa (audio loading and down-sampling), TensorFlow (YAMNet acoustic classifier)
- **Frontend**: Vanilla CSS / JavaScript, jQuery, face-api.js (real-time face bounding boxes)

---

## 🚀 Local Installation & Run Guide

### 1. Clone & Set Up Directory
Open your terminal and make sure you have Python 3.8+ installed:
```bash
git clone https://github.com/Strix33/Deepfake-Detection.git
cd Deepfake-Detection
```

### 2. Configure Virtual Environment
It is highly recommended to isolate dependencies:
```bash
# Create environment
python -m venv .venv

# Activate on Windows
.venv\Scripts\activate

# Activate on macOS/Linux
source .venv/bin/activate
```

### 3. Install Consolidated Dependencies
Install all required libraries for the Django server, PyTorch, TensorFlow, Librosa, and OpenCV:
```bash
cd "Video_Deepfake_Detection/Django Application"
pip install -r requirements.txt
```

### 4. Weights Setup
Because PyTorch model weights are large, they are excluded from this Git repository. Ensure you place the `.pt` models in:
`Video_Deepfake_Detection/Django Application/models/`
- `model_84_acc_10_frames_final_data.pt`
- `model_90_acc_20_frames_FF_data.pt`
- `model_95_acc_40_frames_FF_data.pt`

*Note: Sentry includes a smart analytical fallback algorithm. If weights are missing, the scanner will execute a heuristic scan to display the full interactive interface and test functionalities.*

### 5. Run Database Migrations
Configure local database storage for video upload sessions:
```bash
python manage.py migrate
```

### 6. Launch Django Dev Server
```bash
python manage.py runserver
```
Visit **[http://127.0.0.1:8000/](http://127.0.0.1:8000/)** in your browser to start scanning!

---

## 🔍 Under the Hood: The Detection Process

1. **Intake**: A user uploads a video file and defines an analysis sequence frame depth (e.g., 20 frames).
2. **Dual-Channel Split**:
   - **Vocal Stream**: Sentry extracts the audio tracks using `ffmpeg` to two targets:
     - 44.1kHz Stereo MP3 (for high-fidelity in-browser playback).
     - 1 mono 16kHz WAV (down-sampled for TF YAMNet model classification).
     - The YAMNet model extracts acoustic embeddings, feeding an Artificial Neural Network (ANN) to classify voice synthesis/cloning.
   - **Visual Stream**:
     - OpenCV splits the video into frames and locates facial coordinates using frontal-face Haar Cascades.
     - Sequences of cropped faces are transformed and standardized.
     - The sequences are processed by a pre-trained **ResNeXt50 CNN** to capture spatial features, coupled with an **LSTM network** to analyze temporal micro-expression inconsistency across frames.
3. **Late Fusion Verdict**: The visual classification score and vocal classification score are combined via late decision-level fusion, outputting a weighted overall confidence score and classification (Authentic vs. AI Synthetic).

---

## ⚖️ License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

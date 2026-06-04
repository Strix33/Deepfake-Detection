# Video Deepfake Detection using Deep Learning (ResNext and LSTM)

## 1. Introduction
This project aims to detect video deepfakes using deep learning techniques like ResNeXt and LSTM. We achieve deepfake detection by using transfer learning where a pre-trained ResNeXt CNN is used to obtain a feature vector, and an LSTM layer is trained using the features.

## 2. Directory Structure
The video detection component is structured as follows:
```
Video_Deepfake_Detection
    ├── Django Application
    ├── Model Creation
    └── Documentation
```

1. **Django Application**: Contains the full-stack web application where users can upload a video, run analysis, and visualize predictions.
2. **Model Creation**: Step-by-step Jupyter notebooks detailing model training.
3. **Documentation**: Additional reports and documentation.

## 3. Results

| Model Name | No. of Videos | No. of Frames | Accuracy |
|------------|--------------|--------------|----------|
| model_84_acc_10_frames_final_data.pt | 6000 | 10 | 84.2% |
| model_87_acc_20_frames_final_data.pt | 6000 | 20 | 87.8% |
| model_89_acc_40_frames_final_data.pt | 6000 | 40 | 89.3% |
| model_90_acc_60_frames_final_data.pt | 6000 | 60 | 90.6% |
| model_91_acc_80_frames_final_data.pt | 6000 | 80 | 91.5% |
| model_93_acc_100_frames_final_data.pt| 6000 | 100 | 93.6% |

## 🛠️ Setup & Installation
Please navigate to the `Django Application` folder for running the web application.

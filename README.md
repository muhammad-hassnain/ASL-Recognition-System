# ASL Hand Signs Recognition Project
## Overview
This is a comprehensive deep learning project for American Sign Language (ASL) hand sign recognition developed as part of the ECS 271 Fall'24 Course. The project features a sophisticated machine learning pipeline with four advanced neural network architectures and a user-friendly PyQt-based real-time recognition frontend.

## Project Architecture
- **Models Implemented**:
  1. `model_1`: Convolutional Neural Network (Conv2D)
  2. `model_2`: ResNet50 Transfer Learning Model
  3. `model_3`: VGG-16 Transfer Learning Model
  4. `model_4`: MobileNetV2 Transfer Learning Model

## Prerequisites
- Python 3.8+
- pip package manager
- CUDA-compatible GPU (recommended for faster training)

## Installation

### 1. Clone the Repository
```bash
git clone https://github.com/muhammad-hassnain/ASL-Recognition-System.git
cd 271_Project
```

### 2. Create Virtual Environment
```bash
python3 -m venv asl_env
source asl_env/bin/activate  # On Windows use: asl_env\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

## Model Training

### Training Workflow
For each model, follow these steps:

1. Navigate to the specific model directory:
```bash
cd model_{x}  # Replace {x} with 1, 2, 3, or 4
```

2. Train the model:
```bash
python3 train_model_{x}.py
```

### Training Outputs
After training, the following artifacts will be generated:
- `plots_model_x/`: Directory containing performance visualization plots
- `metrics_output.txt`: Comprehensive model performance metrics
- `model_parameters.txt`: Trained model weights and configurations
- `model_{x}.weights`: Model weight files for future inference

## Model Evaluation

### Invoking Pre-trained Models
To assess a previously trained model's performance:

1. Navigate to the model directory:
```bash
cd model_{x}  # Replace {x} with 1, 2, 3, or 4
```

2. Run model evaluation:
```bash
python3 invoke_model_{x}.py
```

### Evaluation Outputs
- Console display of model performance
- `model_metrics.txt`: Detailed performance metrics
- Precision, Recall, F1-Score, and Confusion Matrix

## Real-time Hand Sign Recognition App

### Application Features
- Real-time hand gesture recognition
- Multi-model support
- Webcam/camera integration
- Prediction confidence display

### Running the Real-time App
```bash
cd model_{x}  # Replace {x} with 1, 2, 3, or 4
python3 video_model_{x}.py
```

### App Interaction
- Open the application
- Position your hand in the camera frame
- App detects and classifies ASL hand signs
- Displays prediction label and confidence score

## Troubleshooting
- Ensure all dependencies are correctly installed
- Check camera permissions
- Verify Python and pip versions
- Confirm GPU drivers for deep-learning models

## Contact and Support
For issues, inquiries, or collaboration:
- **Member 1**: Nafiz Imtiaz Khan
- **Email**: nikhan@ucdavis.edu
- **Member 2**: Muhammad Hassnain
- **Email**: mhassnain@ucdavis.edu
- **Member 3**: Md Raian Latif Nabil
- **Email**: mnabil@ucdavis.edu

## License
MIT License

Copyright (c) 2024 The Regents of the University of California, Davis campus. All Rights Reserved. Used with permission.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.


## Acknowledgments
- ECS 271 Course, Fall'24
- Deep Learning Research Group
- Open-source libraries and frameworks used

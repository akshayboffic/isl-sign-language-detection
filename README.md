# Indian Sign Language (ISL) Detection

A real-time Indian Sign Language (ISL) hand gesture recognition model using machine learning and computer vision techniques. This project utilizes MediaPipe for hand landmark detection and a Random Forest Classifier to predict the sign language gestures. The model is deployed with a Streamlit app for real-time webcam prediction.

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
  - [Web App](#web-app)
  - [Real-Time Prediction](#real-time-prediction)
- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [License](#license)

## Introduction

This project is a gesture recognition system for Indian Sign Language (ISL). It takes in live webcam feed, detects hand landmarks using MediaPipe, and uses a pre-trained Random Forest model to predict the sign being made in real time. The system is built using Python and can be run on any machine with a webcam.

## Features

- **Real-Time Detection**: Detects hand gestures in real-time using your webcam.
- **Gesture Prediction**: Uses a Random Forest Classifier to predict ISL signs such as "Hello", "Yes", "No", etc.
- **Web Interface**: Provides a Streamlit-based web interface to interact with the system.
- **Model Deployment**: The model is saved and can be loaded for inference.

## Installation

### Clone the Repository

First, clone the repository to your local machine:

```bash
git clone https://github.com/<your-username>/isl-sign-language-detection.git
cd isl-sign-language-detection
```

### Install Dependencies

Install the necessary Python libraries:

```bash
pip install -r requirements.txt
```

The `requirements.txt` includes the following libraries:
- mediapipe
- opencv-python
- numpy
- scikit-learn
- streamlit
- Pillow

### Download or Train the Model

You can either:
- Download the pre-trained model (`isl_model.p`), or
- Train the model by running `preprocessing.ipynb` and `classifier.py`.

### Preprocess Data

Run `preprocessing.ipynb` to preprocess the hand gesture image data in the `allimages` folder. This will save the processed data in `isl_data.pickle`.

### Train the Classifier

Run `classifier.py` to train the Random Forest Classifier. It will save the trained model in `isl_model.p`.

## Usage

### Web App

1. Start the Streamlit app by running the following command:
   ```bash
   streamlit run isl_streamlit_app.py
   ```
2. Open the app in your browser. You can now click the checkbox to start the webcam feed. The model will predict the sign language gestures in real time and display them on the screen.

### Real-Time Prediction

You can also run the real-time prediction script (`tscls.ipynb`) for webcam-based prediction:

1. Run the script:
   ```bash
   python tscls.ipynb
   ```
2. The webcam will open, and it will display the predicted gesture on the screen. Press 'q' to exit the webcam feed.

## Project Structure

```
.
├── allimages/                  # Folder containing gesture images (subfolders: hello, yes, no, thankyou, etc.)
├── preprocessing.ipynb         # Preprocessing script to convert images into 'isl_data.pickle'
├── classifier.py               # Script to train the RandomForestClassifier and save the model
├── tscls.ipynb                 # Real-time model for webcam-based gesture recognition
├── isl_streamlit_app.py        # Streamlit app for real-time webcam prediction
├── requirements.txt            # Python dependencies
├── isl_data.pickle             # Processed data (generated from preprocessing.ipynb)
└── isl_model.p                 # Trained model (generated from classifier.py)
```

## Requirements

- Python 3.x
- A webcam (for real-time gesture detection)
- Required Python packages:
  - mediapipe
  - opencv-python
  - numpy
  - scikit-learn
  - streamlit
  - Pillow

## License

This project is licensed under the MIT License - see the LICENSE file for details.


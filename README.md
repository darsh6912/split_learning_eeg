EEG Signal Classification using Imagined Speech Recognition

Project Overview:

This project focuses on EEG Signal Classification for imagined speech recognition using data acquired from RMS EEG devices. The system processes raw EEG signals, extracts features, and classifies imagined words using machine learning models. The project has applications in Brain-Computer Interfaces (BCIs), silent communication, cognitive biometrics, and assistive healthcare.

Features:

EEG Data Acquisition from RMS EEG devices.

Preprocessing: Artifact removal and filtering.

Feature Extraction from EEG frequency bands .

Machine Learning Models for classification.

Evaluation Metrics: Accuracy, Confusion Matrix, and Classification Report.

Applications: BCIs, Assistive Technologies, Healthcare, and Cognitive Studies.

Dataset

Subjects: 11

Imagined Words: 8 (e.g., Silent, Water, Pain, Toilet, Doctor, Move, Yes, No).

Electrodes Used: 8 specific positions associated with language and cognitive processing.

Data Format: Time-series EEG recordings.

System Workflow

Data Acquisition → Raw EEG signals captured using RMS devices.

Preprocessing → Noise/artifact removal, normalization, and band-pass filtering.

Feature Extraction → gini importance

Classification → ML models (Extra trees classifier).

Evaluation → Model accuracy, confusion matrix, classification report.

Technologies Used

Programming Language: Python

Libraries: NumPy, Pandas, SciPy, Zlib, pickle , StandardScaler.

Tools: VS Code, EEG RMS acquisition software

Results

Achieved promising accuracy in classifying imagined speech words.

Demonstrated feasibility of non-invasive EEG-based BCIs for communication.

Visualized EEG signal patterns and classification performance using plots.

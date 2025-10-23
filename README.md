# EEG Signal Classification for Imagined Speech

## 📌 Project Overview
This project focuses on **EEG Signal Classification** for imagined speech recognition using data acquired from RMS EEG devices. The system processes raw EEG signals, extracts features, and classifies imagined words using machine learning models.  

Applications include:
- Brain-Computer Interfaces (BCIs)
- Silent communication
- Cognitive biometrics
- Assistive healthcare

---

## 🚀 Features

- **EEG Data Acquisition:** Collect signals from RMS EEG devices.
- **Preprocessing:** Artifact removal, filtering, and normalization.
- **Feature Extraction:** Extract features from EEG frequency bands (Delta, Theta, Alpha, Beta, Gamma) using Gini Index.
- **Machine Learning Models:** Classification using models such as Extra Trees Classifier.
- **Evaluation Metrics:** Accuracy, Confusion Matrix, Classification Report.
- **Applications:** BCIs, Assistive Technologies, Healthcare, Cognitive Studies.

---

## 🧠 Dataset

- **Subjects:** 11
- **Imagined Words:** 8 (e.g., Silent, Water, Pain, Toilet, Doctor, Move, Yes, No)
- **Electrodes Used:** 8 positions linked to language and cognitive processing
- **Data Format:** Time-series EEG recordings

---

## ⚙️ System Workflow

1. **Data Acquisition:** Raw EEG signals captured using RMS devices.
2. **Preprocessing:** Noise/artifact removal, normalization, and band-pass filtering.
3. **Feature Extraction:** Compute features using Gini Index across EEG bands.
4. **Classification:** Classify imagined words using machine learning models (Extra Trees Classifier).
5. **Evaluation:** Assess model performance using accuracy, confusion matrix, and classification report.

---

## 🛠️ Technologies Used

- **Programming Language:** Python
- **Libraries:** NumPy, Pandas, Zlib, StandardScaler, pickle, Socket
- **Tools:** VS Code

---

## 📊 Results

- Achieved promising accuracy in classifying imagined speech words.
- Demonstrated feasibility of **non-invasive EEG-based BCIs** for communication.
- Visualized EEG signal patterns and classification performance using plots.

---

## 📂 Project Structure
eeg-imagined-speech/
│
├── data/ # Raw and processed EEG datasets
├── preprocessing/ # Scripts for artifact removal and filtering
├── feature_extraction/ # Scripts for extracting EEG features
├── models/ # Trained ML models (pickle files)
├── evaluation/ # Scripts for evaluating model performance
├── results/ # Plots and performance metrics
├── README.md # Project documentation
└── requirements.txt # Python dependencies

yaml
Copy code

---

## 🛠 Installation

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/eeg-imagined-speech.git
cd eeg-imagined-speech

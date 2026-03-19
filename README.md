Project Overview

This project provides a multimodal system for assisting the diagnosis of Obstructive Sleep Apnea (OSA) based on audio and facial image features.
Key functionalities include:

Audio preprocessing and feature extraction (DeepSpectrum, MFCC, etc.)

Facial image preprocessing and feature extraction (MediaPipe, FaceNet, etc.)

Multi-view feature fusion using GCCA

OSA classification using XGBoost

Multimodal fusion classification using mutual information (MI)

Project Structure
project_root/
│
├─ preprocessing/
│  ├─ audio_DeepSpectrum/       # Audio preprocessing and feature extraction
│  └─ face_dataset_deal/        # Facial image preprocessing and feature extraction
│
├─ GCCAfusion.py                # Multi-view GCCA fusion script
├─ MI.py                        # Mutual information based multimodal fusion
├─ XGBoost_c.py                 # XGBoost classification and repeated experiments
└─ README.md
Requirements

Python >= 3.8

numpy

pandas

scikit-learn

xgboost

matplotlib

seaborn

openpyxl

torch (required for MI module)

tensorflow / keras (optional, for audio/facial feature extraction)

Install dependencies via:

pip install numpy pandas scikit-learn xgboost matplotlib seaborn openpyxl torch tensorflow keras
Usage
1. Audio & Facial Feature Extraction

preprocessing/audio_DeepSpectrum/: Extract features from audio files.

preprocessing/face_dataset_deal/: Extract features from facial images.

2. GCCA Fusion
python GCCAfusion.py

Fuse multi-view features and save shared representations to Excel.

3. XGBoost Classification
python XGBoost_c.py

Load GCCA-fused features for classification.

Supports multiple repeated experiments and outputs mean ± std metrics.

4. MI Fusion Classification
python MI.py

Run multimodal mutual information fusion and classification.

Outputs performance metrics and visualizations.

Output

GCCA fused features: Excel files

Classification results: Excel files with mean ± std metrics

Optional: confusion matrices and performance plots

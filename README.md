# Malware Detection for Health Sensor Data
# Overview 
This project focuses on using machine learning to detect potential malware threats embedded in health sensor code an important step for securing IoT and medical devices. The system uses labeled datasets and ensemble learning models to flag suspicious patterns in code behavior.
# Problem
Malware embedded in health sensor firmware or code can compromise patient data and device functionality. I wanted to build a model that detects such patterns based on behavioral features from sensor code samples.

# How I Did It
1. Data Understanding & Preprocessing:
- Collected labeled datasets with code snippets and metadata showing malware vs benign behavior.

- Performed data cleaning: removed nulls, normalized values, and encoded categorical features.

2. Feature Engineering:
- Extracted static features (e.g., opcode frequency, function calls, code size) and behavioral signals (e.g., communication patterns).

- Applied feature selection techniques to remove noisy features and reduce dimensionality.

3. Model Training:
- Trained multiple models: Random Forest, XGBoost, and LightGBM.

- Used cross-validation and grid search for hyperparameter tuning.

- Balanced the dataset using SMOTE to improve class representation.

4. Model Evaluation
- Measured performance using accuracy, F1-score, and confusion matrix.

- XGBoost achieved the best trade-off between recall and precision, detecting malware with high reliability.

# Results
- Achieved ~95% accuracy on test data.

- Final model generalized well on unseen samples, showing potential for real-world deployment in embedded device security.

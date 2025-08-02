# Civil Aircraft Hazard Prediction Using LSTM
# Overview
This project uses time-series deep learning (LSTM) to predict hazardous events in civil aircraft, based on APU (Auxiliary Power Unit) sensor data. The aim is to detect patterns that precede faults or anomalies in engine operation, helping improve aviation safety.

# Problem
Aircraft contain sensors (e.g., APU sensors) that record signals during flight. I aimed to detect hazardous events in advance using this time-series data.

# How I Did It
1. Data Preprocessing:
- Loaded APU sensor data (time-series).

- Cleaned and normalized data to ensure consistent time intervals.

- Handled missing values using interpolation and padding.

2. Feature Engineering:
- Calculated statistical features: rolling averages, signal deltas, and custom domain-specific indicators.

- Used RFECV (Recursive Feature Elimination with Cross-Validation) to reduce input noise and improve model focus.

3. Time-Series Modeling:
- Transformed the dataset into sliding time windows to feed into the LSTM model (e.g., 10-step sequences).

- Built and trained an LSTM model using either TensorFlow or PyTorch.

- Included dropout layers to avoid overfitting and added early stopping during training.

4. Model Evaluation:
- Evaluated on a test set using F1-score, accuracy, and confusion matrix.

- Fine-tuned model architecture and learning rate to stabilize training.

# Results
- Final LSTM model achieved around 92% accuracy.

- Detected hazard patterns with high recall, making it useful for predictive maintenance systems in aviation.

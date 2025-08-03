# Civil Aircraft Hazard Prediction Using (LSTM) Deep Learning
# Situation & Task:
Aircraft engines generate a large amount of operational sensor data, which can be leveraged to predict hazardous events in advance. The objective of this project was to forecast engine-related hazards using time-series APU (Auxiliary Power Unit) sensor data. I aimed to create a deep learning model that could detect early warning signals of malfunction.
# Action:
I started by cleaning and aligning the multivariate time-series data using interpolation and normalization. I used RFECV (Recursive Feature Elimination with Cross-Validation) to identify the most relevant features. Then, I reshaped the data into time-windowed sequences and developed an LSTM (Long Short-Term Memory) model to capture temporal patterns. I optimized the model using dropout regularization and early stopping, and evaluated it with accuracy, precision, and F1-score to measure predictive success.
# Result:
The final LSTM model achieved 92% accuracy and an F1-score of 90%, successfully predicting hazardous conditions based on prior sensor behavior. This project demonstrated the strength of deep learning in preventive maintenance applications and real-time anomaly detection in aviation.
# Tools & Technologies Used:
- Programming: Python
- Libraries: TensorFlow or PyTorch, scikit-learn, pandas, NumPy, matplotlib, seaborn
- Tasks: Time-series modeling, LSTM networks, feature selection, evaluation metrics

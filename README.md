# Malware Detection for Health Sensor Data Using Machine Learning
# Situation & Task:
With the rise of IoT-enabled healthcare systems, ensuring the security of health sensor software has become crucial. The goal of this project was to detect malware patterns hidden in code snippets from health sensors. I was tasked with building a high-accuracy machine learning model that could classify whether a given sample of code was malicious or safe, despite data imbalance and limited labeled examples.Action:
# Action:
I collected and preprocessed labeled code data, extracting both static features (e.g., opcode frequency, function call patterns) and behavioral signals. I implemented several supervised ML models including XGBoost, LightGBM, and Random Forest, comparing their performance using stratified cross-validation. To improve feature quality, I performed correlation analysis, feature scaling, and recursive feature elimination. The pipeline was tested using a combination of accuracy, precision, and F1-score for balanced evaluation.
# Result:
The XGBoost model outperformed others, achieving 95% accuracy and an F1-score of 93% on the test set. This robust performance showed the model’s potential for integration into early-warning systems for secure medical devices. The project showcased the value of ensemble learning in security-critical domains.

# Tools & Technologies Used:
- Programming: Python
- Libraries: scikit-learn, XGBoost, LightGBM, pandas, NumPy, matplotlib
- Tasks: Feature engineering, model evaluation, cross-validation, imbalanced data handling

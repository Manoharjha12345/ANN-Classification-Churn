## ANN Customer Churn Prediction

Predicts whether a bank customer will churn using a Deep Learning 
ANN model built with TensorFlow/Keras.

## Problem Statement
Banks lose revenue when customers leave. This model helps identify 
**at-risk customers early** so the bank can take action to retain them.

##  What it does
- Takes customer data (age, balance, credit score, geography, etc.)
- Preprocesses & encodes features automatically
- Predicts churn probability using a trained ANN
- If probability > 0.5 → Customer likely to churn

## Model Architecture
Input (12 features)
→ Dense(64, ReLU)
→ Dense(32, ReLU)  
→ Dense(1, Sigmoid) → Churn Probability

- Optimizer: Adam
- Loss: Binary Crossentropy
- Early Stopping: patience=5 (prevents overfitting)
- Training monitored via TensorBoard

## Dataset
- 10,000 bank customer records
- Features: CreditScore, Geography, Gender, Age, Tenure, 
  Balance, NumOfProducts, HasCrCard, IsActiveMember, EstimatedSalary
- Target: Exited (1 = churned, 0 = stayed)

## How to Run
pip install -r requirements.txt
streamlit run app.py

##  Tech Stack
TensorFlow · Keras · Scikit-learn · Pandas · NumPy · Streamlit · TensorBoard

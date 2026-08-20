# Credit Card Fraud Detection

A machine-learning-based web application that predicts whether a credit card transaction is potentially fraudulent and provides a fraud probability.

## Project Overview

This project uses XGBoost to classify credit card transactions as legitimate or fraudulent.

The system includes:

- Data inspection and preprocessing
- Class imbalance handling
- XGBoost classification
- Model evaluation
- Model serialization using Joblib
- FastAPI REST API
- HTML/CSS/JavaScript frontend
- Real-time fraud prediction

## Features

The model uses transaction-level features including:

- Transaction amount
- Transaction hour
- Merchant category
- Foreign transaction status
- Location mismatch
- Device trust score
- Transaction velocity over the previous 24 hours
- Cardholder age

The web interface only asks the user for the transaction information that is meaningful to them. Internal risk features are simulated by the backend for demonstration purposes.

## Machine Learning

Algorithm:

- XGBoost Classifier

The dataset contains 10,000 transactions.

Class distribution:

- Legitimate: 98.49%
- Fraudulent: 1.51%

Because the dataset is imbalanced, `scale_pos_weight` was used during XGBoost training.

## Model Evaluation

The model was evaluated using:

- Precision
- Recall
- F1 Score
- ROC-AUC
- PR-AUC
- Confusion Matrix

The current synthetic dataset produced very high evaluation scores. These results should not be interpreted as real-world banking fraud detection performance.

## Technology Stack

### Machine Learning

- Python
- Pandas
- Scikit-learn
- XGBoost
- Joblib

### Backend

- FastAPI
- Pydantic
- Uvicorn

### Frontend

- HTML
- CSS
- JavaScript

## Project Structure

```text
credit-card-fraud-detection/
│
├── data/
│   └── credit_card_fraud_10k.csv
│
├── models/
│   ├── fraud_detection_model.pkl
│   └── preprocessor.pkl
│
├── src/
│   ├── app.py
│   ├── train.py
│   ├── predict.py
│   └── inspect_data.py
│
├── templates/
│   └── index.html
│
├── run.py
├── requirements.txt
├── .gitignore
└── README.md

from pathlib import Path

import joblib
import pandas as pd


# ============================================================
# PROJECT PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = (
    BASE_DIR
    / "models"
    / "fraud_detection_model.pkl"
)

PREPROCESSOR_PATH = (
    BASE_DIR
    / "models"
    / "preprocessor.pkl"
)


# ============================================================
# LOAD MODEL
# ============================================================

print("Loading model...")

model = joblib.load(MODEL_PATH)

preprocessor = joblib.load(
    PREPROCESSOR_PATH
)

print("Model loaded successfully!")


# ============================================================
# PREDICTION FUNCTION
# ============================================================

def predict_transaction(transaction):

    # Convert dictionary into DataFrame
    data = pd.DataFrame(
        [transaction]
    )

    # Preprocess transaction
    processed_data = preprocessor.transform(
        data
    )

    # Prediction
    prediction = model.predict(
        processed_data
    )[0]

    # Fraud probability
    probability = model.predict_proba(
        processed_data
    )[0][1]

    return prediction, probability


# ============================================================
# TEST TRANSACTION
# ============================================================

transaction = {

    "amount": 2500.00,

    "transaction_hour": 23,

    "merchant_category": "Electronics",

    "foreign_transaction": 1,

    "location_mismatch": 1,

    "device_trust_score": 35,

    "velocity_last_24h": 8,

    "cardholder_age": 24
}


# ============================================================
# RUN PREDICTION
# ============================================================

prediction, probability = predict_transaction(
    transaction
)


# ============================================================
# DISPLAY RESULT
# ============================================================

print("\n")
print("=" * 55)
print("       CREDIT CARD FRAUD DETECTION")
print("=" * 55)

print("\nTransaction:")

for key, value in transaction.items():

    print(
        f"{key:25}: {value}"
    )


print("\nPrediction:")

if prediction == 1:

    print("⚠ FRAUDULENT TRANSACTION")

else:

    print("✓ LEGITIMATE TRANSACTION")


print(
    f"\nFraud Probability: "
    f"{probability * 100:.2f}%"
)

print("\n" + "=" * 55)
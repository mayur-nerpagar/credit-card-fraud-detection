from pathlib import Path

import joblib
import pandas as pd

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel


# ============================================================
# PROJECT PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "models" / "fraud_detection_model.pkl"
PREPROCESSOR_PATH = BASE_DIR / "models" / "preprocessor.pkl"
HTML_PATH = BASE_DIR / "templates" / "index.html"


# ============================================================
# FASTAPI APPLICATION
# ============================================================

app = FastAPI(
    title="Credit Card Fraud Detection API",
    description="ML-based Credit Card Fraud Detection System",
    version="3.0.0"
)


# ============================================================
# LOAD MODEL
# ============================================================

print("=" * 60)
print("Loading Credit Card Fraud Detection Model...")
print("=" * 60)

try:
    model = joblib.load(MODEL_PATH)
    preprocessor = joblib.load(PREPROCESSOR_PATH)

    print("Model loaded successfully!")
    print("Preprocessor loaded successfully!")

except Exception as error:
    print("ERROR: Could not load model files.")
    print(error)
    raise


# ============================================================
# USER INPUT MODEL
# ============================================================

class Transaction(BaseModel):

    amount: float

    transaction_hour: int

    merchant_category: str

    foreign_transaction: int

    cardholder_age: int


# ============================================================
# INTERNAL FEATURE GENERATION
# ============================================================

def generate_internal_features(transaction: Transaction):
    """
    Generate internal risk features for demonstration.

    These features would normally come from banking,
    device, location and transaction-history systems.

    In this portfolio project they are simulated
    deterministically from the user's transaction data.
    """

    # --------------------------------------------------------
    # Location mismatch
    # --------------------------------------------------------
    #
    # In a real system this would compare:
    #
    # Current transaction location
    #          VS
    # Cardholder's usual location
    #
    # Here we simulate it.
    #
    # Foreign transactions have a higher chance of
    # being considered a location mismatch.
    # --------------------------------------------------------

    if transaction.foreign_transaction == 1:
        location_mismatch = 1
    else:
        location_mismatch = 0


    # --------------------------------------------------------
    # Device trust score
    # --------------------------------------------------------
    #
    # In a real system this could come from:
    #
    # - Device fingerprint
    # - Previously trusted devices
    # - Login history
    # - Suspicious device activity
    #
    # Here we create a deterministic demo value.
    # --------------------------------------------------------

    device_trust_score = 85

    if transaction.foreign_transaction == 1:
        device_trust_score -= 15

    if transaction.transaction_hour < 6:
        device_trust_score -= 10

    if transaction.transaction_hour >= 23:
        device_trust_score -= 10

    if transaction.amount > 2000:
        device_trust_score -= 10

    # Keep value between 0 and 100
    device_trust_score = max(
        0,
        min(100, device_trust_score)
    )


    # --------------------------------------------------------
    # Transaction velocity
    # --------------------------------------------------------
    #
    # In a real system this would come from transaction
    # history stored by the financial institution.
    #
    # Here we simulate a transaction count.
    # --------------------------------------------------------

    velocity_last_24h = 1

    if transaction.amount > 1500:
        velocity_last_24h += 2

    if transaction.foreign_transaction == 1:
        velocity_last_24h += 2

    if transaction.transaction_hour < 6:
        velocity_last_24h += 2

    if transaction.transaction_hour >= 23:
        velocity_last_24h += 1


    return {
        "location_mismatch": location_mismatch,
        "device_trust_score": device_trust_score,
        "velocity_last_24h": velocity_last_24h
    }


# ============================================================
# HOME PAGE
# ============================================================

@app.get("/", response_class=HTMLResponse)
def home():

    if not HTML_PATH.exists():

        return HTMLResponse(
            content="""
            <h1>Credit Card Fraud Detection</h1>
            <p>UI file not found.</p>
            """,
            status_code=404
        )

    return HTMLResponse(
        content=HTML_PATH.read_text(
            encoding="utf-8"
        )
    )


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/health")
def health():

    return {
        "status": "running",
        "application": "Credit Card Fraud Detection",
        "model": "XGBoost",
        "version": "3.0.0"
    }


# ============================================================
# PREDICTION ENDPOINT
# ============================================================

@app.post("/predict")
def predict(transaction: Transaction):

    # --------------------------------------------------------
    # Generate hidden/internal features
    # --------------------------------------------------------

    internal_features = generate_internal_features(
        transaction
    )


    # --------------------------------------------------------
    # Create complete transaction
    # --------------------------------------------------------

    transaction_data = {

        "amount":
            transaction.amount,

        "transaction_hour":
            transaction.transaction_hour,

        "merchant_category":
            transaction.merchant_category,

        "foreign_transaction":
            transaction.foreign_transaction,

        "location_mismatch":
            internal_features[
                "location_mismatch"
            ],

        "device_trust_score":
            internal_features[
                "device_trust_score"
            ],

        "velocity_last_24h":
            internal_features[
                "velocity_last_24h"
            ],

        "cardholder_age":
            transaction.cardholder_age
    }


    # --------------------------------------------------------
    # Convert to DataFrame
    # --------------------------------------------------------

    data = pd.DataFrame(
        [transaction_data]
    )


    # --------------------------------------------------------
    # Apply saved preprocessing
    # --------------------------------------------------------

    processed_data = preprocessor.transform(
        data
    )


    # --------------------------------------------------------
    # Model prediction
    # --------------------------------------------------------

    prediction = model.predict(
        processed_data
    )[0]


    # --------------------------------------------------------
    # Fraud probability
    # --------------------------------------------------------

    fraud_probability = model.predict_proba(
        processed_data
    )[0][1]


    prediction = int(prediction)

    fraud_probability = float(
        fraud_probability
    )


    # --------------------------------------------------------
    # Human-readable result
    # --------------------------------------------------------

    if prediction == 1:

        result = "Fraudulent Transaction"

    else:

        result = "Legitimate Transaction"


    # --------------------------------------------------------
    # Risk level
    # --------------------------------------------------------

    if fraud_probability >= 0.75:

        risk_level = "HIGH"

    elif fraud_probability >= 0.40:

        risk_level = "MEDIUM"

    else:

        risk_level = "LOW"


    # --------------------------------------------------------
    # Return response
    # --------------------------------------------------------

    return {

        "prediction": prediction,

        "result": result,

        "fraud_probability": round(
            fraud_probability * 100,
            2
        ),

        "risk_level": risk_level

    }
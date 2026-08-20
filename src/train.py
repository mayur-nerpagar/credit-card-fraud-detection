from pathlib import Path

import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from xgboost import XGBClassifier


# ============================================================
# PROJECT PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = BASE_DIR / "data" / "credit_card_fraud_10k.csv"

MODEL_DIR = BASE_DIR / "models"

MODEL_PATH = MODEL_DIR / "fraud_detection_model.pkl"

PREPROCESSOR_PATH = MODEL_DIR / "preprocessor.pkl"

MODEL_DIR.mkdir(exist_ok=True)


# ============================================================
# LOAD DATASET
# ============================================================

print("=" * 65)
print("       CREDIT CARD FRAUD DETECTION")
print("              MODEL TRAINING")
print("=" * 65)

print("\nLoading dataset...")

df = pd.read_csv(DATA_PATH)

print(f"Dataset shape: {df.shape}")


# ============================================================
# BASIC VALIDATION
# ============================================================

required_columns = [
    "transaction_id",
    "amount",
    "transaction_hour",
    "merchant_category",
    "foreign_transaction",
    "location_mismatch",
    "device_trust_score",
    "velocity_last_24h",
    "cardholder_age",
    "is_fraud"
]

missing_columns = [
    column
    for column in required_columns
    if column not in df.columns
]

if missing_columns:

    raise ValueError(
        f"Missing required columns: {missing_columns}"
    )


# ============================================================
# REMOVE TRANSACTION ID
# ============================================================

# transaction_id identifies a transaction but does not
# represent a useful predictive feature.

df = df.drop(
    columns=["transaction_id"]
)


# ============================================================
# SEPARATE FEATURES AND TARGET
# ============================================================

TARGET = "is_fraud"

X = df.drop(
    columns=[TARGET]
)

y = df[TARGET]


# ============================================================
# DISPLAY CLASS DISTRIBUTION
# ============================================================

print("\nTarget distribution:")

print(y.value_counts())

print("\nTarget percentage:")

print(
    (y.value_counts(normalize=True) * 100).round(2)
)


# ============================================================
# IDENTIFY FEATURES
# ============================================================

categorical_features = [
    "merchant_category"
]

numerical_features = [
    "amount",
    "transaction_hour",
    "foreign_transaction",
    "location_mismatch",
    "device_trust_score",
    "velocity_last_24h",
    "cardholder_age"
]


print("\nNumerical features:")

for feature in numerical_features:
    print(f"  - {feature}")


print("\nCategorical features:")

for feature in categorical_features:
    print(f"  - {feature}")


# ============================================================
# TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(

    X,
    y,

    test_size=0.20,

    random_state=42,

    stratify=y
)


print("\nTraining samples:", len(X_train))

print("Testing samples:", len(X_test))


# ============================================================
# CATEGORICAL PREPROCESSING
# ============================================================

categorical_pipeline = Pipeline(
    steps=[

        (
            "imputer",
            SimpleImputer(
                strategy="most_frequent"
            )
        ),

        (
            "encoder",
            OneHotEncoder(
                handle_unknown="ignore"
            )
        )
    ]
)


# ============================================================
# NUMERICAL PREPROCESSING
# ============================================================

numerical_pipeline = Pipeline(
    steps=[

        (
            "imputer",
            SimpleImputer(
                strategy="median"
            )
        )
    ]
)


# ============================================================
# COMBINED PREPROCESSOR
# ============================================================

preprocessor = ColumnTransformer(

    transformers=[

        (
            "numerical",
            numerical_pipeline,
            numerical_features
        ),

        (
            "categorical",
            categorical_pipeline,
            categorical_features
        )
    ]
)


# ============================================================
# PREPROCESS DATA
# ============================================================

print("\nPreprocessing data...")

X_train_processed = preprocessor.fit_transform(
    X_train
)

X_test_processed = preprocessor.transform(
    X_test
)

print("Preprocessing completed.")


# ============================================================
# HANDLE CLASS IMBALANCE
# ============================================================

negative_count = (y_train == 0).sum()

positive_count = (y_train == 1).sum()

scale_pos_weight = (
    negative_count / positive_count
)


print(
    f"\nNegative samples: {negative_count}"
)

print(
    f"Positive samples: {positive_count}"
)

print(
    f"Scale positive weight: "
    f"{scale_pos_weight:.2f}"
)


# ============================================================
# XGBOOST MODEL
# ============================================================

print("\nTraining XGBoost model...")

model = XGBClassifier(

    n_estimators=300,

    max_depth=5,

    learning_rate=0.05,

    subsample=0.8,

    colsample_bytree=0.8,

    objective="binary:logistic",

    eval_metric="logloss",

    scale_pos_weight=scale_pos_weight,

    random_state=42,

    n_jobs=-1
)


model.fit(
    X_train_processed,
    y_train
)


print("Model training completed.")


# ============================================================
# PREDICTIONS
# ============================================================

y_pred = model.predict(
    X_test_processed
)

y_probability = model.predict_proba(
    X_test_processed
)[:, 1]


# ============================================================
# MODEL METRICS
# ============================================================

precision = precision_score(
    y_test,
    y_pred,
    zero_division=0
)

recall = recall_score(
    y_test,
    y_pred,
    zero_division=0
)

f1 = f1_score(
    y_test,
    y_pred,
    zero_division=0
)

roc_auc = roc_auc_score(
    y_test,
    y_probability
)

pr_auc = average_precision_score(
    y_test,
    y_probability
)


# ============================================================
# DISPLAY RESULTS
# ============================================================

print("\n")
print("=" * 65)
print("                 MODEL EVALUATION")
print("=" * 65)

print(
    f"\nPrecision : {precision:.4f}"
)

print(
    f"Recall    : {recall:.4f}"
)

print(
    f"F1 Score  : {f1:.4f}"
)

print(
    f"ROC-AUC   : {roc_auc:.4f}"
)

print(
    f"PR-AUC    : {pr_auc:.4f}"
)


print("\nClassification Report:")

print(
    classification_report(
        y_test,
        y_pred,
        target_names=[
            "Legitimate",
            "Fraud"
        ],
        zero_division=0
    )
)


print("Confusion Matrix:")

print(
    confusion_matrix(
        y_test,
        y_pred
    )
)


# ============================================================
# SAVE MODEL
# ============================================================

print("\nSaving model...")

joblib.dump(
    model,
    MODEL_PATH
)

joblib.dump(
    preprocessor,
    PREPROCESSOR_PATH
)


print("\nModel saved successfully!")

print(
    f"Model      : {MODEL_PATH}"
)

print(
    f"Preprocessor: {PREPROCESSOR_PATH}"
)


print("\n" + "=" * 65)
print("              TRAINING COMPLETED")
print("=" * 65)
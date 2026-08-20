from pathlib import Path
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = BASE_DIR / "data" / "credit_card_fraud_10k.csv"


print("=" * 60)
print("CREDIT CARD FRAUD DETECTION")
print("DATASET INSPECTION")
print("=" * 60)


if not DATA_PATH.exists():

    print("\nERROR: Dataset not found!")
    print(f"Expected file: {DATA_PATH}")
    exit()


df = pd.read_csv(DATA_PATH)


print("\nDataset loaded successfully!")

print(f"\nRows: {df.shape[0]:,}")
print(f"Columns: {df.shape[1]}")


print("\nColumn Names:")

for column in df.columns:
    print(f" - {column}")


print("\nFirst 5 rows:")

print(df.head())


print("\nData Types:")

print(df.dtypes)


print("\nMissing Values:")

print(df.isnull().sum())


if "is_fraud" in df.columns:

    print("\nFraud Distribution:")

    print(df["is_fraud"].value_counts())

    print("\nFraud Percentage:")

    print(
        (df["is_fraud"].value_counts(normalize=True) * 100)
        .round(2)
    )


print("\n" + "=" * 60)
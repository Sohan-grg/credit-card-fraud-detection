
from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "creditcard.csv"

df = pd.read_csv(DATA_PATH)

print("Dataset shape:", df.shape)
print("\nFirst five transactions:")
print(df.head())

print("\nMissing values:", df.isnull().sum().sum())

print("\nClass distribution:")
print(df["Class"].value_counts())

print("\nTransaction amount statistics:")
print(df["Amount"].describe())

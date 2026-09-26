import os
import pandas as pd
from sklearn.model_selection import train_test_split


# ============================================================
# Configuration
# ============================================================

INPUT_FILE = "data/US_Accidents_1M.csv"

TRAIN_FILE = "data/train.csv"
TEST_FILE = "data/test.csv"

REPORT_FILE = "outputs/reports/train_test_distribution.csv"

TEST_SIZE = 0.20
RANDOM_SEED = 42


# ============================================================
# Load dataset
# ============================================================

print("Loading dataset...")

df = pd.read_csv(INPUT_FILE)

print(f"Dataset shape: {df.shape}")


# ============================================================
# Separate target and predictors
# ============================================================

TARGET = "Severity"

X = df.drop(columns=[TARGET])
y = df[TARGET]


# ============================================================
# Stratified train/test split
# ============================================================

print("\nCreating stratified train/test split...")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=TEST_SIZE,
    stratify=y,
    random_state=RANDOM_SEED
)


# ============================================================
# Reconstruct train and test datasets
# ============================================================

train_df = X_train.copy()
train_df[TARGET] = y_train

test_df = X_test.copy()
test_df[TARGET] = y_test


# ============================================================
# Create output directories
# ============================================================

os.makedirs("data", exist_ok=True)
os.makedirs("outputs/reports", exist_ok=True)


# ============================================================
# Save train/test datasets
# ============================================================

train_df.to_csv(TRAIN_FILE, index=False)
test_df.to_csv(TEST_FILE, index=False)


# ============================================================
# Verify target distributions
# ============================================================

full_distribution = (
    y.value_counts(normalize=True)
    .sort_index()
    .mul(100)
)

train_distribution = (
    y_train.value_counts(normalize=True)
    .sort_index()
    .mul(100)
)

test_distribution = (
    y_test.value_counts(normalize=True)
    .sort_index()
    .mul(100)
)


distribution_report = pd.DataFrame({
    "Full_Percentage": full_distribution,
    "Train_Percentage": train_distribution,
    "Test_Percentage": test_distribution,
})

distribution_report.index.name = "Severity"

distribution_report["Full_Count"] = y.value_counts().sort_index()
distribution_report["Train_Count"] = y_train.value_counts().sort_index()
distribution_report["Test_Count"] = y_test.value_counts().sort_index()

distribution_report = distribution_report[
    [
        "Full_Count",
        "Train_Count",
        "Test_Count",
        "Full_Percentage",
        "Train_Percentage",
        "Test_Percentage",
    ]
]

distribution_report.to_csv(REPORT_FILE)


# ============================================================
# Print results
# ============================================================

print("\nTrain/Test split completed.")

print(f"\nTraining set shape: {train_df.shape}")
print(f"Testing set shape:  {test_df.shape}")

print("\nSeverity distribution:")
print(distribution_report.to_string(float_format=lambda x: f"{x:.4f}"))

print("\nFiles created:")
print(f"- {TRAIN_FILE}")
print(f"- {TEST_FILE}")
print(f"- {REPORT_FILE}")
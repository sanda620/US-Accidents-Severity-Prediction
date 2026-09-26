import os
import pandas as pd


# ============================================================
# Configuration
# ============================================================

TRAIN_FILE = "data/train.csv"
TEST_FILE = "data/test.csv"

OUTPUT_FILE = "outputs/reports/duplicate_analysis.csv"


# ============================================================
# Setup
# ============================================================

os.makedirs("outputs/reports", exist_ok=True)

print("=" * 70)
print("STAGE 4.5 - DUPLICATE RECORD ANALYSIS")
print("=" * 70)


# ============================================================
# Load datasets
# ============================================================

print("\nLoading training dataset...")
train = pd.read_csv(TRAIN_FILE)

print(f"Training dataset shape: {train.shape}")

print("\nLoading testing dataset...")
test = pd.read_csv(TEST_FILE)

print(f"Testing dataset shape: {test.shape}")


# ============================================================
# Exact duplicate analysis
# ============================================================

print("\n" + "=" * 70)
print("EXACT DUPLICATE ANALYSIS")
print("=" * 70)

train_duplicates = int(train.duplicated().sum())
test_duplicates = int(test.duplicated().sum())

print(f"\nTraining exact duplicate rows: {train_duplicates:,}")
print(f"Testing exact duplicate rows : {test_duplicates:,}")


# ============================================================
# Duplicate ID analysis
# ============================================================

print("\n" + "=" * 70)
print("DUPLICATE ID ANALYSIS")
print("=" * 70)

train_duplicate_ids = int(train["ID"].duplicated().sum())
test_duplicate_ids = int(test["ID"].duplicated().sum())

print(f"\nTraining duplicate IDs: {train_duplicate_ids:,}")
print(f"Testing duplicate IDs : {test_duplicate_ids:,}")


# ============================================================
# Train/Test ID overlap
# ============================================================

print("\n" + "=" * 70)
print("TRAIN / TEST RECORD OVERLAP")
print("=" * 70)

train_ids = set(train["ID"])
test_ids = set(test["ID"])

overlap_count = len(train_ids.intersection(test_ids))

print(f"\nIDs appearing in both training and testing sets: {overlap_count:,}")


# ============================================================
# Create report
# ============================================================

report = pd.DataFrame({
    "Check": [
        "Training exact duplicate rows",
        "Testing exact duplicate rows",
        "Training duplicate IDs",
        "Testing duplicate IDs",
        "Train/Test overlapping IDs"
    ],
    "Count": [
        train_duplicates,
        test_duplicates,
        train_duplicate_ids,
        test_duplicate_ids,
        overlap_count
    ],
    "Treatment": [
        "No removal required" if train_duplicates == 0
        else "Remove duplicates",
        "No removal required" if test_duplicates == 0
        else "Remove duplicates",
        "No removal required" if train_duplicate_ids == 0
        else "Investigate duplicate IDs",
        "No removal required" if test_duplicate_ids == 0
        else "Investigate duplicate IDs",
        "No leakage detected" if overlap_count == 0
        else "Investigate train/test overlap"
    ]
})


# ============================================================
# Save report
# ============================================================

report.to_csv(OUTPUT_FILE, index=False)

print("\n" + "=" * 70)
print("DUPLICATE ANALYSIS COMPLETED")
print("=" * 70)

print(f"\nSaved report: {OUTPUT_FILE}")
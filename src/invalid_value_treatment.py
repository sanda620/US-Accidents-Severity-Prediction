import pandas as pd
from pathlib import Path

# ============================================================
# Stage 4.4.2 - Invalid Value Treatment
# ============================================================

TRAIN_FILE = Path("data/train_missing_handled.csv")
TEST_FILE = Path("data/test_missing_handled.csv")

OUTPUT_DIR = Path("outputs/reports")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_OUTPUT = Path("data/train_outliers_handled.csv")
TEST_OUTPUT = Path("data/test_outliers_handled.csv")

print("=" * 70)
print("STAGE 4.4.2 - INVALID VALUE TREATMENT")
print("=" * 70)

# ------------------------------------------------------------
# 1. Load datasets
# ------------------------------------------------------------

train = pd.read_csv(TRAIN_FILE)
test = pd.read_csv(TEST_FILE)

print(f"\nTraining shape: {train.shape}")
print(f"Testing shape : {test.shape}")

# ------------------------------------------------------------
# 2. Define domain-based invalid-value rules
# ------------------------------------------------------------

invalid_rules = {
    "Temperature(F)": (-80, 140),
    "Wind_Speed(mph)": (0, 100),
    "Pressure(in)": (20, 35)
}

# ------------------------------------------------------------
# 3. Calculate replacement values from TRAINING DATA ONLY
# ------------------------------------------------------------

replacement_values = {}

for feature in invalid_rules:

    median_value = train[feature].median()

    replacement_values[feature] = median_value

    print(
        f"\n{feature}"
        f"\nTraining median: {median_value}"
    )

# ------------------------------------------------------------
# 4. Replace invalid values with NaN
# ------------------------------------------------------------

treatment_results = []

for feature, (lower, upper) in invalid_rules.items():

    train_invalid = (
        (train[feature] < lower) |
        (train[feature] > upper)
    )

    test_invalid = (
        (test[feature] < lower) |
        (test[feature] > upper)
    )

    train_count = int(train_invalid.sum())
    test_count = int(test_invalid.sum())

    # Replace invalid values with NaN
    train.loc[train_invalid, feature] = pd.NA
    test.loc[test_invalid, feature] = pd.NA

    # Fill using TRAINING median
    median_value = replacement_values[feature]

    train[feature] = train[feature].fillna(median_value)
    test[feature] = test[feature].fillna(median_value)

    treatment_results.append({
        "Feature": feature,
        "Lower_Bound": lower,
        "Upper_Bound": upper,
        "Training_Invalid_Count": train_count,
        "Testing_Invalid_Count": test_count,
        "Training_Median_Used": median_value,
        "Treatment": "Replace invalid values with training median"
    })

# ------------------------------------------------------------
# 5. Verify no invalid values remain
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("VERIFICATION")
print("=" * 70)

for feature, (lower, upper) in invalid_rules.items():

    train_remaining = (
        (train[feature] < lower) |
        (train[feature] > upper)
    ).sum()

    test_remaining = (
        (test[feature] < lower) |
        (test[feature] > upper)
    ).sum()

    print(
        f"{feature:<22} "
        f"Train remaining: {train_remaining} | "
        f"Test remaining: {test_remaining}"
    )

# ------------------------------------------------------------
# 6. Save datasets
# ------------------------------------------------------------

train.to_csv(TRAIN_OUTPUT, index=False)
test.to_csv(TEST_OUTPUT, index=False)

print(f"\nSaved training data: {TRAIN_OUTPUT}")
print(f"Saved testing data : {TEST_OUTPUT}")

# ------------------------------------------------------------
# 7. Save treatment report
# ------------------------------------------------------------

results_df = pd.DataFrame(treatment_results)

report_file = OUTPUT_DIR / "invalid_value_treatment_results.csv"

results_df.to_csv(
    report_file,
    index=False
)

print(f"Saved treatment report: {report_file}")

print("\n" + "=" * 70)
print("INVALID VALUE TREATMENT COMPLETED")
print("=" * 70)
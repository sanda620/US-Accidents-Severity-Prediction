import os
import pandas as pd


# ============================================================
# Configuration
# ============================================================

TRAIN_INPUT = "data/train.csv"
TEST_INPUT = "data/test.csv"

TRAIN_OUTPUT = "data/train_missing_handled.csv"
TEST_OUTPUT = "data/test_missing_handled.csv"

REPORT_OUTPUT = (
    "outputs/reports/missing_value_treatment_results.csv"
)


# ============================================================
# Feature groups
# ============================================================

TARGET = "Severity"

# Potential post-event leakage features
LEAKAGE_FEATURES = [
    "End_Lat",
    "End_Lng",
]

# Features not included in the initial pipeline
UNRESOLVED_FEATURES = [
    "Weather_Timestamp",
    "Description",
]

# Numerical features requiring missing indicators
NUMERIC_INDICATOR_FEATURES = [
    "Precipitation(in)",
    "Wind_Chill(F)",
    "Wind_Speed(mph)",
]

# Numerical features using median imputation
NUMERIC_MEDIAN_FEATURES = [
    "Precipitation(in)",
    "Wind_Chill(F)",
    "Wind_Speed(mph)",
    "Visibility(mi)",
    "Humidity(%)",
    "Temperature(F)",
    "Pressure(in)",
]

# Categorical features using an explicit missing category
CATEGORICAL_FEATURES = [
    "Weather_Condition",
    "Wind_Direction",
    "Airport_Code",
    "Street",
    "Timezone",
    "Zipcode",
    "City",
    "Sunrise_Sunset",
    "Civil_Twilight",
    "Nautical_Twilight",
    "Astronomical_Twilight",
]


# ============================================================
# Load datasets
# ============================================================

print("Loading training dataset...")

train_df = pd.read_csv(TRAIN_INPUT)

print(f"Training dataset shape: {train_df.shape}")

print("\nLoading testing dataset...")

test_df = pd.read_csv(TEST_INPUT)

print(f"Testing dataset shape: {test_df.shape}")


# ============================================================
# Remove leakage and unresolved features
# ============================================================

features_to_remove = (
    LEAKAGE_FEATURES + UNRESOLVED_FEATURES
)

print("\nRemoving features from the initial pipeline:")

for feature in features_to_remove:
    print(f"- {feature}")

train_df = train_df.drop(
    columns=features_to_remove,
    errors="ignore"
)

test_df = test_df.drop(
    columns=features_to_remove,
    errors="ignore"
)


# ============================================================
# Create missingness indicators
# ============================================================

print("\nCreating missingness indicators...")

for feature in NUMERIC_INDICATOR_FEATURES:

    indicator_name = feature.replace("(", "_").replace(")", "")
    indicator_name = indicator_name.replace("/", "_")
    indicator_name = indicator_name.replace(" ", "_")
    indicator_name = f"{indicator_name}_Missing"

    train_df[indicator_name] = (
        train_df[feature].isna().astype("int8")
    )

    test_df[indicator_name] = (
        test_df[feature].isna().astype("int8")
    )

    print(f"- Created {indicator_name}")


# ============================================================
# Calculate medians using TRAINING data only
# ============================================================

print("\nCalculating training-set medians...")

training_medians = {}

for feature in NUMERIC_MEDIAN_FEATURES:

    median_value = train_df[feature].median()

    training_medians[feature] = median_value

    print(
        f"- {feature}: "
        f"{median_value:.6f}"
    )


# ============================================================
# Apply numerical median imputation
# ============================================================

print("\nApplying numerical median imputation...")

for feature, median_value in training_medians.items():

    train_df[feature] = train_df[feature].fillna(
        median_value
    )

    test_df[feature] = test_df[feature].fillna(
        median_value
    )

    print(f"- Imputed {feature}")


# ============================================================
# Apply categorical missing-category treatment
# ============================================================

print("\nApplying categorical missing-category treatment...")

for feature in CATEGORICAL_FEATURES:

    train_df[feature] = train_df[feature].fillna(
        "Missing"
    )

    test_df[feature] = test_df[feature].fillna(
        "Missing"
    )

    print(f"- Processed {feature}")


# ============================================================
# Verify remaining missing values
# ============================================================

print("\nChecking remaining missing values...")

train_missing = train_df.isna().sum()
test_missing = test_df.isna().sum()

remaining_train_missing = (
    train_missing[train_missing > 0]
    .sort_values(ascending=False)
)

remaining_test_missing = (
    test_missing[test_missing > 0]
    .sort_values(ascending=False)
)

print("\nRemaining missing values in training data:")

if remaining_train_missing.empty:
    print("None")

else:
    print(remaining_train_missing.to_string())


print("\nRemaining missing values in testing data:")

if remaining_test_missing.empty:
    print("None")

else:
    print(remaining_test_missing.to_string())


# ============================================================
# Create treatment results report
# ============================================================

report_rows = []

for feature in train_df.columns:

    train_missing_count = train_df[feature].isna().sum()
    test_missing_count = test_df[feature].isna().sum()

    report_rows.append({
        "Feature": feature,
        "Train_Missing_Count": train_missing_count,
        "Test_Missing_Count": test_missing_count,
        "Train_Missing_Percentage": (
            train_missing_count / len(train_df)
        ) * 100,
        "Test_Missing_Percentage": (
            test_missing_count / len(test_df)
        ) * 100,
    })

treatment_results = pd.DataFrame(report_rows)


# ============================================================
# Save processed datasets and report
# ============================================================

os.makedirs("data", exist_ok=True)
os.makedirs("outputs/reports", exist_ok=True)

print("\nSaving processed datasets...")

train_df.to_csv(TRAIN_OUTPUT, index=False)
test_df.to_csv(TEST_OUTPUT, index=False)

treatment_results.to_csv(
    REPORT_OUTPUT,
    index=False
)


# ============================================================
# Final output
# ============================================================

print("\nMissing-value treatment completed.")

print(f"\nProcessed training shape: {train_df.shape}")
print(f"Processed testing shape:  {test_df.shape}")

print("\nFiles created:")
print(f"- {TRAIN_OUTPUT}")
print(f"- {TEST_OUTPUT}")
print(f"- {REPORT_OUTPUT}")
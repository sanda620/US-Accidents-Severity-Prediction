import os
import numpy as np
import pandas as pd


# ============================================================
# Configuration
# ============================================================

TRAIN_FILE = "data/train_outliers_handled.csv"
TEST_FILE = "data/test_outliers_handled.csv"

TRAIN_OUTPUT = "data/train_temporal_features.csv"
TEST_OUTPUT = "data/test_temporal_features.csv"

REPORT_FILE = "outputs/reports/temporal_feature_engineering.csv"


# ============================================================
# Setup
# ============================================================

os.makedirs("outputs/reports", exist_ok=True)

print("=" * 70)
print("STAGE 4.6 - TEMPORAL FEATURE ENGINEERING")
print("=" * 70)


# ============================================================
# Temporal feature creation function
# ============================================================

def create_temporal_features(df, dataset_name):
    print(f"\nProcessing {dataset_name} dataset...")

    if "Start_Time" not in df.columns:
        raise ValueError("Start_Time column is missing from the dataset.")

    # --------------------------------------------------------
    # Parse mixed timestamp formats
    # --------------------------------------------------------

    start_time = pd.to_datetime(
        df["Start_Time"].astype(str).str.strip(),
        format="mixed",
        errors="coerce"
    )

    failed = int(start_time.isna().sum())

    print(f"Timestamp parsing failures: {failed:,}")

    if failed > 0:
        raise ValueError(
            f"{failed:,} Start_Time values could not be parsed."
        )

    # --------------------------------------------------------
    # Basic calendar features
    # --------------------------------------------------------

    df["Start_Year"] = start_time.dt.year
    df["Start_Month"] = start_time.dt.month
    df["Start_Day"] = start_time.dt.day
    df["Start_Hour"] = start_time.dt.hour
    df["Start_DayOfWeek"] = start_time.dt.dayofweek

    # --------------------------------------------------------
    # Weekend indicator
    # Monday = 0 ... Sunday = 6
    # --------------------------------------------------------

    df["Is_Weekend"] = (
        df["Start_DayOfWeek"] >= 5
    ).astype(int)

    # --------------------------------------------------------
    # Night-time indicator
    #
    # Night is defined as:
    # 00:00-05:59 and 20:00-23:59
    # --------------------------------------------------------

    df["Is_Night"] = (
        (df["Start_Hour"] < 6) |
        (df["Start_Hour"] >= 20)
    ).astype(int)

    # --------------------------------------------------------
    # Cyclical hour encoding
    #
    # Represents the circular relationship between
    # 23:00 and 00:00.
    # --------------------------------------------------------

    df["Hour_Sin"] = np.sin(
        2 * np.pi * df["Start_Hour"] / 24
    )

    df["Hour_Cos"] = np.cos(
        2 * np.pi * df["Start_Hour"] / 24
    )

    # --------------------------------------------------------
    # Cyclical month encoding
    #
    # Represents December and January as adjacent months.
    # --------------------------------------------------------

    df["Month_Sin"] = np.sin(
        2 * np.pi * (df["Start_Month"] - 1) / 12
    )

    df["Month_Cos"] = np.cos(
        2 * np.pi * (df["Start_Month"] - 1) / 12
    )

    return df, start_time


# ============================================================
# Load training data
# ============================================================

print("\nLoading training dataset...")

train = pd.read_csv(TRAIN_FILE)

print(f"Training dataset shape: {train.shape}")


# ============================================================
# Load testing data
# ============================================================

print("\nLoading testing dataset...")

test = pd.read_csv(TEST_FILE)

print(f"Testing dataset shape: {test.shape}")


# ============================================================
# Create temporal features
# ============================================================

train, train_start_time = create_temporal_features(
    train,
    "training"
)

test, test_start_time = create_temporal_features(
    test,
    "testing"
)


# ============================================================
# Validate temporal ranges
# ============================================================

print("\n" + "=" * 70)
print("TEMPORAL FEATURE VALIDATION")
print("=" * 70)

temporal_columns = [
    "Start_Year",
    "Start_Month",
    "Start_Day",
    "Start_Hour",
    "Start_DayOfWeek",
    "Is_Weekend",
    "Is_Night",
    "Hour_Sin",
    "Hour_Cos",
    "Month_Sin",
    "Month_Cos"
]

print("\nTraining temporal feature summary:")

summary = train[temporal_columns].describe().T

print(summary)


# ============================================================
# Validate no missing values
# ============================================================

train_missing = train[temporal_columns].isna().sum().sum()
test_missing = test[temporal_columns].isna().sum().sum()

print("\nMissing temporal feature values:")
print(f"Training: {train_missing:,}")
print(f"Testing : {test_missing:,}")


if train_missing > 0 or test_missing > 0:
    raise ValueError(
        "Missing values detected in engineered temporal features."
    )


# ============================================================
# Validate year distribution
# ============================================================

print("\nTraining year distribution:")
print(
    train["Start_Year"]
    .value_counts()
    .sort_index()
)

print("\nTesting year distribution:")
print(
    test["Start_Year"]
    .value_counts()
    .sort_index()
)


# ============================================================
# Validate hour distribution
# ============================================================

print("\nTraining hour distribution:")
print(
    train["Start_Hour"]
    .value_counts()
    .sort_index()
)


# ============================================================
# Validate weekend/night features
# ============================================================

print("\nWeekend distribution:")
print(
    train["Is_Weekend"]
    .value_counts()
    .sort_index()
)

print("\nNight distribution:")
print(
    train["Is_Night"]
    .value_counts()
    .sort_index()
)


# ============================================================
# Save datasets
# ============================================================

print("\nSaving engineered datasets...")

train.to_csv(
    TRAIN_OUTPUT,
    index=False
)

test.to_csv(
    TEST_OUTPUT,
    index=False
)


# ============================================================
# Create report
# ============================================================

report_rows = []

for feature in temporal_columns:

    report_rows.append({
        "Feature": feature,
        "Data_Type": str(train[feature].dtype),
        "Unique_Values": int(train[feature].nunique()),
        "Missing_Count": int(train[feature].isna().sum()),
        "Training_Min": train[feature].min(),
        "Training_Max": train[feature].max(),
        "Description": {
            "Start_Year":
                "Year of accident start time",
            "Start_Month":
                "Month of accident start time",
            "Start_Day":
                "Day of month of accident start time",
            "Start_Hour":
                "Hour of accident start time",
            "Start_DayOfWeek":
                "Day of week; Monday=0, Sunday=6",
            "Is_Weekend":
                "Indicator for Saturday or Sunday",
            "Is_Night":
                "Indicator for nighttime accident",
            "Hour_Sin":
                "Cyclical sine encoding of hour",
            "Hour_Cos":
                "Cyclical cosine encoding of hour",
            "Month_Sin":
                "Cyclical sine encoding of month",
            "Month_Cos":
                "Cyclical cosine encoding of month"
        }[feature]
    })


report = pd.DataFrame(report_rows)

report.to_csv(
    REPORT_FILE,
    index=False
)


# ============================================================
# Final summary
# ============================================================

print("\n" + "=" * 70)
print("TEMPORAL FEATURE ENGINEERING COMPLETED")
print("=" * 70)

print("\nCreated features:")

for feature in temporal_columns:
    print(f"- {feature}")

print("\nFiles created:")
print(f"- {TRAIN_OUTPUT}")
print(f"- {TEST_OUTPUT}")
print(f"- {REPORT_FILE}")

print("\nFinal dataset shapes:")
print(f"Training: {train.shape}")
print(f"Testing : {test.shape}")

print("=" * 70)
"""
STAGE 4.10 - FINAL FEATURE SELECTION / MODELING DATASET PREPARATION

Purpose:
- Load numerically transformed train/test datasets.
- Remove identifiers and constant/non-predictive features.
- Separate target variable.
- Verify train/test feature consistency.
- Check for obvious leakage features.
- Validate missing/infinite values.
- Save final modeling datasets and feature-selection report.
"""

import os
import numpy as np
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

TRAIN_INPUT = "data/train_numerical_features.csv"
TEST_INPUT = "data/test_numerical_features.csv"

TRAIN_OUTPUT = "data/train_final.csv"
TEST_OUTPUT = "data/test_final.csv"

REPORT_OUTPUT = "outputs/reports/final_feature_selection.csv"

TARGET = "Severity"

# Features that should not be used for prediction
EXCLUDE_FEATURES = [
    "ID",
    "Country",
    "Turning_Loop",
]

# Features representing information that may become available
# after the accident event or otherwise create leakage.
LEAKAGE_FEATURES = [
    "End_Lat",
    "End_Lng",
    "End_Time",
    "Description",
]

# Raw timestamp was already removed in Stage 4.9, but include it
# here as a safety check.
RAW_TIMESTAMP_FEATURES = [
    "Start_Time",
    "End_Time",
]


# ============================================================
# HELPERS
# ============================================================

def check_missing_and_infinite(df, name):
    """Check missing and infinite values."""

    missing = int(df.isna().sum().sum())

    numeric_df = df.select_dtypes(include=[np.number])

    if numeric_df.shape[1] > 0:
        infinite = int(
            np.isinf(numeric_df.to_numpy()).sum()
        )
    else:
        infinite = 0

    print(f"{name} missing values   : {missing}")
    print(f"{name} infinite values  : {infinite}")

    return missing, infinite


def feature_category(feature):
    """Classify a feature for the report."""

    if feature == TARGET:
        return "TARGET"

    if feature in EXCLUDE_FEATURES:
        return "EXCLUDE"

    if feature in LEAKAGE_FEATURES:
        return "LEAKAGE"

    if feature.endswith("_log1p"):
        return "TRANSFORMED_NUMERICAL"

    if feature.endswith("_Missing"):
        return "MISSING_INDICATOR"

    if feature.startswith(("Start_", "Is_", "Hour_", "Month_")):
        return "TEMPORAL"

    if feature.startswith(
        (
            "Source_",
            "State_",
            "Timezone_",
            "Wind_Direction_",
            "Sunrise_Sunset_",
            "Civil_Twilight_",
            "Nautical_Twilight_",
            "Astronomical_Twilight_",
        )
    ):
        return "ONE_HOT"

    if feature in [
        "Street",
        "City",
        "County",
        "Zipcode",
        "Airport_Code",
        "Weather_Condition",
    ]:
        return "FREQUENCY_ENCODED"

    return "NUMERICAL"


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("STAGE 4.10 - FINAL FEATURE SELECTION")
    print("=" * 70)

    os.makedirs("outputs/reports", exist_ok=True)

    # --------------------------------------------------------
    # 1. LOAD DATA
    # --------------------------------------------------------

    print("\nLoading training dataset...")
    train = pd.read_csv(TRAIN_INPUT)

    print("Loading testing dataset...")
    test = pd.read_csv(TEST_INPUT)

    print(f"\nTraining shape: {train.shape}")
    print(f"Testing shape : {test.shape}")

    # --------------------------------------------------------
    # 2. VERIFY TARGET
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("1. TARGET VALIDATION")
    print("=" * 70)

    if TARGET not in train.columns:
        raise ValueError(
            f"Target column '{TARGET}' not found in training dataset."
        )

    if TARGET not in test.columns:
        raise ValueError(
            f"Target column '{TARGET}' not found in testing dataset."
        )

    print(f"Target column: {TARGET}")

    print("\nTraining target distribution:")
    print(train[TARGET].value_counts().sort_index())

    print("\nTesting target distribution:")
    print(test[TARGET].value_counts().sort_index())

    # --------------------------------------------------------
    # 3. IDENTIFY FEATURES
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("2. FEATURE IDENTIFICATION")
    print("=" * 70)

    all_columns = list(train.columns)

    predictor_columns = [
        col for col in all_columns
        if col != TARGET
    ]

    print(f"\nInitial predictor count: {len(predictor_columns)}")

    # --------------------------------------------------------
    # 4. CHECK EXCLUSION FEATURES
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("3. EXCLUSION / LEAKAGE CHECK")
    print("=" * 70)

    existing_exclusions = [
        col for col in EXCLUDE_FEATURES
        if col in train.columns
    ]

    existing_leakage = [
        col for col in LEAKAGE_FEATURES
        if col in train.columns
    ]

    print("\nFeatures marked for exclusion:")

    if existing_exclusions:
        for col in existing_exclusions:
            print(f"  - {col}")
    else:
        print("  None")

    print("\nPotential leakage features found:")

    if existing_leakage:
        for col in existing_leakage:
            print(f"  - {col}")
    else:
        print("  None")

    # Raw timestamps should also be absent
    existing_raw_timestamps = [
        col for col in RAW_TIMESTAMP_FEATURES
        if col in train.columns
    ]

    if existing_raw_timestamps:
        print("\nRaw timestamp features still present:")
        for col in existing_raw_timestamps:
            print(f"  - {col}")
    else:
        print("\nRaw timestamp features: None")

    # --------------------------------------------------------
    # 5. BUILD FINAL EXCLUSION LIST
    # --------------------------------------------------------

    final_exclusions = []

    for col in EXCLUDE_FEATURES:
        if col in train.columns:
            final_exclusions.append(col)

    for col in LEAKAGE_FEATURES:
        if col in train.columns:
            final_exclusions.append(col)

    for col in RAW_TIMESTAMP_FEATURES:
        if col in train.columns:
            final_exclusions.append(col)

    # Remove duplicates while preserving order
    final_exclusions = list(dict.fromkeys(final_exclusions))

    print("\nFinal features to remove:")

    if final_exclusions:
        for col in final_exclusions:
            print(f"  - {col}")
    else:
        print("  None")

    # --------------------------------------------------------
    # 6. REMOVE EXCLUDED FEATURES
    # --------------------------------------------------------

    train_final = train.drop(
        columns=final_exclusions,
        errors="ignore"
    ).copy()

    test_final = test.drop(
        columns=final_exclusions,
        errors="ignore"
    ).copy()

    # --------------------------------------------------------
    # 7. TARGET / PREDICTOR SEPARATION CHECK
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("4. FINAL FEATURE SET")
    print("=" * 70)

    if TARGET not in train_final.columns:
        raise ValueError("Target disappeared from training dataset.")

    if TARGET not in test_final.columns:
        raise ValueError("Target disappeared from testing dataset.")

    X_train = train_final.drop(columns=[TARGET])
    y_train = train_final[TARGET]

    X_test = test_final.drop(columns=[TARGET])
    y_test = test_final[TARGET]

    print(f"\nFinal predictor count: {X_train.shape[1]}")
    print(f"Training rows         : {X_train.shape[0]}")
    print(f"Testing rows          : {X_test.shape[0]}")

    # --------------------------------------------------------
    # 8. TRAIN / TEST COLUMN CONSISTENCY
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("5. TRAIN / TEST FEATURE CONSISTENCY")
    print("=" * 70)

    train_features = list(X_train.columns)
    test_features = list(X_test.columns)

    train_only = sorted(
        set(train_features) - set(test_features)
    )

    test_only = sorted(
        set(test_features) - set(train_features)
    )

    if train_only:
        print("\nFeatures only in training:")
        for col in train_only:
            print(f"  - {col}")

    if test_only:
        print("\nFeatures only in testing:")
        for col in test_only:
            print(f"  - {col}")

    if train_only or test_only:
        raise ValueError(
            "Training and testing feature columns do not match."
        )

    print("\nTraining and testing features match.")
    print(f"Feature count: {len(train_features)}")

    # --------------------------------------------------------
    # 9. TARGET LEAKAGE CHECK
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("6. LEAKAGE VALIDATION")
    print("=" * 70)

    leakage_remaining = [
        col for col in X_train.columns
        if col in LEAKAGE_FEATURES
    ]

    if leakage_remaining:
        print("\nWARNING - potential leakage features remain:")

        for col in leakage_remaining:
            print(f"  - {col}")

        raise ValueError(
            "Potential leakage features remain in final predictors."
        )

    print("\nNo explicitly identified leakage features remain.")

    # --------------------------------------------------------
    # 10. ID CHECK
    # --------------------------------------------------------

    if "ID" in X_train.columns:
        raise ValueError(
            "ID is still present in the final predictor dataset."
        )

    print("ID excluded successfully.")

    # --------------------------------------------------------
    # 11. CONSTANT FEATURE CHECK
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("7. CONSTANT FEATURE CHECK")
    print("=" * 70)

    constant_features = []

    for col in X_train.columns:
        if X_train[col].nunique(dropna=False) <= 1:
            constant_features.append(col)

    if constant_features:
        print("\nConstant features found:")

        for col in constant_features:
            print(f"  - {col}")

        print("\nRemoving constant features...")

        X_train = X_train.drop(
            columns=constant_features
        )

        X_test = X_test.drop(
            columns=constant_features
        )

    else:
        print("\nNo constant predictor features found.")

    print(
        f"\nFinal predictor count after constant check: "
        f"{X_train.shape[1]}"
    )

    # --------------------------------------------------------
    # 12. MISSING / INFINITE VALIDATION
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("8. DATA QUALITY VALIDATION")
    print("=" * 70)

    train_check = pd.concat(
        [X_train, y_train],
        axis=1
    )

    test_check = pd.concat(
        [X_test, y_test],
        axis=1
    )

    train_missing, train_inf = check_missing_and_infinite(
        train_check,
        "Training"
    )

    test_missing, test_inf = check_missing_and_infinite(
        test_check,
        "Testing"
    )

    if train_missing > 0 or test_missing > 0:
        raise ValueError(
            "Missing values remain in final modeling datasets."
        )

    if train_inf > 0 or test_inf > 0:
        raise ValueError(
            "Infinite values remain in final modeling datasets."
        )

    print("\nNo missing or infinite values detected.")

    # --------------------------------------------------------
    # 13. DATA TYPE CHECK
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("9. DATA TYPE VALIDATION")
    print("=" * 70)

    non_numeric_train = X_train.select_dtypes(
        exclude=[np.number, "bool"]
    ).columns.tolist()

    non_numeric_test = X_test.select_dtypes(
        exclude=[np.number, "bool"]
    ).columns.tolist()

    print(
        f"\nNon-numeric training predictors: "
        f"{len(non_numeric_train)}"
    )

    if non_numeric_train:
        for col in non_numeric_train[:20]:
            print(f"  - {col}")

        if len(non_numeric_train) > 20:
            print(
                f"  ... and "
                f"{len(non_numeric_train) - 20} more"
            )

    print(
        f"\nNon-numeric testing predictors: "
        f"{len(non_numeric_test)}"
    )

    if non_numeric_test:
        for col in non_numeric_test[:20]:
            print(f"  - {col}")

    # --------------------------------------------------------
    # 14. FINAL COLUMN ORDER
    # --------------------------------------------------------

    # Force test columns into exactly the same order as train.
    X_test = X_test[X_train.columns]

    train_final = pd.concat(
        [X_train, y_train],
        axis=1
    )

    test_final = pd.concat(
        [X_test, y_test],
        axis=1
    )

    # --------------------------------------------------------
    # 15. FEATURE REPORT
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("10. FEATURE REPORT")
    print("=" * 70)

    report_rows = []

    for feature in X_train.columns:

        report_rows.append({
            "Feature": feature,
            "Category": feature_category(feature),
            "Data_Type": str(X_train[feature].dtype),
            "Train_Unique_Values": int(
                X_train[feature].nunique(dropna=False)
            ),
            "Test_Unique_Values": int(
                X_test[feature].nunique(dropna=False)
            ),
            "Train_Missing": int(
                X_train[feature].isna().sum()
            ),
            "Test_Missing": int(
                X_test[feature].isna().sum()
            ),
        })

    report = pd.DataFrame(report_rows)

    report.to_csv(
        REPORT_OUTPUT,
        index=False
    )

    print(f"\nSaved feature report:")
    print(f"  {REPORT_OUTPUT}")

    # --------------------------------------------------------
    # 16. FEATURE CATEGORY SUMMARY
    # --------------------------------------------------------

    print("\nFeature category counts:")

    print(
        report["Category"]
        .value_counts()
        .sort_index()
    )

    # --------------------------------------------------------
    # 17. SAVE FINAL DATASETS
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("11. SAVING FINAL MODELING DATASETS")
    print("=" * 70)

    train_final.to_csv(
        TRAIN_OUTPUT,
        index=False
    )

    test_final.to_csv(
        TEST_OUTPUT,
        index=False
    )

    print(f"\nTraining output:")
    print(f"  {TRAIN_OUTPUT}")

    print(f"\nTesting output:")
    print(f"  {TEST_OUTPUT}")

    # --------------------------------------------------------
    # 18. FINAL VALIDATION
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("12. FINAL VALIDATION")
    print("=" * 70)

    print(f"\nFinal training shape: {train_final.shape}")
    print(f"Final testing shape : {test_final.shape}")

    final_train_features = [
        col for col in train_final.columns
        if col != TARGET
    ]

    final_test_features = [
        col for col in test_final.columns
        if col != TARGET
    ]

    if final_train_features != final_test_features:
        raise ValueError(
            "Final training/testing feature order does not match."
        )

    print("\nTarget:", TARGET)
    print("Train/test feature order: MATCH")
    print("Missing values: 0")
    print("Infinite values: 0")
    print("Leakage features: EXCLUDED")
    print("ID: EXCLUDED")

    print("\n" + "=" * 70)
    print("FINAL FEATURE SELECTION COMPLETED")
    print("=" * 70)

    print("\nFiles created:")
    print(f"- {TRAIN_OUTPUT}")
    print(f"- {TEST_OUTPUT}")
    print(f"- {REPORT_OUTPUT}")

    print("\nThe datasets are ready for model training.")


if __name__ == "__main__":
    main()
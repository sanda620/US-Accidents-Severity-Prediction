import os
import numpy as np
import pandas as pd


# ============================================================
# Configuration
# ============================================================

TRAIN_INPUT = "data/train_categorical_encoded.csv"
TEST_INPUT = "data/test_categorical_encoded.csv"

TRAIN_OUTPUT = "data/train_numerical_features.csv"
TEST_OUTPUT = "data/test_numerical_features.csv"

REPORT_PATH = (
    "outputs/reports/numerical_transformation_results.csv"
)


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 70)
    print("NUMERICAL FEATURE TRANSFORMATION")
    print("=" * 70)

    os.makedirs("outputs/reports", exist_ok=True)

    # --------------------------------------------------------
    # 1. Load datasets
    # --------------------------------------------------------

    print("\nLoading training dataset...")
    train = pd.read_csv(TRAIN_INPUT)

    print("Loading test dataset...")
    test = pd.read_csv(TEST_INPUT)

    print(f"\nTraining shape before transformation: {train.shape}")
    print(f"Test shape before transformation:     {test.shape}")

    # --------------------------------------------------------
    # 2. Verify required columns
    # --------------------------------------------------------

    required_columns = [
        "Distance(mi)",
        "Precipitation(in)",
        "Start_Time",
        "End_Time",
    ]

    missing_train = [
        col for col in required_columns
        if col not in train.columns
    ]

    missing_test = [
        col for col in required_columns
        if col not in test.columns
    ]

    if missing_train:
        raise ValueError(
            f"Missing required training columns: {missing_train}"
        )

    if missing_test:
        raise ValueError(
            f"Missing required test columns: {missing_test}"
        )

    # --------------------------------------------------------
    # 3. Apply log1p transformations
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("APPLYING LOG1P TRANSFORMATIONS")
    print("=" * 70)

    # --------------------------------------------------------
    # Distance
    # --------------------------------------------------------

    print("\nTransforming Distance(mi)...")

    train_distance = pd.to_numeric(
        train["Distance(mi)"],
        errors="coerce"
    )

    test_distance = pd.to_numeric(
        test["Distance(mi)"],
        errors="coerce"
    )

    if (train_distance < 0).any():
        raise ValueError(
            "Negative values detected in training Distance(mi)."
        )

    if (test_distance < 0).any():
        raise ValueError(
            "Negative values detected in test Distance(mi)."
        )

    train["Distance_log1p"] = np.log1p(
        train_distance
    )

    test["Distance_log1p"] = np.log1p(
        test_distance
    )

    # --------------------------------------------------------
    # Precipitation
    # --------------------------------------------------------

    print("Transforming Precipitation(in)...")

    train_precipitation = pd.to_numeric(
        train["Precipitation(in)"],
        errors="coerce"
    )

    test_precipitation = pd.to_numeric(
        test["Precipitation(in)"],
        errors="coerce"
    )

    if (train_precipitation < 0).any():
        raise ValueError(
            "Negative values detected in training "
            "Precipitation(in)."
        )

    if (test_precipitation < 0).any():
        raise ValueError(
            "Negative values detected in test "
            "Precipitation(in)."
        )

    train["Precipitation_log1p"] = np.log1p(
        train_precipitation
    )

    test["Precipitation_log1p"] = np.log1p(
        test_precipitation
    )

    # --------------------------------------------------------
    # 4. Remove raw timestamps
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("REMOVING RAW TIMESTAMP FEATURES")
    print("=" * 70)

    print("\nRemoving:")
    print("  - Start_Time")
    print("  - End_Time")

    train = train.drop(
        columns=["Start_Time", "End_Time"]
    )

    test = test.drop(
        columns=["Start_Time", "End_Time"]
    )

    # --------------------------------------------------------
    # 5. Validate transformed values
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("VALIDATING TRANSFORMED FEATURES")
    print("=" * 70)

    transformed_features = [
        "Distance_log1p",
        "Precipitation_log1p",
    ]

    validation_rows = []

    for feature in transformed_features:

        train_values = pd.to_numeric(
            train[feature],
            errors="coerce"
        )

        test_values = pd.to_numeric(
            test[feature],
            errors="coerce"
        )

        train_nan = int(train_values.isna().sum())
        test_nan = int(test_values.isna().sum())

        train_inf = int(
            np.isinf(train_values).sum()
        )

        test_inf = int(
            np.isinf(test_values).sum()
        )

        validation_rows.append({
            "Feature": feature,
            "Train_Missing": train_nan,
            "Test_Missing": test_nan,
            "Train_Infinite": train_inf,
            "Test_Infinite": test_inf,
            "Train_Min": train_values.min(),
            "Train_Max": train_values.max(),
            "Test_Min": test_values.min(),
            "Test_Max": test_values.max(),
        })

        print(f"\n{feature}")
        print(f"  Train missing:   {train_nan}")
        print(f"  Test missing:    {test_nan}")
        print(f"  Train infinite:  {train_inf}")
        print(f"  Test infinite:   {test_inf}")

    validation_df = pd.DataFrame(
        validation_rows
    )

    # --------------------------------------------------------
    # 6. Check entire datasets for missing values
    # --------------------------------------------------------

    print("\nChecking complete datasets...")

    train_missing_total = int(
        train.isna().sum().sum()
    )

    test_missing_total = int(
        test.isna().sum().sum()
    )

    train_inf_total = int(
        np.isinf(
            train.select_dtypes(
                include=[np.number]
            )
        ).sum().sum()
    )

    test_inf_total = int(
        np.isinf(
            test.select_dtypes(
                include=[np.number]
            )
        ).sum().sum()
    )

    print(
        f"\nTraining missing values: {train_missing_total}"
    )

    print(
        f"Test missing values:     {test_missing_total}"
    )

    print(
        f"Training infinite values: {train_inf_total}"
    )

    print(
        f"Test infinite values:     {test_inf_total}"
    )

    if train_missing_total > 0:
        raise ValueError(
            "Training dataset contains missing values."
        )

    if test_missing_total > 0:
        raise ValueError(
            "Test dataset contains missing values."
        )

    if train_inf_total > 0:
        raise ValueError(
            "Training dataset contains infinite values."
        )

    if test_inf_total > 0:
        raise ValueError(
            "Test dataset contains infinite values."
        )

    # --------------------------------------------------------
    # 7. Verify train/test columns match
    # --------------------------------------------------------

    print("\nChecking train/test column consistency...")

    if list(train.columns) != list(test.columns):

        train_only = sorted(
            set(train.columns) - set(test.columns)
        )

        test_only = sorted(
            set(test.columns) - set(train.columns)
        )

        raise ValueError(
            "Train/test columns do not match.\n"
            f"Train only: {train_only}\n"
            f"Test only: {test_only}"
        )

    print("Train/test columns match.")

    # --------------------------------------------------------
    # 8. Save datasets
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("SAVING TRANSFORMED DATASETS")
    print("=" * 70)

    train.to_csv(
        TRAIN_OUTPUT,
        index=False
    )

    test.to_csv(
        TEST_OUTPUT,
        index=False
    )

    validation_df.to_csv(
        REPORT_PATH,
        index=False
    )

    print(f"\nTraining output:")
    print(f"  {TRAIN_OUTPUT}")

    print(f"\nTest output:")
    print(f"  {TEST_OUTPUT}")

    print(f"\nTransformation report:")
    print(f"  {REPORT_PATH}")

    # --------------------------------------------------------
    # 9. Final information
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("FINAL DATASET INFORMATION")
    print("=" * 70)

    print(
        f"\nTraining shape after transformation: "
        f"{train.shape}"
    )

    print(
        f"Test shape after transformation:     "
        f"{test.shape}"
    )

    print(
        "\nNew features created:"
    )

    print("  - Distance_log1p")
    print("  - Precipitation_log1p")

    print(
        "\nColumns removed:"
    )

    print("  - Start_Time")
    print("  - End_Time")

    print(
        "\nColumns intentionally retained:"
    )

    print("  - Distance(mi)")
    print("  - Precipitation(in)")
    print("  - Pressure(in)")
    print("  - Visibility(mi)")

    print(
        "\nOriginal numerical columns are retained "
        "so their usefulness can be compared during "
        "model development."
    )

    print("\n" + "=" * 70)
    print("NUMERICAL FEATURE TRANSFORMATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
import os
import pickle
import numpy as np
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

TRAIN_FILE = "../data/train.csv"
FINAL_TRAIN_FILE = "../data/train_final.csv"

MODEL_DIR = "../models"
ARTIFACT_FILE = os.path.join(
    MODEL_DIR,
    "preprocessing_artifacts.pkl"
)


# ============================================================
# FEATURE DEFINITIONS
# ============================================================

LEAKAGE_FEATURES = [
    "End_Lat",
    "End_Lng"
]

UNRESOLVED_FEATURES = [
    "Weather_Timestamp",
    "Description"
]

NUMERIC_INDICATOR_FEATURES = [
    "Precipitation(in)",
    "Wind_Chill(F)",
    "Wind_Speed(mph)"
]

NUMERIC_MEDIAN_FEATURES = [
    "Precipitation(in)",
    "Wind_Chill(F)",
    "Wind_Speed(mph)",
    "Visibility(mi)",
    "Humidity(%)",
    "Temperature(F)",
    "Pressure(in)"
]

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
    "Astronomical_Twilight"
]

FREQUENCY_FEATURES = [
    "Street",
    "City",
    "County",
    "Zipcode",
    "Airport_Code",
    "Weather_Condition"
]

ONE_HOT_FEATURES = [
    "Source",
    "State",
    "Timezone",
    "Wind_Direction",
    "Sunrise_Sunset",
    "Civil_Twilight",
    "Nautical_Twilight",
    "Astronomical_Twilight"
]

EXCLUDE_FEATURES = [
    "Country"
]

RAW_TIMESTAMP_FEATURES = [
    "Start_Time",
    "End_Time"
]


# ============================================================
# INVALID VALUE RULES
# ============================================================

INVALID_RULES = {
    "Temperature(F)": (-80, 140),
    "Wind_Speed(mph)": (0, 100),
    "Pressure(in)": (20, 35),
    "Visibility(mi)": (0, 100),
    "Humidity(%)": (0, 100),
    "Precipitation(in)": (0, None),
    "Distance(mi)": (0, 200)
}


# ============================================================
# CATEGORY NORMALIZATION
# ============================================================

def normalize_category(series):
    return (
        series
        .astype("string")
        .fillna("__MISSING__")
    )


# ============================================================
# BUILD ARTIFACTS
# ============================================================

def main():

    print("Loading training dataset...")

    train = pd.read_csv(TRAIN_FILE)

    print(f"Training shape: {train.shape}")

    print("\nLoading final feature dataset...")

    final_train = pd.read_csv(FINAL_TRAIN_FILE)

    print(f"Final training shape: {final_train.shape}")


    # --------------------------------------------------------
    # 1. Training medians
    # --------------------------------------------------------

    training_medians = {}

    for feature in NUMERIC_MEDIAN_FEATURES:

        if feature in train.columns:

            value = pd.to_numeric(
                train[feature],
                errors="coerce"
            ).median()

            training_medians[feature] = float(value)

    print("\nTraining medians:")

    for feature, value in training_medians.items():
        print(f"{feature}: {value}")


    # --------------------------------------------------------
    # 2. Frequency encoding mappings
    # --------------------------------------------------------

    frequency_mappings = {}

    print("\nBuilding frequency mappings...")

    for feature in FREQUENCY_FEATURES:

        if feature not in train.columns:
            continue

        normalized = normalize_category(
            train[feature]
        )

        mapping = (
            normalized
            .value_counts(normalize=True)
            .to_dict()
        )

        frequency_mappings[feature] = mapping

        print(
            f"{feature}: "
            f"{len(mapping):,} categories"
        )


    # --------------------------------------------------------
    # 3. One-hot category columns
    # --------------------------------------------------------

    one_hot_columns = {}

    print("\nBuilding one-hot category definitions...")

    for feature in ONE_HOT_FEATURES:

        if feature not in train.columns:
            continue

        normalized = normalize_category(
            train[feature]
        )

        dummy = pd.get_dummies(
            normalized,
            prefix=feature,
            dtype=np.int8
        )

        one_hot_columns[feature] = list(
            dummy.columns
        )

        print(
            f"{feature}: "
            f"{len(dummy.columns)} columns"
        )


    # --------------------------------------------------------
    # 4. Final model feature columns
    # --------------------------------------------------------

    final_feature_columns = [
        column
        for column in final_train.columns
        if column != "Severity"
    ]

    print(
        "\nFinal model feature count:",
        len(final_feature_columns)
    )


    # --------------------------------------------------------
    # 5. Save everything
    # --------------------------------------------------------

    artifacts = {

        "training_medians":
            training_medians,

        "frequency_mappings":
            frequency_mappings,

        "one_hot_columns":
            one_hot_columns,

        "final_feature_columns":
            final_feature_columns,

        "invalid_rules":
            INVALID_RULES,

        "numeric_indicator_features":
            NUMERIC_INDICATOR_FEATURES,

        "numeric_median_features":
            NUMERIC_MEDIAN_FEATURES,

        "frequency_features":
            FREQUENCY_FEATURES,

        "one_hot_features":
            ONE_HOT_FEATURES,

        "exclude_features":
            EXCLUDE_FEATURES,

        "leakage_features":
            LEAKAGE_FEATURES,

        "unresolved_features":
            UNRESOLVED_FEATURES
    }


    os.makedirs(
        MODEL_DIR,
        exist_ok=True
    )


    with open(
        ARTIFACT_FILE,
        "wb"
    ) as file:

        pickle.dump(
            artifacts,
            file
        )


    print("\n====================================")
    print("Preprocessing artifacts saved.")
    print("====================================")
    print(
        f"File: {ARTIFACT_FILE}"
    )
    print(
        f"Final features: "
        f"{len(final_feature_columns)}"
    )


if __name__ == "__main__":
    main()
import os
import numpy as np
import pandas as pd


# ============================================================
# Configuration
# ============================================================

TRAIN_FILE = "data/train_temporal_features.csv"
TEST_FILE = "data/test_temporal_features.csv"

TRAIN_OUTPUT = "data/train_categorical_encoded.csv"
TEST_OUTPUT = "data/test_categorical_encoded.csv"

REPORT_OUTPUT = "outputs/reports/categorical_encoding_results.csv"

TARGET = "Severity"

# Categorical encoding decisions from Stage 4.7.1
ONE_HOT_FEATURES = [
    "Source",
    "State",
    "Timezone",
    "Wind_Direction",
    "Sunrise_Sunset",
    "Civil_Twilight",
    "Nautical_Twilight",
    "Astronomical_Twilight",
]

FREQUENCY_FEATURES = [
    "Street",
    "City",
    "County",
    "Zipcode",
    "Airport_Code",
    "Weather_Condition",
]

EXCLUDE_FEATURES = [
    "Country",
]


# ============================================================
# Utility functions
# ============================================================

def normalize_category(series):
    """
    Convert categorical values to strings while preserving
    missing values as an explicit category.
    """
    return series.astype("string").fillna("__MISSING__")


def build_frequency_mapping(series):
    """
    Create frequency mapping using TRAINING DATA ONLY.
    """
    normalized = normalize_category(series)

    frequencies = normalized.value_counts(normalize=True)

    return frequencies.to_dict()


def apply_frequency_encoding(series, mapping):
    """
    Apply training-derived frequency mapping.

    Unseen test categories receive frequency 0.
    """
    normalized = normalize_category(series)

    return normalized.map(mapping).fillna(0.0)


# ============================================================
# Main
# ============================================================

print("=" * 70)
print("STAGE 4.7.2 - CATEGORICAL FEATURE ENCODING")
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
# Preserve target
# ============================================================

if TARGET not in train.columns:
    raise ValueError(f"Target column '{TARGET}' not found in training data.")

if TARGET not in test.columns:
    raise ValueError(f"Target column '{TARGET}' not found in testing data.")


# ============================================================
# Validate feature availability
# ============================================================

all_categorical_features = (
    ONE_HOT_FEATURES
    + FREQUENCY_FEATURES
    + EXCLUDE_FEATURES
)

missing_train_features = [
    feature
    for feature in all_categorical_features
    if feature not in train.columns
]

missing_test_features = [
    feature
    for feature in all_categorical_features
    if feature not in test.columns
]

if missing_train_features:
    raise ValueError(
        f"Missing training features: {missing_train_features}"
    )

if missing_test_features:
    raise ValueError(
        f"Missing testing features: {missing_test_features}"
    )


# ============================================================
# Encoding report
# ============================================================

encoding_results = []


# ============================================================
# Step 1: Frequency encoding
# ============================================================

print("\n" + "=" * 70)
print("1. FREQUENCY ENCODING")
print("=" * 70)

for feature in FREQUENCY_FEATURES:

    print(f"\nProcessing: {feature}")

    # Build mapping ONLY from training data
    mapping = build_frequency_mapping(train[feature])

    # Apply mapping
    train_encoded = apply_frequency_encoding(
        train[feature],
        mapping
    )

    test_encoded = apply_frequency_encoding(
        test[feature],
        mapping
    )

    # Replace original column
    train[feature] = train_encoded
    test[feature] = test_encoded

    unseen_count = (
        normalize_category(test[feature])
        .isin([])
    )

    # Count original test categories not present in training
    train_categories = set(
        normalize_category(
            pd.read_csv(
                TRAIN_FILE,
                usecols=[feature]
            )[feature]
        )
    )

    test_categories = set(
        normalize_category(
            pd.read_csv(
                TEST_FILE,
                usecols=[feature]
            )[feature]
        )
    )

    unseen_categories = test_categories - train_categories

    print(
        f"Training categories : {len(train_categories):,}"
    )

    print(
        f"Testing categories  : {len(test_categories):,}"
    )

    print(
        f"Unseen test categories: {len(unseen_categories):,}"
    )

    print(
        "Encoding: frequency"
    )

    encoding_results.append({
        "Feature": feature,
        "Encoding": "FREQUENCY",
        "Train_Unique_Values": len(train_categories),
        "Test_Unique_Values": len(test_categories),
        "Unseen_Test_Categories": len(unseen_categories),
        "Output_Type": "float64",
    })


# ============================================================
# Step 2: One-hot encoding
# ============================================================

print("\n" + "=" * 70)
print("2. ONE-HOT ENCODING")
print("=" * 70)

for feature in ONE_HOT_FEATURES:

    print(f"\nProcessing: {feature}")

    train[feature] = normalize_category(train[feature])
    test[feature] = normalize_category(test[feature])

    train_categories = set(train[feature].unique())
    test_categories = set(test[feature].unique())

    unseen_categories = test_categories - train_categories

    # --------------------------------------------------------
    # IMPORTANT:
    # Create categories from TRAINING DATA only.
    # --------------------------------------------------------

    train_dummies = pd.get_dummies(
        train[feature],
        prefix=feature,
        dtype=np.int8
    )

    test_dummies = pd.get_dummies(
        test[feature],
        prefix=feature,
        dtype=np.int8
    )

    # Align test columns to training columns
    test_dummies = test_dummies.reindex(
        columns=train_dummies.columns,
        fill_value=0
    )

    print(
        f"Training categories : {len(train_categories):,}"
    )

    print(
        f"Testing categories  : {len(test_categories):,}"
    )

    print(
        f"Unseen test categories: {len(unseen_categories):,}"
    )

    print(
        f"Created dummy columns: {train_dummies.shape[1]:,}"
    )

    # Remove original categorical feature
    train.drop(columns=[feature], inplace=True)
    test.drop(columns=[feature], inplace=True)

    # Add encoded columns
    train = pd.concat(
        [train, train_dummies],
        axis=1
    )

    test = pd.concat(
        [test, test_dummies],
        axis=1
    )

    encoding_results.append({
        "Feature": feature,
        "Encoding": "ONE_HOT",
        "Train_Unique_Values": len(train_categories),
        "Test_Unique_Values": len(test_categories),
        "Unseen_Test_Categories": len(unseen_categories),
        "Output_Type": "int8",
    })


# ============================================================
# Step 3: Exclude constant categorical features
# ============================================================

print("\n" + "=" * 70)
print("3. EXCLUDING CONSTANT FEATURES")
print("=" * 70)

for feature in EXCLUDE_FEATURES:

    if feature in train.columns:
        train.drop(columns=[feature], inplace=True)

    if feature in test.columns:
        test.drop(columns=[feature], inplace=True)

    print(f"Excluded: {feature}")

    encoding_results.append({
        "Feature": feature,
        "Encoding": "EXCLUDE",
        "Train_Unique_Values": 1,
        "Test_Unique_Values": 1,
        "Unseen_Test_Categories": 0,
        "Output_Type": "excluded",
    })


# ============================================================
# Step 4: Ensure train/test columns match
# ============================================================

print("\n" + "=" * 70)
print("4. TRAIN / TEST COLUMN VALIDATION")
print("=" * 70)

train_columns = set(train.columns)
test_columns = set(test.columns)

missing_in_test = train_columns - test_columns
missing_in_train = test_columns - train_columns

if missing_in_test:
    raise ValueError(
        f"Columns missing from testing data: {missing_in_test}"
    )

if missing_in_train:
    raise ValueError(
        f"Columns missing from training data: {missing_in_train}"
    )

# Ensure exact same order
test = test[train.columns]

print("Training and testing columns match.")

print(
    f"Training columns: {len(train.columns):,}"
)

print(
    f"Testing columns : {len(test.columns):,}"
)


# ============================================================
# Step 5: Check missing values
# ============================================================

print("\n" + "=" * 70)
print("5. MISSING VALUE VALIDATION")
print("=" * 70)

train_missing = int(train.isna().sum().sum())
test_missing = int(test.isna().sum().sum())

print(
    f"Training remaining missing values: {train_missing:,}"
)

print(
    f"Testing remaining missing values : {test_missing:,}"
)

if train_missing > 0 or test_missing > 0:
    print(
        "\nWARNING: Missing values remain."
    )


# ============================================================
# Step 6: Save datasets
# ============================================================

print("\n" + "=" * 70)
print("6. SAVING ENCODED DATASETS")
print("=" * 70)

os.makedirs(
    os.path.dirname(TRAIN_OUTPUT),
    exist_ok=True
)

os.makedirs(
    os.path.dirname(REPORT_OUTPUT),
    exist_ok=True
)

train.to_csv(
    TRAIN_OUTPUT,
    index=False
)

test.to_csv(
    TEST_OUTPUT,
    index=False
)

encoding_report = pd.DataFrame(
    encoding_results
)

encoding_report.to_csv(
    REPORT_OUTPUT,
    index=False
)


# ============================================================
# Final summary
# ============================================================

print("\n" + "=" * 70)
print("CATEGORICAL FEATURE ENCODING COMPLETED")
print("=" * 70)

print(
    f"\nOriginal training shape : "
    f"(799673, 56)"
)

print(
    f"Encoded training shape  : "
    f"{train.shape}"
)

print(
    f"Original testing shape  : "
    f"(199919, 56)"
)

print(
    f"Encoded testing shape   : "
    f"{test.shape}"
)

print("\nEncoding summary:")
print(
    encoding_report[
        [
            "Feature",
            "Encoding",
            "Train_Unique_Values",
            "Test_Unique_Values",
            "Unseen_Test_Categories",
        ]
    ].to_string(index=False)
)

print("\nFiles created:")
print(f"- {TRAIN_OUTPUT}")
print(f"- {TEST_OUTPUT}")
print(f"- {REPORT_OUTPUT}")

print("=" * 70)
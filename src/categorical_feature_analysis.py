import os
import pandas as pd


# ============================================================
# Configuration
# ============================================================

TRAIN_FILE = "data/train_temporal_features.csv"
TEST_FILE = "data/test_temporal_features.csv"

OUTPUT_FILE = "outputs/reports/categorical_feature_analysis.csv"


# ============================================================
# Setup
# ============================================================

os.makedirs("outputs/reports", exist_ok=True)

print("=" * 70)
print("STAGE 4.7.1 - CATEGORICAL FEATURE ANALYSIS")
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
# Candidate categorical features
# ============================================================

categorical_features = [
    "Source",
    "Street",
    "City",
    "County",
    "State",
    "Zipcode",
    "Country",
    "Timezone",
    "Airport_Code",
    "Wind_Direction",
    "Weather_Condition",
    "Sunrise_Sunset",
    "Civil_Twilight",
    "Nautical_Twilight",
    "Astronomical_Twilight"
]


# ============================================================
# Validate columns
# ============================================================

missing_columns = [
    col for col in categorical_features
    if col not in train.columns
]

if missing_columns:
    raise ValueError(
        f"Categorical columns missing from training dataset: "
        f"{missing_columns}"
    )


# ============================================================
# Determine encoding strategy
# ============================================================

def determine_encoding(unique_count, feature):
    """
    Select an encoding strategy based on cardinality
    and the role of the feature.
    """

    # Constant features
    if unique_count <= 1:
        return (
            "EXCLUDE",
            "Constant feature with no predictive variation."
        )

    # Explicit exclusions
    if feature == "Country":
        return (
            "EXCLUDE",
            "Constant geographic feature; no useful variation."
        )

    # Low-cardinality features
    if unique_count <= 10:
        return (
            "ONE_HOT",
            "Low-cardinality categorical feature suitable "
            "for one-hot encoding."
        )

    # Moderate-cardinality features
    if unique_count <= 50:
        return (
            "ONE_HOT",
            "Moderate/low cardinality; one-hot encoding is "
            "feasible for the modeling pipeline."
        )

    # High-cardinality geographic/text-like identifiers
    if feature in [
        "Street",
        "City",
        "Zipcode",
        "Airport_Code"
    ]:
        return (
            "FREQUENCY",
            "High-cardinality feature; frequency encoding "
            "avoids a very large one-hot feature matrix."
        )

    # Other high-cardinality categorical variables
    return (
        "FREQUENCY",
        "High-cardinality categorical feature; frequency "
        "encoding reduces dimensionality."
    )


# ============================================================
# Analyze categorical features
# ============================================================

print("\n" + "=" * 70)
print("CATEGORICAL FEATURE ANALYSIS")
print("=" * 70)

report_rows = []

for feature in categorical_features:

    train_series = train[feature].astype("string")
    test_series = test[feature].astype("string")

    train_unique = int(train_series.nunique(dropna=False))
    test_unique = int(test_series.nunique(dropna=False))

    train_missing = int(train[feature].isna().sum())
    test_missing = int(test[feature].isna().sum())

    # Category overlap
    train_categories = set(train_series.dropna().unique())
    test_categories = set(test_series.dropna().unique())

    unseen_test_categories = test_categories - train_categories

    unseen_count = len(unseen_test_categories)

    if train_unique > 0:
        unseen_percentage = (
            unseen_count / train_unique
        ) * 100
    else:
        unseen_percentage = 0.0

    encoding, reason = determine_encoding(
        train_unique,
        feature
    )

    # --------------------------------------------------------
    # Print feature information
    # --------------------------------------------------------

    print("\n" + "-" * 70)
    print(f"Feature: {feature}")
    print(f"Training unique values : {train_unique:,}")
    print(f"Testing unique values  : {test_unique:,}")
    print(f"Training missing       : {train_missing:,}")
    print(f"Testing missing        : {test_missing:,}")
    print(
        f"Unseen test categories: "
        f"{unseen_count:,}"
    )
    print(
        f"Encoding strategy     : "
        f"{encoding}"
    )
    print(f"Reason                : {reason}")

    # --------------------------------------------------------
    # Save analysis
    # --------------------------------------------------------

    report_rows.append({
        "Feature": feature,
        "Data_Type": str(train[feature].dtype),
        "Train_Unique_Values": train_unique,
        "Test_Unique_Values": test_unique,
        "Train_Missing_Count": train_missing,
        "Test_Missing_Count": test_missing,
        "Unseen_Test_Categories": unseen_count,
        "Unseen_Test_Percentage": round(
            unseen_percentage,
            4
        ),
        "Encoding_Strategy": encoding,
        "Reason": reason
    })


# ============================================================
# Create report
# ============================================================

report = pd.DataFrame(report_rows)

report.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# Encoding summary
# ============================================================

print("\n" + "=" * 70)
print("ENCODING STRATEGY SUMMARY")
print("=" * 70)

print(
    report[
        [
            "Feature",
            "Train_Unique_Values",
            "Test_Unique_Values",
            "Unseen_Test_Categories",
            "Encoding_Strategy"
        ]
    ].to_string(index=False)
)


print("\n" + "=" * 70)
print("ENCODING COUNTS")
print("=" * 70)

print(
    report["Encoding_Strategy"]
    .value_counts()
)


# ============================================================
# Save detailed unseen-category report
# ============================================================

unseen_rows = []

for feature in categorical_features:

    train_categories = set(
        train[feature]
        .astype("string")
        .dropna()
        .unique()
    )

    test_categories = set(
        test[feature]
        .astype("string")
        .dropna()
        .unique()
    )

    unseen_categories = sorted(
        test_categories - train_categories
    )

    for category in unseen_categories:
        unseen_rows.append({
            "Feature": feature,
            "Unseen_Test_Category": category
        })


unseen_report = pd.DataFrame(
    unseen_rows
)

unseen_output = (
    "outputs/reports/"
    "categorical_unseen_test_categories.csv"
)

unseen_report.to_csv(
    unseen_output,
    index=False
)


# ============================================================
# Final summary
# ============================================================

print("\n" + "=" * 70)
print("CATEGORICAL FEATURE ANALYSIS COMPLETED")
print("=" * 70)

print(f"\nMain report:")
print(f"- {OUTPUT_FILE}")

print("\nUnseen-category report:")
print(f"- {unseen_output}")

print("\nImportant:")
print(
    "No categorical transformations were applied yet. "
    "The encoding strategies will be applied in the "
    "next preprocessing step using training data only."
)

print("=" * 70)
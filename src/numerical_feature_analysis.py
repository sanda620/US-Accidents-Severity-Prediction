import os
import numpy as np
import pandas as pd


# ============================================================
# Configuration
# ============================================================

TRAIN_PATH = "data/train_categorical_encoded.csv"
TEST_PATH = "data/test_categorical_encoded.csv"

OUTPUT_DIR = "outputs/reports"

ANALYSIS_REPORT = os.path.join(
    OUTPUT_DIR,
    "numerical_feature_analysis.csv"
)

CORRELATION_REPORT = os.path.join(
    OUTPUT_DIR,
    "numerical_correlation_with_severity.csv"
)

SUMMARY_REPORT = os.path.join(
    OUTPUT_DIR,
    "numerical_feature_summary.csv"
)


# ============================================================
# Feature groups
# ============================================================

# True continuous numerical variables
CONTINUOUS_FEATURES = [
    "Start_Lat",
    "Start_Lng",
    "Distance(mi)",
    "Temperature(F)",
    "Wind_Chill(F)",
    "Humidity(%)",
    "Pressure(in)",
    "Visibility(mi)",
    "Wind_Speed(mph)",
    "Precipitation(in)",
]

# Numerical temporal features
TEMPORAL_NUMERICAL_FEATURES = [
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
    "Month_Cos",
]

# Missing-value indicators
MISSING_INDICATORS = [
    "Precipitation_in_Missing",
    "Wind_Chill_F_Missing",
    "Wind_Speed_mph_Missing",
]

# Frequency-encoded categorical variables.
# These are numeric representations of categories,
# but they are NOT continuous measurements.
FREQUENCY_ENCODED_FEATURES = [
    "Street",
    "City",
    "County",
    "Zipcode",
    "Airport_Code",
    "Weather_Condition",
]

# Binary one-hot encoded features.
# These should not be treated as continuous numerical
# measurements for skewness/IQR analysis.
ONE_HOT_PREFIXES = [
    "Source_",
    "State_",
    "Timezone_",
    "Wind_Direction_",
    "Sunrise_Sunset_",
    "Civil_Twilight_",
    "Nautical_Twilight_",
    "Astronomical_Twilight_",
]


# ============================================================
# Helper functions
# ============================================================

def iqr_outlier_percentage(series):
    """Calculate IQR-based outlier percentage."""

    series = pd.to_numeric(
        series,
        errors="coerce"
    ).dropna()

    if len(series) == 0:
        return 0.0

    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)

    iqr = q3 - q1

    if iqr == 0:
        return 0.0

    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    outliers = (
        (series < lower_bound) |
        (series > upper_bound)
    )

    return outliers.mean() * 100


def calculate_skewness(series):
    """Calculate skewness safely."""

    series = pd.to_numeric(
        series,
        errors="coerce"
    ).dropna()

    if len(series) < 3:
        return np.nan

    return series.skew()


# ============================================================
# Main
# ============================================================

def main():

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 70)
    print("NUMERICAL FEATURE ANALYSIS")
    print("=" * 70)

    # --------------------------------------------------------
    # 1. Load data
    # --------------------------------------------------------

    print("\nLoading training dataset...")
    train = pd.read_csv(TRAIN_PATH)

    print("Loading test dataset...")
    test = pd.read_csv(TEST_PATH)

    print(f"\nTraining shape: {train.shape}")
    print(f"Test shape:     {test.shape}")

    # --------------------------------------------------------
    # 2. Verify expected features
    # --------------------------------------------------------

    continuous_features = [
        col for col in CONTINUOUS_FEATURES
        if col in train.columns
    ]

    temporal_features = [
        col for col in TEMPORAL_NUMERICAL_FEATURES
        if col in train.columns
    ]

    missing_indicator_features = [
        col for col in MISSING_INDICATORS
        if col in train.columns
    ]

    frequency_features = [
        col for col in FREQUENCY_ENCODED_FEATURES
        if col in train.columns
    ]

    print("\n" + "=" * 70)
    print("FEATURE GROUPS")
    print("=" * 70)

    print(
        f"\nContinuous numerical features: "
        f"{len(continuous_features)}"
    )

    for col in continuous_features:
        print(f"  - {col}")

    print(
        f"\nTemporal numerical features: "
        f"{len(temporal_features)}"
    )

    for col in temporal_features:
        print(f"  - {col}")

    print(
        f"\nMissing-value indicators: "
        f"{len(missing_indicator_features)}"
    )

    for col in missing_indicator_features:
        print(f"  - {col}")

    print(
        f"\nFrequency-encoded categorical features: "
        f"{len(frequency_features)}"
    )

    for col in frequency_features:
        print(f"  - {col}")

    # --------------------------------------------------------
    # 3. Continuous numerical analysis
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("CONTINUOUS NUMERICAL FEATURE ANALYSIS")
    print("=" * 70)

    analysis_rows = []

    for column in continuous_features:

        train_series = pd.to_numeric(
            train[column],
            errors="coerce"
        )

        test_series = pd.to_numeric(
            test[column],
            errors="coerce"
        )

        train_non_missing = train_series.dropna()

        if len(train_non_missing) == 0:
            continue

        q1 = train_non_missing.quantile(0.25)
        q3 = train_non_missing.quantile(0.75)

        analysis_rows.append({
            "Feature": column,
            "Feature_Group": "Continuous",
            "Train_Missing_Count": int(
                train_series.isna().sum()
            ),
            "Train_Missing_Percentage": (
                train_series.isna().mean() * 100
            ),
            "Test_Missing_Count": int(
                test_series.isna().sum()
            ),
            "Test_Missing_Percentage": (
                test_series.isna().mean() * 100
            ),
            "Mean": train_non_missing.mean(),
            "Median": train_non_missing.median(),
            "Std": train_non_missing.std(),
            "Min": train_non_missing.min(),
            "Q1": q1,
            "Q3": q3,
            "Max": train_non_missing.max(),
            "IQR": q3 - q1,
            "Skewness": train_non_missing.skew(),
            "IQR_Outlier_Percentage": (
                iqr_outlier_percentage(train_series)
            ),
            "Unique_Values": int(
                train_series.nunique()
            )
        })

    analysis_df = pd.DataFrame(analysis_rows)

    if len(analysis_df) > 0:

        analysis_df["Absolute_Skewness"] = (
            analysis_df["Skewness"].abs()
        )

        analysis_df = analysis_df.sort_values(
            "Absolute_Skewness",
            ascending=False
        )

    analysis_df.to_csv(
        ANALYSIS_REPORT,
        index=False
    )

    print(
        f"\nSaved continuous feature analysis:"
        f"\n  {ANALYSIS_REPORT}"
    )

    # --------------------------------------------------------
    # 4. Skewness
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("SKEWNESS ANALYSIS")
    print("=" * 70)

    highly_skewed = analysis_df[
        analysis_df["Absolute_Skewness"] > 1
    ]

    if len(highly_skewed) == 0:

        print("\nNo continuous features have |skewness| > 1.")

    else:

        print(
            "\nContinuous features with "
            "|skewness| > 1:"
        )

        for _, row in highly_skewed.iterrows():

            print(
                f"  {row['Feature']}: "
                f"{row['Skewness']:.4f}"
            )

    # --------------------------------------------------------
    # 5. IQR outliers
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("IQR OUTLIER ANALYSIS")
    print("=" * 70)

    outlier_features = analysis_df[
        analysis_df["IQR_Outlier_Percentage"] > 1
    ].sort_values(
        "IQR_Outlier_Percentage",
        ascending=False
    )

    if len(outlier_features) == 0:

        print(
            "\nNo continuous features have "
            "more than 1% IQR outliers."
        )

    else:

        print(
            "\nContinuous features with "
            "more than 1% IQR outliers:"
        )

        for _, row in outlier_features.iterrows():

            print(
                f"  {row['Feature']}: "
                f"{row['IQR_Outlier_Percentage']:.4f}%"
            )

    # --------------------------------------------------------
    # 6. Correlation with Severity
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("CORRELATION WITH SEVERITY")
    print("=" * 70)

    correlation_rows = []

    severity = pd.to_numeric(
        train["Severity"],
        errors="coerce"
    )

    # Only continuous numerical variables are included
    # in this correlation analysis.
    for column in continuous_features:

        feature = pd.to_numeric(
            train[column],
            errors="coerce"
        )

        valid = pd.DataFrame({
            "feature": feature,
            "severity": severity
        }).dropna()

        if len(valid) == 0:
            continue

        pearson = valid["feature"].corr(
            valid["severity"],
            method="pearson"
        )

        spearman = valid["feature"].corr(
            valid["severity"],
            method="spearman"
        )

        correlation_rows.append({
            "Feature": column,
            "Pearson_Correlation": pearson,
            "Spearman_Correlation": spearman,
            "Absolute_Pearson": abs(pearson),
            "Absolute_Spearman": abs(spearman)
        })

    correlation_df = pd.DataFrame(
        correlation_rows
    )

    correlation_df = correlation_df.sort_values(
        "Absolute_Spearman",
        ascending=False
    )

    correlation_df.to_csv(
        CORRELATION_REPORT,
        index=False
    )

    print(
        f"\nSaved correlation analysis:"
        f"\n  {CORRELATION_REPORT}"
    )

    # --------------------------------------------------------
    # 7. Print correlations
    # --------------------------------------------------------

    print(
        "\nContinuous features ranked by "
        "absolute Spearman correlation:"
    )

    for _, row in correlation_df.iterrows():

        print(
            f"  {row['Feature']}: "
            f"Spearman={row['Spearman_Correlation']:.4f}, "
            f"Pearson={row['Pearson_Correlation']:.4f}"
        )

    # --------------------------------------------------------
    # 8. Transformation recommendations
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("POTENTIAL TRANSFORMATION CANDIDATES")
    print("=" * 70)

    if len(highly_skewed) == 0:

        print("\nNo strong transformation candidates.")

    else:

        for _, row in highly_skewed.iterrows():

            feature = row["Feature"]
            skewness = row["Skewness"]

            if feature in [
                "Distance(mi)",
                "Precipitation(in)"
            ]:

                recommendation = (
                    "Consider log1p transformation"
                )

            else:

                recommendation = (
                    "Review distribution before transformation"
                )

            print(
                f"  {feature}: "
                f"skewness={skewness:.4f} -> "
                f"{recommendation}"
            )

    # --------------------------------------------------------
    # 9. Compact summary
    # --------------------------------------------------------

    summary_rows = []

    for _, row in analysis_df.iterrows():

        feature = row["Feature"]

        correlation_match = correlation_df[
            correlation_df["Feature"] == feature
        ]

        if len(correlation_match) > 0:

            spearman = correlation_match.iloc[0][
                "Spearman_Correlation"
            ]

        else:

            spearman = np.nan

        skewness = row["Skewness"]

        if abs(skewness) > 1:

            action = "Consider transformation"

        elif abs(skewness) > 0.5:

            action = "Moderate skewness"

        else:

            action = "No strong skewness"

        summary_rows.append({
            "Feature": feature,
            "Feature_Group": row["Feature_Group"],
            "Skewness": skewness,
            "IQR_Outlier_Percentage": (
                row["IQR_Outlier_Percentage"]
            ),
            "Spearman_Correlation_With_Severity": spearman,
            "Suggested_Action": action
        })

    summary_df = pd.DataFrame(
        summary_rows
    )

    summary_df = summary_df.sort_values(
        "Spearman_Correlation_With_Severity",
        key=lambda x: x.abs(),
        ascending=False
    )

    summary_df.to_csv(
        SUMMARY_REPORT,
        index=False
    )

    print(
        f"\nSaved summary:"
        f"\n  {SUMMARY_REPORT}"
    )

    # --------------------------------------------------------
    # 10. Frequency-encoded variables
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("FREQUENCY-ENCODED FEATURES")
    print("=" * 70)

    print(
        "\nThese features are numeric representations of "
        "categorical variables and are therefore excluded "
        "from continuous-feature skewness/IQR analysis:"
    )

    for column in frequency_features:

        print(f"  - {column}")

    # --------------------------------------------------------
    # 11. Binary / one-hot variables
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("BINARY / ONE-HOT FEATURES")
    print("=" * 70)

    one_hot_columns = []

    for column in train.columns:

        for prefix in ONE_HOT_PREFIXES:

            if column.startswith(prefix):

                one_hot_columns.append(column)
                break

    print(
        f"\nDetected {len(one_hot_columns)} "
        "one-hot/binary columns."
    )

    print(
        "These are excluded from continuous numerical "
        "distribution analysis."
    )

    # --------------------------------------------------------
    # 12. Leakage reminder
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("PIPELINE REMINDER")
    print("=" * 70)

    if "End_Time" in train.columns:

        print(
            "\nEnd_Time is still present."
        )

        print(
            "Remove End_Time before model training because "
            "it represents information available after "
            "the accident event."
        )

    if "Start_Time" in train.columns:

        print(
            "\nStart_Time is still present."
        )

        print(
            "Temporal features have already been extracted. "
            "The raw Start_Time column can be removed before "
            "model training."
        )

    print("\n" + "=" * 70)
    print("NUMERICAL FEATURE ANALYSIS COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
import os
import pandas as pd


# ============================================================
# Configuration
# ============================================================

INPUT_FILE = "data/train.csv"
OUTPUT_FILE = "outputs/reports/missing_value_treatment_plan.csv"


# ============================================================
# Load training data
# ============================================================

print("Loading training dataset...")

df = pd.read_csv(INPUT_FILE)

print(f"Training dataset shape: {df.shape}")


# ============================================================
# Define missing-value treatment decisions
# ============================================================

decisions = {

    # --------------------------------------------------------
    # Leakage-risk features
    # --------------------------------------------------------
    "End_Lat": (
        "EXCLUDE",
        "Potential post-event information and approximately 51% missing; "
        "exclude rather than impute to avoid leakage."
    ),

    "End_Lng": (
        "EXCLUDE",
        "Potential post-event information and approximately 51% missing; "
        "exclude rather than impute to avoid leakage."
    ),

    # --------------------------------------------------------
    # Weather numerical features
    # --------------------------------------------------------
    "Precipitation(in)": (
        "MISSING_INDICATOR_AND_MEDIAN",
        "High missingness (32.42%) and zero-inflated distribution; "
        "retain missingness information using an indicator and impute "
        "the numerical value using a training-set median."
    ),

    "Wind_Chill(F)": (
        "MISSING_INDICATOR_AND_MEDIAN",
        "High missingness (29.45%); retain missingness information "
        "and use training-set median imputation."
    ),

    "Wind_Speed(mph)": (
        "MISSING_INDICATOR_AND_MEDIAN",
        "Moderate missingness (8.12%) and suspicious extreme values; "
        "retain missingness information and use training-set median "
        "imputation after invalid-value investigation."
    ),

    "Visibility(mi)": (
        "MEDIAN",
        "Low missingness (2.27%); median imputation is robust to "
        "the concentrated distribution around 10 miles."
    ),

    "Humidity(%)": (
        "MEDIAN",
        "Low missingness (2.22%); median imputation is appropriate "
        "for a bounded numerical weather variable."
    ),

    "Temperature(F)": (
        "MEDIAN",
        "Low missingness (2.09%); median imputation is appropriate "
        "for the numerical weather variable."
    ),

    "Pressure(in)": (
        "MEDIAN",
        "Low missingness (1.78%); median imputation is robust to "
        "the suspicious extreme values identified during EDA."
    ),

    # --------------------------------------------------------
    # Weather categorical features
    # --------------------------------------------------------
    "Weather_Condition": (
        "MISSING_CATEGORY",
        "Categorical weather variable with 2.23% missing; "
        "represent missing observations explicitly rather than "
        "assuming the most common weather condition."
    ),

    "Wind_Direction": (
        "MISSING_CATEGORY",
        "Categorical weather variable with 2.17% missing; "
        "represent missing observations as a separate category."
    ),

    # --------------------------------------------------------
    # Weather timestamp
    # --------------------------------------------------------
    "Weather_Timestamp": (
        "INVESTIGATE",
        "Missingness is low (1.51%), but prediction-time availability "
        "must be verified before deciding whether to retain the feature."
    ),

    # --------------------------------------------------------
    # Geographic / categorical features
    # --------------------------------------------------------
    "Airport_Code": (
        "MISSING_CATEGORY",
        "Only 0.26% missing; preserve missingness explicitly while "
        "high-cardinality usefulness is evaluated."
    ),

    "Street": (
        "MISSING_CATEGORY",
        "Very low missingness (0.11%) but very high cardinality; "
        "represent missing values explicitly before deciding the "
        "final encoding strategy."
    ),

    "Timezone": (
        "MISSING_CATEGORY",
        "Very low missingness (0.10%) and only four categories; "
        "represent missing observations explicitly."
    ),

    "Zipcode": (
        "MISSING_CATEGORY",
        "Very low missingness (0.02%) but high cardinality; "
        "represent missing observations explicitly before encoding."
    ),

    "City": (
        "MISSING_CATEGORY",
        "Extremely low missingness (0.003%); use an explicit "
        "missing category for consistency."
    ),

    # --------------------------------------------------------
    # Temporal categorical features
    # --------------------------------------------------------
    "Sunrise_Sunset": (
        "MISSING_CATEGORY",
        "Very low missingness (0.24%); represent missing values "
        "explicitly and later assess redundancy with Start_Time-derived "
        "day/night features."
    ),

    "Civil_Twilight": (
        "MISSING_CATEGORY",
        "Very low missingness (0.24%); represent missing values "
        "explicitly and later assess redundancy."
    ),

    "Nautical_Twilight": (
        "MISSING_CATEGORY",
        "Very low missingness (0.24%); represent missing values "
        "explicitly and later assess redundancy."
    ),

    "Astronomical_Twilight": (
        "MISSING_CATEGORY",
        "Very low missingness (0.24%); represent missing values "
        "explicitly and later assess redundancy."
    ),

    # --------------------------------------------------------
    # Text
    # --------------------------------------------------------
    "Description": (
        "MISSING_CATEGORY",
        "Only one missing observation; if the text feature is retained "
        "for a later NLP experiment, missing text can be represented "
        "explicitly."
    ),
}


# ============================================================
# Validate decisions
# ============================================================

missing_columns = set(
    df.columns[df.isnull().any()]
)

decision_columns = set(decisions.keys())

missing_decisions = missing_columns - decision_columns

if missing_decisions:
    raise ValueError(
        f"Missing treatment decisions for: "
        f"{sorted(missing_decisions)}"
    )


# ============================================================
# Build treatment plan
# ============================================================

rows = []

for feature, (decision, reason) in decisions.items():

    missing_count = df[feature].isna().sum()
    missing_percentage = (
        missing_count / len(df)
    ) * 100

    rows.append({
        "Feature": feature,
        "Missing_Count": missing_count,
        "Missing_Percentage": missing_percentage,
        "Treatment": decision,
        "Reason": reason
    })


treatment_plan = pd.DataFrame(rows)

treatment_plan = treatment_plan.sort_values(
    by="Missing_Percentage",
    ascending=False
)


# ============================================================
# Save report
# ============================================================

os.makedirs("outputs/reports", exist_ok=True)

treatment_plan.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# Display results
# ============================================================

print("\nMissing-value treatment plan:")
print(
    treatment_plan.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)

print("\nTreatment summary:")
print(
    treatment_plan["Treatment"]
    .value_counts()
    .to_string()
)

print(f"\nSaved report to: {OUTPUT_FILE}")
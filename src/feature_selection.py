import pandas as pd
from pathlib import Path

# ============================================================
# Stage 4.1 - Feature Audit
# US Accidents Severity Prediction
# ============================================================

INPUT_FILE = Path("data/US_Accidents_1M.csv")
OUTPUT_FILE = Path("outputs/reports/feature_audit.csv")

# ------------------------------------------------------------
# Load dataset
# ------------------------------------------------------------

print("Loading dataset...")
df = pd.read_csv(INPUT_FILE)

print(f"Dataset shape: {df.shape}")
print()

# ------------------------------------------------------------
# Define target
# ------------------------------------------------------------

TARGET = "Severity"

# ------------------------------------------------------------
# Feature decisions based on Stage 3 EDA
# ------------------------------------------------------------

decisions = {

    # -------------------------
    # Target
    # -------------------------
    "Severity": (
        "TARGET",
        "Target variable for multiclass classification"
    ),

    # -------------------------
    # Exclude
    # -------------------------
    "ID": (
        "EXCLUDE",
        "Identifier with no meaningful predictive information"
    ),

    "Country": (
        "EXCLUDE",
        "Constant column; contains no variation"
    ),

    "Turning_Loop": (
        "EXCLUDE",
        "Constant column; contains no variation"
    ),

    # -------------------------
    # Potential leakage
    # -------------------------
    "End_Time": (
        "LEAKAGE_RISK",
        "Potential post-event information unavailable at prediction time"
    ),

    "End_Lat": (
        "LEAKAGE_RISK",
        "Potential post-event information and approximately 51% missing"
    ),

    "End_Lng": (
        "LEAKAGE_RISK",
        "Potential post-event information and approximately 51% missing"
    ),

    # -------------------------
    # Data source
    # -------------------------
    "Source": (
        "INVESTIGATE",
        "Record source with 3 categories; may reflect data-collection differences or source bias"
    ),

    # -------------------------
    # Temporal
    # -------------------------
    "Start_Time": (
        "TEMPORAL",
        "Used to derive hour, day, month, year, weekday, weekend, night and cyclical time features"
    ),

    "Weather_Timestamp": (
        "INVESTIGATE",
        "Weather observation time must be checked for prediction-time availability"
    ),

    # -------------------------
    # Geographic
    # -------------------------
    "Start_Lat": (
        "GEOGRAPHIC",
        "Accident starting latitude; potentially useful for spatial patterns"
    ),

    "Start_Lng": (
        "GEOGRAPHIC",
        "Accident starting longitude; potentially useful for spatial patterns"
    ),

    # -------------------------
    # High-cardinality categorical
    # -------------------------
    "Street": (
        "HIGH_CARDINALITY",
        "Approximately 126k unique values; investigate frequency encoding or alternative representation"
    ),

    "City": (
        "CATEGORICAL",
        "Categorical geographic variable; encoding strategy to be determined"
    ),

    "County": (
        "CATEGORICAL",
        "Categorical geographic variable; encoding strategy to be determined"
    ),

    "State": (
        "CATEGORICAL",
        "Low-cardinality geographic variable suitable for categorical encoding"
    ),

    "Zipcode": (
        "HIGH_CARDINALITY",
        "Approximately 212k unique values; investigate frequency encoding or exclusion"
    ),

    "Timezone": (
        "CATEGORICAL",
        "Low-cardinality categorical variable"
    ),

    "Airport_Code": (
        "HIGH_CARDINALITY",
        "Approximately 1,950 unique values; investigate usefulness and encoding"
    ),

    # -------------------------
    # Text
    # -------------------------
    "Description": (
        "TEXT_INVESTIGATE",
        "High-cardinality free-text field; possible post-event information; NLP not included in initial pipeline"
    ),

    # -------------------------
    # Numerical accident context
    # -------------------------
    "Distance(mi)": (
        "NUMERICAL_INVESTIGATE",
        "Strong right skew; investigate log1p transformation and prediction-time availability"
    ),

    # -------------------------
    # Weather
    # -------------------------
    "Temperature(F)": (
        "WEATHER_NUMERICAL",
        "Weather variable with approximately 2.09% missing values"
    ),

    "Wind_Chill(F)": (
        "WEATHER_NUMERICAL",
        "Approximately 29.50% missing; investigate missingness indicator and imputation"
    ),

    "Humidity(%)": (
        "WEATHER_NUMERICAL",
        "Approximately 2.22% missing"
    ),

    "Pressure(in)": (
        "WEATHER_NUMERICAL",
        "Contains suspicious extreme values; requires domain-aware validation"
    ),

    "Visibility(mi)": (
        "WEATHER_NUMERICAL",
        "Highly concentrated at 10 miles; IQR method is unsuitable for direct outlier removal"
    ),

    "Wind_Direction": (
        "WEATHER_CATEGORICAL",
        "Categorical weather variable with approximately 2.18% missing"
    ),

    "Wind_Speed(mph)": (
        "WEATHER_NUMERICAL",
        "Contains suspicious extreme value of 822.8 mph; requires validation"
    ),

    "Precipitation(in)": (
        "WEATHER_NUMERICAL",
        "Approximately 32.45% missing and strongly zero-inflated; investigate missingness indicator and log1p"
    ),

    "Weather_Condition": (
        "WEATHER_CATEGORICAL",
        "Categorical weather variable with approximately 2.24% missing"
    ),

    # -------------------------
    # Road/context Boolean variables
    # -------------------------
    "Amenity": (
        "BOOLEAN",
        "Road/environment context indicator"
    ),

    "Bump": (
        "BOOLEAN",
        "Road/environment context indicator"
    ),

    "Crossing": (
        "BOOLEAN",
        "Road/environment context indicator"
    ),

    "Give_Way": (
        "BOOLEAN",
        "Road/environment context indicator"
    ),

    "Junction": (
        "BOOLEAN",
        "Road/environment context indicator"
    ),

    "No_Exit": (
        "BOOLEAN",
        "Road/environment context indicator"
    ),

    "Railway": (
        "BOOLEAN",
        "Road/environment context indicator"
    ),

    "Roundabout": (
        "BOOLEAN",
        "Road/environment context indicator"
    ),

    "Station": (
        "BOOLEAN",
        "Road/environment context indicator"
    ),

    "Stop": (
        "BOOLEAN",
        "Road/environment context indicator"
    ),

    "Traffic_Calming": (
        "BOOLEAN",
        "Road/environment context indicator"
    ),

    "Traffic_Signal": (
        "BOOLEAN",
        "Road/environment context indicator"
    ),

    # -------------------------
    # Twilight variables
    # -------------------------
    "Sunrise_Sunset": (
        "TEMPORAL_CATEGORICAL",
        "Day/night context; potentially redundant with Start_Time-derived features"
    ),

    "Civil_Twilight": (
        "TEMPORAL_CATEGORICAL",
        "Twilight context; investigate redundancy with derived temporal features"
    ),

    "Nautical_Twilight": (
        "TEMPORAL_CATEGORICAL",
        "Twilight context; investigate redundancy with derived temporal features"
    ),

    "Astronomical_Twilight": (
        "TEMPORAL_CATEGORICAL",
        "Twilight context; investigate redundancy with derived temporal features"
    ),
}

# ------------------------------------------------------------
# Verify all columns are accounted for
# ------------------------------------------------------------

dataset_columns = set(df.columns)
audited_columns = set(decisions.keys())

missing_from_audit = dataset_columns - audited_columns
extra_in_audit = audited_columns - dataset_columns

if missing_from_audit:
    raise ValueError(
        f"Columns missing from feature audit: {sorted(missing_from_audit)}"
    )

if extra_in_audit:
    raise ValueError(
        f"Columns in audit but not dataset: {sorted(extra_in_audit)}"
    )

# ------------------------------------------------------------
# Create feature audit table
# ------------------------------------------------------------

rows = []

for column in df.columns:

    decision, reason = decisions[column]

    rows.append({
        "Feature": column,
        "Data_Type": str(df[column].dtype),
        "Unique_Values": df[column].nunique(dropna=True),
        "Missing_Count": int(df[column].isna().sum()),
        "Missing_Percentage": round(
            df[column].isna().mean() * 100, 2
        ),
        "Decision": decision,
        "Reason": reason
    })

audit_df = pd.DataFrame(rows)

# ------------------------------------------------------------
# Save report
# ------------------------------------------------------------

OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

audit_df.to_csv(
    OUTPUT_FILE,
    index=False
)

# ------------------------------------------------------------
# Print summary
# ------------------------------------------------------------

print("Feature audit completed.")
print()
print(audit_df.to_string(index=False))

print("\nDecision summary:")
print(audit_df["Decision"].value_counts())

print(f"\nSaved report to: {OUTPUT_FILE}")
import pandas as pd
from pathlib import Path

# ============================================================
# Stage 4.4.1 - Invalid Value Analysis
# ============================================================

INPUT_FILE = Path("data/train_missing_handled.csv")
OUTPUT_DIR = Path("outputs/reports")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 70)
print("STAGE 4.4.1 - INVALID VALUE ANALYSIS")
print("=" * 70)

# ------------------------------------------------------------
# 1. Load training data
# ------------------------------------------------------------

df = pd.read_csv(INPUT_FILE)

print(f"\nTraining dataset shape: {df.shape}")

# ------------------------------------------------------------
# 2. Define suspicious/domain-questionable ranges
# ------------------------------------------------------------
# These are investigation thresholds, NOT automatic deletion rules.

checks = {
    "Temperature(F)": {
        "condition": (df["Temperature(F)"] < -80) | (df["Temperature(F)"] > 140),
        "reason": "Physically unusual temperature"
    },

    "Wind_Speed(mph)": {
        "condition": (df["Wind_Speed(mph)"] < 0) | (df["Wind_Speed(mph)"] > 100),
        "reason": "Physically questionable wind speed"
    },

    "Pressure(in)": {
        "condition": (df["Pressure(in)"] < 20) | (df["Pressure(in)"] > 35),
        "reason": "Physically questionable atmospheric pressure"
    },

    "Visibility(mi)": {
        "condition": (df["Visibility(mi)"] < 0) | (df["Visibility(mi)"] > 100),
        "reason": "Outside expected visibility range"
    },

    "Humidity(%)": {
        "condition": (df["Humidity(%)"] < 0) | (df["Humidity(%)"] > 100),
        "reason": "Outside valid percentage range"
    },

    "Precipitation(in)": {
        "condition": df["Precipitation(in)"] < 0,
        "reason": "Negative precipitation"
    },

    "Distance(mi)": {
        "condition": (df["Distance(mi)"] < 0) | (df["Distance(mi)"] > 200),
        "reason": "Negative or extremely large distance"
    }
}

# ------------------------------------------------------------
# 3. Count suspicious records
# ------------------------------------------------------------

results = []

print("\nSuspicious-value summary:")
print("-" * 70)

for feature, info in checks.items():

    condition = info["condition"].fillna(False)

    count = int(condition.sum())
    percentage = (count / len(df)) * 100

    results.append({
        "Feature": feature,
        "Suspicious_Count": count,
        "Suspicious_Percentage": round(percentage, 4),
        "Reason": info["reason"]
    })

    print(
        f"{feature:<22} "
        f"{count:>8,} records "
        f"({percentage:>7.4f}%)"
    )

results_df = pd.DataFrame(results)

# ------------------------------------------------------------
# 4. Save summary
# ------------------------------------------------------------

summary_file = OUTPUT_DIR / "invalid_value_summary.csv"
results_df.to_csv(summary_file, index=False)

print(f"\nSaved: {summary_file}")

# ------------------------------------------------------------
# 5. Investigate suspicious records
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("DETAILED INVESTIGATION")
print("=" * 70)

for feature, info in checks.items():

    condition = info["condition"].fillna(False)

    suspicious = df.loc[condition]

    if len(suspicious) == 0:
        continue

    print(f"\n{'-' * 70}")
    print(f"Feature: {feature}")
    print(f"Reason : {info['reason']}")
    print(f"Count  : {len(suspicious):,}")

    # Show extreme values
    print("\nExtreme values:")

    if pd.api.types.is_numeric_dtype(df[feature]):
        print(
            suspicious[feature]
            .sort_values()
            .head(5)
            .to_string(index=False)
        )

        print("...")

        print(
            suspicious[feature]
            .sort_values(ascending=False)
            .head(5)
            .to_string(index=False)
        )

    # Severity distribution
    print("\nSeverity distribution:")

    severity_counts = (
        suspicious["Severity"]
        .value_counts()
        .sort_index()
    )

    print(severity_counts.to_string())

    # Year distribution
    if "Start_Time" in suspicious.columns:

        years = pd.to_datetime(
            suspicious["Start_Time"],
            errors="coerce"
        ).dt.year

        print("\nYear distribution:")

        print(
            years.value_counts()
            .sort_index()
            .to_string()
        )

    # State distribution
    if "State" in suspicious.columns:

        print("\nTop 10 states:")

        print(
            suspicious["State"]
            .value_counts()
            .head(10)
            .to_string()
        )

    # Source distribution
    if "Source" in suspicious.columns:

        print("\nSource distribution:")

        print(
            suspicious["Source"]
            .value_counts()
            .to_string()
        )

# ------------------------------------------------------------
# 6. Create combined suspicious-record indicator
# ------------------------------------------------------------

combined_suspicious = pd.Series(False, index=df.index)

for feature, info in checks.items():

    condition = info["condition"].fillna(False)

    combined_suspicious = (
        combined_suspicious | condition
    )

combined_count = int(combined_suspicious.sum())
combined_percentage = (
    combined_count / len(df)
) * 100

print("\n" + "=" * 70)
print("COMBINED RESULT")
print("=" * 70)

print(
    f"Records with at least one suspicious value: "
    f"{combined_count:,}"
)

print(
    f"Percentage of training data: "
    f"{combined_percentage:.4f}%"
)

# ------------------------------------------------------------
# 7. Save suspicious records
# ------------------------------------------------------------

suspicious_columns = [
    "ID",
    "Severity",
    "Start_Time",
    "State",
    "Source",
    "Temperature(F)",
    "Wind_Speed(mph)",
    "Pressure(in)",
    "Visibility(mi)",
    "Precipitation(in)",
    "Distance(mi)"
]

suspicious_columns = [
    col for col in suspicious_columns
    if col in df.columns
]

suspicious_records = df.loc[
    combined_suspicious,
    suspicious_columns
]

suspicious_file = OUTPUT_DIR / "suspicious_records.csv"

suspicious_records.to_csv(
    suspicious_file,
    index=False
)

print(f"\nSaved: {suspicious_file}")

print("\n" + "=" * 70)
print("INVALID VALUE ANALYSIS COMPLETED")
print("=" * 70)
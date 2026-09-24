import os
import pandas as pd
import numpy as np

# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------
INPUT_FILE = "data/US_Accidents_1M.csv"
REPORT_DIR = "outputs/reports"

os.makedirs(REPORT_DIR, exist_ok=True)


# ---------------------------------------------------------
# Load dataset
# ---------------------------------------------------------
print("Loading dataset...")

df = pd.read_csv(INPUT_FILE)

print(f"Dataset shape: {df.shape}")


# ---------------------------------------------------------
# Numerical variables to investigate
# ---------------------------------------------------------
features = [
    "Distance(mi)",
    "Temperature(F)",
    "Wind_Chill(F)",
    "Humidity(%)",
    "Pressure(in)",
    "Visibility(mi)",
    "Wind_Speed(mph)",
    "Precipitation(in)"
]


# ---------------------------------------------------------
# IQR analysis
# ---------------------------------------------------------
results = []

for feature in features:

    series = df[feature].dropna()

    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)

    iqr = q3 - q1

    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    outliers = series[
        (series < lower_bound) |
        (series > upper_bound)
    ]

    results.append({
        "Feature": feature,
        "Q1": q1,
        "Median": series.median(),
        "Q3": q3,
        "IQR": iqr,
        "Lower_Bound": lower_bound,
        "Upper_Bound": upper_bound,
        "Minimum": series.min(),
        "Maximum": series.max(),
        "Outlier_Count": len(outliers),
        "Outlier_Percentage": len(outliers) / len(series) * 100
    })


outlier_report = pd.DataFrame(results)

outlier_report = outlier_report.round(4)


# ---------------------------------------------------------
# Save report
# ---------------------------------------------------------
output_file = (
    f"{REPORT_DIR}/outlier_analysis.csv"
)

outlier_report.to_csv(
    output_file,
    index=False
)


# ---------------------------------------------------------
# Display results
# ---------------------------------------------------------
print("\nIQR-based outlier analysis:")
print(
    outlier_report.to_string(index=False)
)

print(
    f"\nSaved report to: {output_file}"
)


# ---------------------------------------------------------
# Domain-oriented extreme observations
# ---------------------------------------------------------
print("\nExtreme observations:")

for feature in features:

    series = df[feature]

    print(
        f"{feature}: "
        f"min={series.min()}, "
        f"max={series.max()}, "
        f"median={series.median()}"
    )
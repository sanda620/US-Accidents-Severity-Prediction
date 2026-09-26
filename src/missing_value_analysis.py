import os
import pandas as pd


# ============================================================
# Configuration
# ============================================================

INPUT_FILE = "data/train.csv"
REPORT_FILE = "outputs/reports/train_missing_value_analysis.csv"


# ============================================================
# Load training data
# ============================================================

print("Loading training dataset...")

df = pd.read_csv(INPUT_FILE)

print(f"Training dataset shape: {df.shape}")


# ============================================================
# Calculate missing-value statistics
# ============================================================

missing_count = df.isnull().sum()
missing_percentage = (missing_count / len(df)) * 100

analysis = pd.DataFrame({
    "Feature": df.columns,
    "Data_Type": df.dtypes.astype(str).values,
    "Missing_Count": missing_count.values,
    "Missing_Percentage": missing_percentage.values,
    "Unique_Values": [
        df[col].nunique(dropna=True)
        for col in df.columns
    ]
})


# ============================================================
# Keep features with missing values
# ============================================================

analysis = analysis[
    analysis["Missing_Count"] > 0
].copy()

analysis = analysis.sort_values(
    by="Missing_Percentage",
    ascending=False
)


# ============================================================
# Categorize missingness level
# ============================================================

def classify_missingness(percentage):

    if percentage == 0:
        return "NONE"
    elif percentage < 1:
        return "VERY_LOW"
    elif percentage < 5:
        return "LOW"
    elif percentage < 20:
        return "MODERATE"
    elif percentage < 50:
        return "HIGH"
    else:
        return "VERY_HIGH"


analysis["Missingness_Level"] = (
    analysis["Missing_Percentage"]
    .apply(classify_missingness)
)


# ============================================================
# Create output directory
# ============================================================

os.makedirs("outputs/reports", exist_ok=True)


# ============================================================
# Save report
# ============================================================

analysis.to_csv(REPORT_FILE, index=False)


# ============================================================
# Display results
# ============================================================

print("\nTraining-set missing value analysis:")
print(
    analysis.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)

print("\nMissingness level summary:")
print(
    analysis["Missingness_Level"]
    .value_counts()
    .to_string()
)

print(f"\nSaved report to: {REPORT_FILE}")
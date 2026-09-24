import os
import pandas as pd


# ============================================================
# Configuration
# ============================================================

DATA_FILE = "data/US_Accidents_1M.csv"
OUTPUT_DIR = "outputs/reports"

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# Load dataset
# ============================================================

print("=" * 70)
print("STAGE 3 - DATA QUALITY ANALYSIS")
print("=" * 70)

print("\nLoading dataset...")

df = pd.read_csv(DATA_FILE)

print("Dataset loaded successfully.")


# ============================================================
# 1. Dataset dimensions
# ============================================================

print("\n" + "=" * 70)
print("1. DATASET DIMENSIONS")
print("=" * 70)

print(f"Rows    : {df.shape[0]:,}")
print(f"Columns : {df.shape[1]}")


# ============================================================
# 2. Column information
# ============================================================

print("\n" + "=" * 70)
print("2. COLUMN INFORMATION")
print("=" * 70)

column_info = pd.DataFrame({
    "Column": df.columns,
    "Data_Type": df.dtypes.astype(str).values,
    "Unique_Values": [
        df[column].nunique(dropna=True)
        for column in df.columns
    ],
    "Missing_Values": [
        df[column].isna().sum()
        for column in df.columns
    ],
    "Missing_Percentage": [
        df[column].isna().mean() * 100
        for column in df.columns
    ]
})

print(column_info.to_string(index=False))

column_info.to_csv(
    f"{OUTPUT_DIR}/column_profile.csv",
    index=False
)


# ============================================================
# 3. Duplicate records
# ============================================================

print("\n" + "=" * 70)
print("3. DUPLICATE RECORDS")
print("=" * 70)

duplicate_count = df.duplicated().sum()

print(f"Duplicate rows: {duplicate_count:,}")

duplicate_result = pd.DataFrame({
    "Metric": ["Duplicate Rows"],
    "Count": [duplicate_count]
})

duplicate_result.to_csv(
    f"{OUTPUT_DIR}/duplicate_analysis.csv",
    index=False
)


# ============================================================
# 4. Constant columns
# ============================================================

print("\n" + "=" * 70)
print("4. CONSTANT COLUMNS")
print("=" * 70)

constant_columns = [
    column
    for column in df.columns
    if df[column].nunique(dropna=False) <= 1
]

if constant_columns:
    print("Constant columns:")
    for column in constant_columns:
        print(f"- {column}")
else:
    print("No constant columns found.")

constant_result = pd.DataFrame({
    "Constant_Column": constant_columns
})

constant_result.to_csv(
    f"{OUTPUT_DIR}/constant_columns.csv",
    index=False
)


# ============================================================
# 5. Target variable
# ============================================================

print("\n" + "=" * 70)
print("5. TARGET VARIABLE - SEVERITY")
print("=" * 70)

severity_counts = (
    df["Severity"]
    .value_counts()
    .sort_index()
)

severity_percentage = (
    df["Severity"]
    .value_counts(normalize=True)
    .sort_index()
    * 100
)

severity_summary = pd.DataFrame({
    "Severity": severity_counts.index,
    "Count": severity_counts.values,
    "Percentage": severity_percentage.values
})

print(
    severity_summary.to_string(index=False)
)

severity_summary.to_csv(
    f"{OUTPUT_DIR}/severity_distribution.csv",
    index=False
)


# ============================================================
# 6. Numeric summary
# ============================================================

print("\n" + "=" * 70)
print("6. NUMERICAL VARIABLE SUMMARY")
print("=" * 70)

numeric_summary = df.describe().T

numeric_summary.to_csv(
    f"{OUTPUT_DIR}/numeric_summary.csv"
)

print(numeric_summary.to_string())


# ============================================================
# 7. Memory usage
# ============================================================

print("\n" + "=" * 70)
print("7. MEMORY USAGE")
print("=" * 70)

memory_mb = (
    df.memory_usage(deep=True).sum()
    / (1024 ** 2)
)

print(f"Memory usage: {memory_mb:.2f} MB")


# ============================================================
# Complete
# ============================================================

print("\n" + "=" * 70)
print("DATA QUALITY ANALYSIS COMPLETE")
print("=" * 70)

print(f"\nReports saved to: {OUTPUT_DIR}/")
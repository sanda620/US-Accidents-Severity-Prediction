import os
import pandas as pd
import matplotlib.pyplot as plt

# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------
INPUT_FILE = "data/US_Accidents_1M.csv"
FIGURE_DIR = "outputs/figures"
REPORT_DIR = "outputs/reports"

os.makedirs(FIGURE_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)


# ---------------------------------------------------------
# Load dataset
# ---------------------------------------------------------
print("Loading dataset...")
df = pd.read_csv(INPUT_FILE)

print(f"Dataset shape: {df.shape}")


# ---------------------------------------------------------
# Prepare time features
# ---------------------------------------------------------
df["Start_Time"] = pd.to_datetime(df["Start_Time"], errors="coerce")

df["Year"] = df["Start_Time"].dt.year
df["Month"] = df["Start_Time"].dt.month
df["Hour"] = df["Start_Time"].dt.hour
df["Day_of_Week"] = df["Start_Time"].dt.day_name()


# =========================================================
# 1. SEVERITY DISTRIBUTION
# =========================================================
severity_counts = df["Severity"].value_counts().sort_index()

plt.figure(figsize=(8, 5))
severity_counts.plot(kind="bar")

plt.title("Traffic Accident Severity Distribution")
plt.xlabel("Severity")
plt.ylabel("Number of Accidents")
plt.xticks(rotation=0)
plt.tight_layout()

plt.savefig(
    f"{FIGURE_DIR}/01_severity_distribution.png",
    dpi=300
)
plt.close()

severity_percent = (
    severity_counts / severity_counts.sum() * 100
).round(2)

severity_report = pd.DataFrame({
    "Severity": severity_counts.index,
    "Count": severity_counts.values,
    "Percentage": severity_percent.values
})

severity_report.to_csv(
    f"{REPORT_DIR}/severity_distribution.csv",
    index=False
)


# =========================================================
# 2. MISSING VALUES
# =========================================================
missing = df.isnull().sum()
missing_percent = (missing / len(df) * 100).round(2)

missing_report = pd.DataFrame({
    "Missing_Count": missing,
    "Missing_Percentage": missing_percent
})

missing_report = missing_report[
    missing_report["Missing_Count"] > 0
].sort_values(
    "Missing_Percentage",
    ascending=False
)

missing_report.to_csv(
    f"{REPORT_DIR}/missing_values.csv"
)

plt.figure(figsize=(10, 7))

missing_plot = missing_report.head(20)

missing_plot["Missing_Percentage"].sort_values().plot(
    kind="barh"
)

plt.title("Top Variables with Missing Values")
plt.xlabel("Missing Values (%)")
plt.ylabel("Feature")
plt.tight_layout()

plt.savefig(
    f"{FIGURE_DIR}/02_missing_values.png",
    dpi=300
)
plt.close()


# =========================================================
# 3. SEVERITY BY YEAR - RAW COUNTS
# =========================================================
year_severity = pd.crosstab(
    df["Year"],
    df["Severity"]
)

year_severity.to_csv(
    f"{REPORT_DIR}/year_severity_counts.csv"
)

plt.figure(figsize=(10, 6))

year_severity.plot(
    kind="bar",
    figsize=(10, 6)
)

plt.title("Accident Severity by Year")
plt.xlabel("Year")
plt.ylabel("Number of Accidents")
plt.xticks(rotation=45)
plt.tight_layout()

plt.savefig(
    f"{FIGURE_DIR}/03_year_severity_counts.png",
    dpi=300
)
plt.close()


# =========================================================
# 4. SEVERITY BY YEAR - PERCENTAGES
# =========================================================
year_severity_percent = (
    pd.crosstab(
        df["Year"],
        df["Severity"],
        normalize="index"
    ) * 100
)

year_severity_percent = year_severity_percent.round(2)

year_severity_percent.to_csv(
    f"{REPORT_DIR}/year_severity_percentages.csv"
)

plt.figure(figsize=(10, 6))

year_severity_percent.plot(
    kind="bar",
    figsize=(10, 6)
)

plt.title("Severity Distribution Within Each Year")
plt.xlabel("Year")
plt.ylabel("Percentage of Accidents (%)")
plt.xticks(rotation=45)
plt.legend(title="Severity")
plt.tight_layout()

plt.savefig(
    f"{FIGURE_DIR}/04_year_severity_percentages.png",
    dpi=300
)
plt.close()


# =========================================================
# 5. MISSINGNESS BY SEVERITY
# =========================================================
important_missing_features = [
    "End_Lat",
    "End_Lng",
    "Wind_Chill(F)",
    "Precipitation(in)",
    "Wind_Speed(mph)",
    "Temperature(F)",
    "Humidity(%)",
    "Pressure(in)",
    "Visibility(mi)",
    "Weather_Condition"
]

missing_by_severity = {}

for feature in important_missing_features:

    if feature in df.columns:

        result = (
            df.groupby("Severity")[feature]
            .apply(lambda x: x.isna().mean() * 100)
        )

        missing_by_severity[feature] = result

missing_severity_df = pd.DataFrame(
    missing_by_severity
).T

missing_severity_df.columns = [
    f"Severity_{int(col)}"
    for col in missing_severity_df.columns
]

missing_severity_df = missing_severity_df.round(2)

missing_severity_df.to_csv(
    f"{REPORT_DIR}/missingness_by_severity.csv"
)

plt.figure(figsize=(12, 7))

missing_severity_df.plot(
    kind="bar",
    figsize=(12, 7)
)

plt.title("Missing Value Percentage by Severity")
plt.xlabel("Feature")
plt.ylabel("Missing Values (%)")
plt.xticks(rotation=45, ha="right")
plt.legend(title="Severity")
plt.tight_layout()

plt.savefig(
    f"{FIGURE_DIR}/05_missingness_by_severity.png",
    dpi=300
)
plt.close()


# =========================================================
# 6. WEATHER VARIABLES VS SEVERITY
# =========================================================
weather_features = [
    "Temperature(F)",
    "Humidity(%)",
    "Visibility(mi)",
    "Wind_Speed(mph)",
    "Precipitation(in)"
]

weather_summary = (
    df.groupby("Severity")[weather_features]
    .agg(["mean", "median"])
)

weather_summary.to_csv(
    f"{REPORT_DIR}/weather_by_severity.csv"
)


# Boxplots for important weather variables
for feature in weather_features:

    plt.figure(figsize=(9, 6))

    df.boxplot(
        column=feature,
        by="Severity"
    )

    plt.title(f"{feature} by Severity")
    plt.suptitle("")
    plt.xlabel("Severity")
    plt.ylabel(feature)
    plt.tight_layout()

    safe_name = (
        feature
        .replace("(", "")
        .replace(")", "")
        .replace("/", "_")
        .replace(" ", "_")
    )

    plt.savefig(
        f"{FIGURE_DIR}/weather_{safe_name}_by_severity.png",
        dpi=300
    )

    plt.close()


# =========================================================
# 7. SEVERITY BY STATE
# =========================================================
state_severity = pd.crosstab(
    df["State"],
    df["Severity"],
    normalize="index"
) * 100

state_severity = state_severity.round(2)

state_severity.to_csv(
    f"{REPORT_DIR}/state_severity_percentages.csv"
)

# Select states with highest number of accidents
top_states = df["State"].value_counts().head(15).index

state_plot = state_severity.loc[
    state_severity.index.intersection(top_states)
]

plt.figure(figsize=(12, 7))

state_plot.plot(
    kind="bar",
    figsize=(12, 7)
)

plt.title(
    "Severity Distribution for Top 15 States by Accident Count"
)

plt.xlabel("State")
plt.ylabel("Percentage of Accidents (%)")
plt.xticks(rotation=45)
plt.legend(title="Severity")
plt.tight_layout()

plt.savefig(
    f"{FIGURE_DIR}/06_state_severity.png",
    dpi=300
)

plt.close()


# =========================================================
# 8. SUMMARY
# =========================================================
print("\nEDA visualization completed successfully.")

print("\nGenerated figures:")
for file in sorted(os.listdir(FIGURE_DIR)):
    print(" -", file)

print("\nGenerated reports:")
for file in sorted(os.listdir(REPORT_DIR)):
    print(" -", file)

print("\nKey EDA areas completed:")
print("1. Severity distribution")
print("2. Missing-value analysis")
print("3. Year × Severity analysis")
print("4. Within-year severity percentages")
print("5. Missingness by severity")
print("6. Weather × Severity analysis")
print("7. State × Severity analysis")
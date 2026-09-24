import os
import pandas as pd

# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------
INPUT_FILE = "data/US_Accidents_1M.csv"
OUTPUT_FILE = "outputs/reports/leakage_analysis.csv"

os.makedirs("outputs/reports", exist_ok=True)

# ---------------------------------------------------------
# Load dataset
# ---------------------------------------------------------
df = pd.read_csv(INPUT_FILE, nrows=1000)

print(f"Loaded sample for leakage assessment: {df.shape}")


# ---------------------------------------------------------
# Leakage / prediction-time availability assessment
# ---------------------------------------------------------
assessment = {

    "ID": {
        "availability": "Exclude",
        "reason": "Unique record identifier; does not represent accident characteristics."
    },

    "Source": {
        "availability": "Investigate",
        "reason": "May reflect the data collection source rather than the accident itself."
    },

    "Severity": {
        "availability": "Target",
        "reason": "Prediction target; must not be used as an input feature."
    },

    "Start_Time": {
        "availability": "Available",
        "reason": "Represents the accident start/reporting time and can support temporal features."
    },

    "End_Time": {
        "availability": "Potential leakage",
        "reason": "Event end time may only become known after the incident progresses."
    },

    "Start_Lat": {
        "availability": "Available",
        "reason": "Starting geographic location can reasonably be available when the incident is reported."
    },

    "Start_Lng": {
        "availability": "Available",
        "reason": "Starting geographic location can reasonably be available when the incident is reported."
    },

    "End_Lat": {
        "availability": "Potential leakage",
        "reason": "Ending location may be determined after the event and is also substantially missing."
    },

    "End_Lng": {
        "availability": "Potential leakage",
        "reason": "Ending location may be determined after the event and is also substantially missing."
    },

    "Distance(mi)": {
        "availability": "Investigate",
        "reason": "May describe the spatial extent of the event and could contain post-event information."
    },

    "Description": {
        "availability": "Investigate",
        "reason": "Free-text information may contain details added after initial reporting."
    },

    "Street": {
        "availability": "Available",
        "reason": "Road/street information can potentially be known at incident reporting."
    },

    "City": {
        "availability": "Available",
        "reason": "Location information can potentially be known at incident reporting."
    },

    "County": {
        "availability": "Available",
        "reason": "Geographical information associated with the incident location."
    },

    "State": {
        "availability": "Available",
        "reason": "Geographical information associated with the incident location."
    },

    "Zipcode": {
        "availability": "Available",
        "reason": "Location information can potentially be known at incident reporting."
    },

    "Country": {
        "availability": "Exclude",
        "reason": "Constant value in the working dataset."
    },

    "Timezone": {
        "availability": "Available",
        "reason": "Can be derived from or associated with the incident location."
    },

    "Airport_Code": {
        "availability": "Investigate",
        "reason": "Related to nearby weather reporting infrastructure rather than directly to the accident."
    },

    "Weather_Timestamp": {
        "availability": "Investigate",
        "reason": "Must ensure the weather observation does not occur after the prediction time."
    },

    "Temperature(F)": {
        "availability": "Investigate",
        "reason": "Potentially available from contemporaneous weather observations."
    },

    "Wind_Chill(F)": {
        "availability": "Investigate",
        "reason": "Potentially available from contemporaneous weather observations."
    },

    "Humidity(%)": {
        "availability": "Investigate",
        "reason": "Potentially available from contemporaneous weather observations."
    },

    "Pressure(in)": {
        "availability": "Investigate",
        "reason": "Potentially available from contemporaneous weather observations."
    },

    "Visibility(mi)": {
        "availability": "Investigate",
        "reason": "Potentially available from contemporaneous weather observations."
    },

    "Wind_Direction": {
        "availability": "Investigate",
        "reason": "Potentially available from contemporaneous weather observations."
    },

    "Wind_Speed(mph)": {
        "availability": "Investigate",
        "reason": "Potentially available from contemporaneous weather observations."
    },

    "Precipitation(in)": {
        "availability": "Investigate",
        "reason": "Potentially available from contemporaneous weather observations, but missingness is substantial."
    },

    "Weather_Condition": {
        "availability": "Investigate",
        "reason": "Potentially available from contemporaneous weather observations."
    },

    "Amenity": {
        "availability": "Available",
        "reason": "Road/environment context that can potentially be known at the incident location."
    },

    "Bump": {
        "availability": "Available",
        "reason": "Road/environment context that can potentially be known at the incident location."
    },

    "Crossing": {
        "availability": "Available",
        "reason": "Road intersection/context information."
    },

    "Give_Way": {
        "availability": "Available",
        "reason": "Road traffic-control context."
    },

    "Junction": {
        "availability": "Available",
        "reason": "Road intersection/context information."
    },

    "No_Exit": {
        "availability": "Available",
        "reason": "Road/environment context."
    },

    "Railway": {
        "availability": "Available",
        "reason": "Road/environment context."
    },

    "Roundabout": {
        "availability": "Available",
        "reason": "Road/environment context."
    },

    "Station": {
        "availability": "Available",
        "reason": "Road/environment context."
    },

    "Stop": {
        "availability": "Available",
        "reason": "Traffic-control context."
    },

    "Traffic_Calming": {
        "availability": "Available",
        "reason": "Road/environment context."
    },

    "Traffic_Signal": {
        "availability": "Available",
        "reason": "Traffic-control context."
    },

    "Turning_Loop": {
        "availability": "Exclude",
        "reason": "Constant value in the working dataset."
    },

    "Sunrise_Sunset": {
        "availability": "Available",
        "reason": "Can be derived from accident time and location."
    },

    "Civil_Twilight": {
        "availability": "Available",
        "reason": "Can be derived from accident time and location."
    },

    "Nautical_Twilight": {
        "availability": "Available",
        "reason": "Can be derived from accident time and location."
    },

    "Astronomical_Twilight": {
        "availability": "Available",
        "reason": "Can be derived from accident time and location."
    }
}


# ---------------------------------------------------------
# Convert assessment to DataFrame
# ---------------------------------------------------------
rows = []

for feature in df.columns:

    if feature in assessment:

        rows.append({
            "Feature": feature,
            "Availability_Assessment":
                assessment[feature]["availability"],
            "Reason":
                assessment[feature]["reason"]
        })

    else:

        rows.append({
            "Feature": feature,
            "Availability_Assessment": "Needs review",
            "Reason": "No assessment defined."
        })


leakage_df = pd.DataFrame(rows)


# ---------------------------------------------------------
# Save report
# ---------------------------------------------------------
leakage_df.to_csv(
    OUTPUT_FILE,
    index=False
)


# ---------------------------------------------------------
# Display summary
# ---------------------------------------------------------
print("\nPrediction-time availability assessment:")
print(
    leakage_df[
        ["Feature", "Availability_Assessment"]
    ].to_string(index=False)
)

print("\nAssessment counts:")
print(
    leakage_df["Availability_Assessment"]
    .value_counts()
)

print(f"\nSaved report to: {OUTPUT_FILE}")
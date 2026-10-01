import os
import pickle
import numpy as np
import pandas as pd
import joblib

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

MODEL_DIR = os.path.join(
    BASE_DIR,
    "models"
)

MODEL_PATH = os.path.join(
    MODEL_DIR,
    "final_xgboost_model.pkl"
)

ARTIFACT_PATH = os.path.join(
    MODEL_DIR,
    "preprocessing_artifacts.pkl"
)


# ============================================================
# LOAD MODEL
# ============================================================

model = joblib.load(
    MODEL_PATH
)


# ============================================================
# LOAD PREPROCESSING ARTIFACTS
# ============================================================

with open(
    ARTIFACT_PATH,
    "rb"
) as file:

    artifacts = pickle.load(file)


training_medians = artifacts[
    "training_medians"
]

frequency_mappings = artifacts[
    "frequency_mappings"
]

one_hot_columns = artifacts[
    "one_hot_columns"
]

final_feature_columns = artifacts[
    "final_feature_columns"
]

invalid_rules = artifacts[
    "invalid_rules"
]

numeric_indicator_features = artifacts[
    "numeric_indicator_features"
]

numeric_median_features = artifacts[
    "numeric_median_features"
]

frequency_features = artifacts[
    "frequency_features"
]

one_hot_features = artifacts[
    "one_hot_features"
]

exclude_features = artifacts[
    "exclude_features"
]

leakage_features = artifacts[
    "leakage_features"
]

unresolved_features = artifacts[
    "unresolved_features"
]


# ============================================================
# FASTAPI APPLICATION
# ============================================================

app = FastAPI(
    title="US Accidents Severity Prediction API",
    description=(
        "Backend API for predicting accident severity "
        "using the optimized XGBoost model."
    ),
    version="1.0.0"
)


# ============================================================
# CORS
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# INPUT SCHEMA
# ============================================================

class AccidentInput(BaseModel):

    # Basic accident information
    source: str = Field(..., min_length=1)
    street: str = Field(..., min_length=1)
    city: str = Field(..., min_length=1)
    county: str = Field(..., min_length=1)
    state: str = Field(..., min_length=1)
    zipcode: str = Field(..., min_length=1)

    # Location
    start_lat: float = Field(..., ge=-90, le=90)
    start_lng: float = Field(..., ge=-180, le=180)

    # Accident distance
    distance: float = Field(..., ge=0, le=200)

    # Time
    start_time: str = Field(..., min_length=1)

    # Weather
    temperature: float = Field(..., ge=-80, le=140)
    wind_chill: float | None = Field(
        default=None,
        ge=-100,
        le=150
    )

    humidity: float = Field(
        ...,
        ge=0,
        le=100
    )

    pressure: float = Field(
        ...,
        ge=20,
        le=35
    )

    visibility: float = Field(
        ...,
        ge=0,
        le=100
    )

    wind_speed: float | None = Field(
        default=None,
        ge=0,
        le=100
    )

    precipitation: float | None = Field(
        default=None,
        ge=0
    )

    wind_direction: str | None = None

    weather_condition: str | None = None

    # Other categorical information
    airport_code: str | None = None
    timezone: str | None = None

    sunrise_sunset: str | None = None
    civil_twilight: str | None = None
    nautical_twilight: str | None = None
    astronomical_twilight: str | None = None


# ============================================================
# ROOT ENDPOINT
# ============================================================

@app.get("/")
def root():

    return {
        "message":
            "US Accidents Severity Prediction API",
        "status":
            "running"
    }


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/health")
def health():

    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "feature_count":
            len(final_feature_columns)
    }


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def create_input_dataframe(
    data: AccidentInput
):

    values = data.model_dump()

    df = pd.DataFrame(
        [{
            "Source":
                values["source"],

            "Street":
                values["street"],

            "City":
                values["city"],

            "County":
                values["county"],

            "State":
                values["state"],

            "Zipcode":
                values["zipcode"],

            "Start_Lat":
                values["start_lat"],

            "Start_Lng":
                values["start_lng"],

            "Distance(mi)":
                values["distance"],

            "Start_Time":
                values["start_time"],

            "Temperature(F)":
                values["temperature"],

            "Wind_Chill(F)":
                values["wind_chill"],

            "Humidity(%)":
                values["humidity"],

            "Pressure(in)":
                values["pressure"],

            "Visibility(mi)":
                values["visibility"],

            "Wind_Speed(mph)":
                values["wind_speed"],

            "Precipitation(in)":
                values["precipitation"],

            "Wind_Direction":
                values["wind_direction"],

            "Weather_Condition":
                values["weather_condition"],

            "Airport_Code":
                values["airport_code"],

            "Timezone":
                values["timezone"],

            "Sunrise_Sunset":
                values["sunrise_sunset"],

            "Civil_Twilight":
                values["civil_twilight"],

            "Nautical_Twilight":
                values["nautical_twilight"],

            "Astronomical_Twilight":
                values["astronomical_twilight"]
        }]
    )

    return df


# ============================================================
# TEMPORAL FEATURES
# ============================================================

def create_temporal_features(df):

    start_time = pd.to_datetime(
        df["Start_Time"]
        .astype(str)
        .str.strip(),
        format="mixed",
        errors="coerce"
    )

    if start_time.isna().any():

        raise ValueError(
            "Invalid Start_Time value."
        )

    df["Start_Year"] = (
        start_time.dt.year
    )

    df["Start_Month"] = (
        start_time.dt.month
    )

    df["Start_Day"] = (
        start_time.dt.day
    )

    df["Start_Hour"] = (
        start_time.dt.hour
    )

    df["Start_DayOfWeek"] = (
        start_time.dt.dayofweek
    )

    df["Is_Weekend"] = (
        df["Start_DayOfWeek"] >= 5
    ).astype(int)

    df["Is_Night"] = (
        (df["Start_Hour"] < 6) |
        (df["Start_Hour"] >= 20)
    ).astype(int)

    df["Hour_Sin"] = np.sin(
        2 * np.pi *
        df["Start_Hour"] / 24
    )

    df["Hour_Cos"] = np.cos(
        2 * np.pi *
        df["Start_Hour"] / 24
    )

    df["Month_Sin"] = np.sin(
        2 * np.pi *
        (df["Start_Month"] - 1) / 12
    )

    df["Month_Cos"] = np.cos(
        2 * np.pi *
        (df["Start_Month"] - 1) / 12
    )

    return df


# ============================================================
# NUMERIC MISSING VALUE TREATMENT
# ============================================================

def apply_missing_value_treatment(df):

    # Missingness indicators
    for feature in numeric_indicator_features:

        if feature not in df.columns:
            continue

        indicator_name = (
            feature
            .replace("(", "_")
            .replace(")", "")
            .replace("/", "_")
            .replace(" ", "_")
            + "_Missing"
        )

        df[indicator_name] = (
            df[feature]
            .isna()
            .astype("int8")
        )


    # Numeric median imputation
    for feature in numeric_median_features:

        if feature not in df.columns:
            continue

        median_value = (
            training_medians[feature]
        )

        df[feature] = (
            pd.to_numeric(
                df[feature],
                errors="coerce"
            )
            .fillna(median_value)
        )


    # Categorical missing treatment
    categorical_columns = [
        "Weather_Condition",
        "Wind_Direction",
        "Airport_Code",
        "Street",
        "Timezone",
        "Zipcode",
        "City",
        "Sunrise_Sunset",
        "Civil_Twilight",
        "Nautical_Twilight",
        "Astronomical_Twilight"
    ]

    for feature in categorical_columns:

        if feature in df.columns:

            df[feature] = (
                df[feature]
                .astype("string")
                .fillna("Missing")
            )

    return df


# ============================================================
# INVALID VALUE TREATMENT
# ============================================================

def apply_invalid_value_treatment(df):

    for feature, rule in invalid_rules.items():

        if feature not in df.columns:
            continue

        minimum, maximum = rule

        values = pd.to_numeric(
            df[feature],
            errors="coerce"
        )

        invalid = pd.Series(
            False,
            index=df.index
        )

        if minimum is not None:

            invalid |= (
                values < minimum
            )

        if maximum is not None:

            invalid |= (
                values > maximum
            )

        df.loc[
            invalid,
            feature
        ] = np.nan

        df[feature] = (
            pd.to_numeric(
                df[feature],
                errors="coerce"
            )
            .fillna(
                training_medians.get(
                    feature,
                    0
                )
            )
        )

    return df


# ============================================================
# CATEGORICAL ENCODING
# ============================================================

def normalize_category(series):

    return (
        series
        .astype("string")
        .fillna("__MISSING__")
    )


def apply_categorical_encoding(df):

    # Frequency encoding
    for feature in frequency_features:

        if feature not in df.columns:
            continue

        mapping = (
            frequency_mappings[feature]
        )

        normalized = normalize_category(
            df[feature]
        )

        df[feature] = (
            normalized
            .map(mapping)
            .fillna(0.0)
        )


    # One-hot encoding
    for feature in one_hot_features:

        if feature not in df.columns:
            continue

        normalized = normalize_category(
            df[feature]
        )

        dummies = pd.get_dummies(
            normalized,
            prefix=feature,
            dtype=np.int8
        )

        expected_columns = (
            one_hot_columns[feature]
        )

        dummies = dummies.reindex(
            columns=expected_columns,
            fill_value=0
        )

        df = df.drop(
            columns=[feature]
        )

        df = pd.concat(
            [df, dummies],
            axis=1
        )

    # Exclude Country
    for feature in exclude_features:

        if feature in df.columns:

            df = df.drop(
                columns=[feature]
            )

    return df


# ============================================================
# NUMERICAL TRANSFORMATIONS
# ============================================================

def apply_numerical_transformations(df):

    distance = pd.to_numeric(
        df["Distance(mi)"],
        errors="coerce"
    )

    precipitation = pd.to_numeric(
        df["Precipitation(in)"],
        errors="coerce"
    )

    df["Distance_log1p"] = (
        np.log1p(distance)
    )

    df["Precipitation_log1p"] = (
        np.log1p(precipitation)
    )

    return df


# ============================================================
# FINAL FEATURE PREPARATION
# ============================================================

def prepare_features(data):

    df = create_input_dataframe(
        data
    )

    # Remove leakage/unresolved features
    df = df.drop(
        columns=[
            column
            for column in (
                leakage_features +
                unresolved_features
            )
            if column in df.columns
        ],
        errors="ignore"
    )

    # Missing values
    df = apply_missing_value_treatment(
        df
    )

    # Invalid values
    df = apply_invalid_value_treatment(
        df
    )

    # Temporal features
    df = create_temporal_features(
        df
    )

    # Categorical encoding
    df = apply_categorical_encoding(
        df
    )

    # Numerical transformations
    df = apply_numerical_transformations(
        df
    )

    # Remove raw timestamps
    df = df.drop(
        columns=[
            "Start_Time",
            "End_Time"
        ],
        errors="ignore"
    )

    # Remove final exclusions
    df = df.drop(
        columns=[
            "ID",
            "Turning_Loop"
        ],
        errors="ignore"
    )

    # Ensure every final model column exists
    for column in final_feature_columns:

        if column not in df.columns:

            df[column] = 0

    # Keep exactly the same feature order
    df = df[
        final_feature_columns
    ]

    # Convert everything to numeric
    df = df.apply(
        pd.to_numeric,
        errors="coerce"
    )

    # Final missing-value protection
    df = df.fillna(0)

    # Infinite-value protection
    df = df.replace(
        [np.inf, -np.inf],
        0
    )

    return df


# ============================================================
# SEVERITY DESCRIPTION
# ============================================================

def severity_description(
    severity
):

    descriptions = {

        1:
            "Minor impact accident",

        2:
            "Moderate impact accident",

        3:
            "Significant impact accident",

        4:
            "Severe impact accident"
    }

    return descriptions.get(
        severity,
        "Unknown severity"
    )


# ============================================================
# PREDICTION ENDPOINT
# ============================================================

@app.post("/predict")
def predict(
    accident: AccidentInput
):

    try:

        # Prepare features
        X = prepare_features(
            accident
        )

        # Predict
        prediction = model.predict(
            X
        )

        # Model was trained with labels 0-3,
        # while the original dataset uses 1-4.
        severity = int(
            prediction[0]
        ) + 1

        # Probability
        probabilities = model.predict_proba(
            X
        )[0]

        probability_dict = {
            str(index + 1):
                round(
                    float(probability),
                    4
                )
            for index, probability
            in enumerate(probabilities)
        }

        return {

            "success": True,

            "severity": severity,

            "description":
                severity_description(
                    severity
                ),

            "probabilities":
                probability_dict
        }

    except Exception as error:

        return {

            "success": False,

            "error":
                str(error)
        }
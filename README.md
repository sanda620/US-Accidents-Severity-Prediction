# US Accidents Severity Prediction

## Project Overview

This project develops a machine learning system to predict the severity of road accidents in the United States using accident, location, time, and weather information.

The project uses the US Accidents dataset covering 2016–2023. It includes data sampling, exploratory data analysis, preprocessing, model development, hyperparameter optimization, model evaluation, backend development, and frontend integration.

The final model is an optimized **XGBoost classifier** that predicts one of four accident severity classes.

## Project Objectives

- Analyze and understand the US Accidents dataset.
- Clean and preprocess the data for machine learning.
- Engineer useful temporal and numerical features.
- Train and compare multiple classification models.
- Optimize model hyperparameters.
- Evaluate the final model using appropriate classification metrics.
- Develop a backend API for accident severity predictions.
- Create a user-friendly frontend for entering accident information and viewing predictions.

## Dataset

The project uses the US Accidents dataset covering 2016–2023.

- Original dataset: approximately 7.7 million records.
- Stratified sample: 999,592 records.
- Training dataset: 799,673 records.
- Testing dataset: 199,919 records.
- Final model inputs: 138 features.
- Target variable: `Severity`.

### Severity Classes

| Severity | Description |
|---|---|
| 1 | Lowest severity class |
| 2 | Moderate severity class |
| 3 | High severity class |
| 4 | Highest severity class |

The severity labels follow the original dataset. The descriptions shown in the application are simplified explanations for users.

## Technologies Used

- **Python** — primary programming language
- **Pandas and NumPy** — data processing and feature engineering
- **Scikit-learn** — model evaluation and hyperparameter search
- **XGBoost** — final classification model
- **FastAPI** — backend API
- **Uvicorn** — API server
- **HTML, CSS, JavaScript** — frontend
- **Jupyter Notebook** — data analysis and model development
- **Git and GitHub** — version control

## Project Structure

```text
US-Accidents-Severity-Prediction/
│
├── data/
│   ├── train_final.csv
│   └── test_final.csv
│
├── notebooks/
│   ├── 01_sampling.ipynb
│   ├── 02_eda.ipynb
│   ├── 03_preprocessing.ipynb
│   ├── 04_model_development.ipynb
│   └── 05_model_optimization.ipynb
│
├── models/
│   ├── final_xgboost_model.pkl
│   └── preprocessing_artifacts.pkl
│
├── backend/
│   ├── app.py
│   └── build_artifacts.py
│
├── frontend/
│   ├── index.html
│   ├── style.css
│   └── script.js
│
├── requirements.txt
└── README.md
```

*Note: This structure describes the intended project layout. Update the file list if your repository uses different filenames or folders.*

## Methodology

### 1. Data Sampling and Exploratory Data Analysis

A stratified sample of 999,592 records was created from the original dataset to make analysis and model training more manageable. Sampling considered both year and accident severity.

Exploratory data analysis examined data types, missing values, duplicate records, constant columns, descriptive statistics, and target-class distribution.

The target variable is imbalanced, with Severity 2 representing the majority of records.

### 2. Data Preprocessing

Preprocessing included:

- Missing-value analysis and treatment.
- Invalid-value investigation and treatment.
- Duplicate-record checks.
- Categorical feature encoding.
- Temporal feature engineering.
- Numerical transformations.
- Final feature selection.
- Creation of training and testing datasets.

Temporal features include year, month, day, hour, day of the week, weekend and night indicators, and cyclical encodings for hour and month.

The final modeling datasets contain 138 predictor features and the target variable.

### 3. Model Development

Four classification algorithms were evaluated:

- Logistic Regression
- Decision Tree
- Random Forest
- XGBoost

Models were compared using accuracy, macro precision, macro recall, macro F1-score, and weighted F1-score.

Macro F1-score was particularly important because it gives equal importance to each severity class.

### 4. Hyperparameter Optimization

Randomized Search Cross-Validation was used to optimize Random Forest and XGBoost models.

The optimized XGBoost model achieved the highest validation Macro F1-score among the evaluated candidates and was selected as the final model.

### 5. Backend Development

The backend uses FastAPI to:

- Load the trained XGBoost model.
- Load preprocessing artifacts.
- Receive accident information through an API request.
- Validate input fields.
- Prepare model features.
- Generate a severity prediction.
- Return the predicted class and model probabilities.

### 6. Frontend Development

The frontend provides a form for entering accident location, time, distance, and weather information.

It sends the input to the backend API and displays the predicted severity and model probability estimates. The interface also provides basic input validation and error messages.

## Final Model Performance

The optimized XGBoost model was evaluated on the held-out test dataset.

| Metric | Test Result |
|---|---:|
| Accuracy | 0.898379 |
| Macro Precision | 0.786912 |
| Macro Recall | 0.631647 |
| Macro F1-score | 0.682952 |
| Weighted F1-score | 0.892291 |

The model achieved approximately **89.84% test accuracy** and a **0.6830 Macro F1-score**.

Although overall accuracy is high, performance differs between severity classes because of the imbalanced target distribution. In particular, the highest severity class is more difficult to identify reliably. The results should therefore be interpreted using class-specific metrics as well as overall accuracy.


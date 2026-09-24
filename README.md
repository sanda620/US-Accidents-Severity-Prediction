# US Accidents Severity Prediction

## Project Overview

This project develops a machine learning-based decision support system
for predicting traffic-impact severity using the US Accidents dataset.

The project focuses on understanding accident patterns, preprocessing
large-scale accident data, engineering meaningful features, selecting
relevant features, and developing machine learning classification models.

## Dataset

Dataset: US Accidents (2016–2023)

Original dataset source:
https://www.kaggle.com/datasets/sobhanmoosavi/us-accidents

The locally available dataset contains approximately 6.67 million
accident records.

Due to computational constraints and project requirements, a
stratified sample of approximately 1 million records was created
while preserving the Year × Severity distribution.

Working dataset:

- Records: 999,458
- Features: 46
- Period: 2016–2023
- Target: Severity
- Task: Multiclass classification

## Target Variable

`Severity` represents the traffic-impact severity level of an accident.

The project initially retains four classes:

- Severity 1
- Severity 2
- Severity 3
- Severity 4

## Project Workflow

1. Problem Understanding
2. Dataset Identification
3. Data Sampling
4. Data Understanding and EDA
5. Data Preprocessing
6. Feature Engineering
7. Feature Selection
8. Machine Learning Model Development
9. Model Evaluation
10. Hyperparameter Optimization
11. Prediction System Development
12. Testing and Documentation

## Project Status

Currently working on:

**Stage 3 – Data Understanding and Exploratory Data Analysis**
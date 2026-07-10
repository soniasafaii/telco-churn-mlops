# Telco Customer Churn Prediction - MLOps Project

## Overview

This project implements an end-to-end Machine Learning pipeline for predicting customer churn using the Telco Customer Churn dataset.

The main goal of this project is not only to train a classification model, but also to build a complete MLOps workflow including:

* Data versioning
* Data preprocessing
* Feature engineering
* Model training and evaluation
* Experiment tracking using MLflow
* Model registration
* Docker-based deployment preparation

---

# Project Objectives

Customer churn prediction is an important business problem in telecommunication companies.

The objective of this project is to build a machine learning system that predicts whether a customer is likely to leave the company based on customer information, service usage, and billing features.

The final system provides:

* A reproducible training pipeline
* Multiple model comparison
* Best model selection
* MLflow experiment management
* Dockerized model environment

---

# Project Architecture

```
Dataset
   |
   |
Data Loader
   |
   |
Data Preparation
   |
   |
Feature Engineering
   |
   |
Train / Validation / Test Split
   |
   |
Model Training
   |
   |
Model Evaluation
   |
   |
MLflow Tracking
   |
   |
Model Registry
   |
   |
Docker Deployment
```

---

# Dataset Versions

The project uses three different dataset versions.

## v1 - Raw Dataset

Location:

```
data/v1/
```

Contains the original Telco Customer Churn Excel file.

Processing steps:

* Removing unnecessary columns
* Converting data types
* Handling missing values
* Encoding categorical variables

---

## v2 - Preprocessed Dataset

Location:

```
data/v2/
```

This version contains:

* Cleaned data
* Converted numerical features
* Encoded categorical variables

The purpose of v2 is to evaluate the impact of preprocessing before feature engineering.

---

## v3 - Feature Engineered Dataset

Location:

```
data/v3/
```

This is the final dataset version used for modeling.

Additional features:

### Average Monthly Spend

```
Total Charges / (Tenure Months + 1)
```

### Total Services

Number of subscribed services:

* Phone Service
* Online Security
* Online Backup
* Device Protection
* Tech Support
* Streaming TV
* Streaming Movies

### Is New Customer

Identifies customers with less than or equal to 12 months tenure.

### High Monthly Charges

Binary feature showing customers with above-average monthly charges.

### Charge per Tenure

```
Monthly Charges / (Tenure Months + 1)
```

---

# Machine Learning Models

Four classification algorithms were implemented and compared:

## Logistic Regression

Used as a baseline linear model.

## Random Forest

An ensemble tree-based model.

## XGBoost

Gradient boosting model optimized for classification tasks.

## CatBoost

Gradient boosting algorithm designed for handling categorical data efficiently.

---

# Model Evaluation

The following metrics were used:

* Accuracy
* Precision
* Recall
* F1-score
* ROC-AUC

Model selection was performed based on F1-score because churn prediction is an imbalanced classification problem.

---

# Validation Strategy

The dataset was divided into:

```
Training Set      70%

Validation Set   15%

Test Set         15%
```

Stratified splitting was used to preserve churn class distribution.

Additionally, Stratified K-Fold Cross Validation with 5 folds was performed.

---

# Final Results

## Best Model

The best performing model was:

```
Logistic Regression
```

## Test Performance

| Metric    | Score |
| --------- | ----: |
| Accuracy  | 0.812 |
| Precision | 0.677 |
| Recall    | 0.566 |
| F1-score  | 0.616 |
| ROC-AUC   | 0.859 |

Confusion Matrix:

```
[[700  76]
 [122 159]]
```

---

# MLflow Integration

MLflow was used for experiment tracking and model management.

Tracked information:

## Parameters

Examples:

* Dataset version
* Model name
* Random state
* Model hyperparameters

## Metrics

Logged metrics:

* Accuracy
* Precision
* Recall
* F1-score
* ROC-AUC

## Artifacts

Stored artifacts:

* Confusion Matrix

## Model Registry

The final model was registered as:

```
TelcoChurnModel
```

The model corresponding to the final dataset version (v3) is stored in MLflow Model Registry.

---

# Project Structure

```
telco_churn_mlops/

│
├── data/
│   ├── v1/
│   ├── v2/
│   └── v3/
│
├── src/
│   │
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── features.py
│   ├── train.py
│   ├── evaluate.py
│   └── mlflow_logger.py
│
├── run_pipeline.py
│
├── requirements.txt
│
├── Dockerfile
│
└── README.md
```

---

# Running the Project

## 1. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 2. Run Training Pipeline

```bash
python run_pipeline.py
```

The pipeline will execute:

* v1 training
* v2 training
* v3 training

and log experiments into MLflow.

---

# MLflow UI

To open MLflow dashboard:

```bash
mlflow ui
```

Then open:

```
http://localhost:5000
```

The dashboard provides:

* Experiment comparison
* Metrics visualization
* Registered models

---

# Docker Support

The project includes Docker support for creating a reproducible environment.

Build image:

```bash
docker build -t telco-churn-mlops .
```

Run container:

```bash
docker run telco-churn-mlops
```

Docker guarantees that:

* Dependencies are fixed
* Environment is reproducible
* Pipeline runs independently from local machine configuration

---

# Technologies Used

## Programming Language

* Python

## Machine Learning

* Scikit-learn
* XGBoost
* CatBoost

## Data Processing

* Pandas
* NumPy

## Experiment Tracking

* MLflow

## Deployment

* Docker

---

# Future Improvements

Possible improvements:

* Create a FastAPI inference service
* Deploy the registered MLflow model
* Add CI/CD pipeline using GitHub Actions
* Add automated testing
* Add monitoring for model performance

---

# Author

Computer Engineering Student

Machine Learning & MLOps Project

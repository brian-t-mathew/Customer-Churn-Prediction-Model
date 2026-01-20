📉 Customer Churn Prediction (EDA + Bagging Ensemble + CLI Input)

This project builds an end-to-end Customer Churn Prediction system using Machine Learning. It includes full Exploratory Data Analysis (EDA), feature engineering (including sentiment score extraction from customer feedback), preprocessing using ColumnTransformer, handling imbalance using SMOTE, model comparison, and finally applying Bagging (Bootstrap Aggregation) on the best performing model selected by F1-score.

The final model supports churn prediction using direct user input from the command line (input()).

📂 Project Files

Churn_Prediction_EDA_Bagging_BestModel_CLI_Input.ipynb → Main notebook (complete corrected project)

churn.csv → Dataset

best_churn_model_bagged.pkl → Saved final trained model (generated after running the notebook)

🎯 Objective

To predict whether a customer is:

Active (not likely to churn)

Churn (likely to churn)

using structured customer data.

📌 Dataset Overview

Dataset: churn.csv

Rows: 36,992

Columns: 24

Target Column: churn_risk_score

Target Distribution

The target column contains two classes:

Class	Meaning	Count
0	Active / Not at churn risk	16,980
1	Churn / At churn risk	20,012
🔎 EDA Included

The notebook performs Exploratory Data Analysis such as:

✅ dataset shape and columns
✅ .info() and .describe()
✅ missing values percentage per column
✅ target distribution bar chart
✅ categorical value distributions (top categories)
✅ numeric feature histograms
✅ correlation heatmap for numeric features

⚙️ Feature Engineering

The project creates new meaningful features:

1) Date Features

From:

joining_date

last_visit_time

Creates:

days_since_joining

days_since_last_visit

2) Sentiment Score Feature

From:

feedback (text column)

Creates:

sentiment_score using NLTK VADER sentiment analysis

🧠 Preprocessing + ML Pipeline

The project uses a clean and industry-standard pipeline:

✅ Preprocessing

categorical encoding: OneHotEncoder(handle_unknown="ignore")

numeric columns passed as-is

✅ Handling Class Imbalance

SMOTE (Synthetic Minority Oversampling Technique)

✅ Feature Selection

SelectKBest(f_classif, k=25) selects top features

🤖 Models Trained & Compared

The notebook trains multiple models and selects the best one based on F1-score:

Logistic Regression

Decision Tree

Random Forest

XGBoost (XGBClassifier)

LightGBM (LGBMClassifier)

A model comparison plot is also generated.

🧩 Bagging on Best Model (Bootstrap Aggregation)

After selecting the best model, the project applies:

✅ BaggingClassifier on the best estimator

This improves model stability by training several bootstrapped versions of the best model and combining predictions.

Final saved model:

best_churn_model_bagged.pkl

🧪 Evaluation

Each model is evaluated using:

Accuracy

F1-score

Confusion Matrix

Classification Report

The final Bagged Best Model is evaluated using the same metrics.

💻 User Input (Command Line)

The notebook supports direct command-line style input using Python input():

Example:

Enter customer details below. Press Enter to use defaults.

Age (default: 25):
Gender (M/F) (default: M):
Region Category (default: Unknown):
...
✅ Prediction: Churn

🛠️ Tech Stack

Python

Pandas, NumPy

Matplotlib

Scikit-learn

imbalanced-learn (SMOTE)

XGBoost

LightGBM

NLTK (VADER Sentiment Analyzer)

Joblib (model saving)

▶️ How to Run
1) Install dependencies
pip install pandas numpy matplotlib scikit-learn imbalanced-learn xgboost lightgbm nltk joblib

2) Run notebook
jupyter notebook


Open:

Churn_Prediction_EDA_Bagging_BestModel_CLI_Input.ipynb

Run all cells.

📌 Business Impact

This churn prediction system helps businesses:

identify customers likely to churn

reduce churn and revenue loss

improve retention campaigns

prioritize high-risk users

👨‍💻 Author

Brian Mathew
Customer Churn Prediction Project

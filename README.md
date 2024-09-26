# Credit Card Fraud Detection Project

## Overview
This project is focused on detecting fraudulent transactions using machine learning models. The dataset used is the **Credit Card Fraud Detection dataset**, which is known for being highly imbalanced, with the vast majority of transactions being non-fraudulent. The goal of this project is to explore different models to detect fraudulent transactions and compare their performance, especially after rebalancing the dataset using the **SMOTE** (Synthetic Minority Over-sampling Technique).

## Dataset
The dataset contains transactions made by credit cards in September 2013 by European cardholders.
The dataset contains 284,807 transactions, with only 492 cases of fraud (Class 1), representing 0.17% of the data. The rest (Class 0) are non-fraudulent transactions. Given this imbalance, detecting fraud is particularly challenging.

- **Rows**: 284,807
- **Columns**: 31 (including the target variable `Class`)
- **Target Variable**: `Class` (1 for fraud, 0 for non-fraud)

The features are numerical and have been anonymized using PCA (Principal Component Analysis), except for the features `Time` and `Amount`.

The dataset can be found here : https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

## Steps

### 1. Data Exploration
The first step of the project consists of quickly exploring the dataset:
- Checking for data types.
- Visualizing the class imbalance to understand the challenge.

### 2. Model Training and Comparison
In this step, several machine learning models are trained and their performances are compared:
- **Logistic Regression**
- **Ridge Regression**
- **Lasso Regression**
- **XGBoost**
- **Random Forest**

Given the severe class imbalance, the initial models did not perform well in detecting fraud. Among the models, **XGBoost** and **Random Forest** achieved the best performance, even without addressing the class imbalance directly.

### 3. Synthetic Minority Over-sampling Technique (SMOTE) for Class Rebalancing and Retraining
To improve the models' ability to detect fraudulent transactions, I applied **SMOTE** to rebalance the classes by oversampling the minority class. SMOTE aims to balance the dataset by generating synthetic instances of the minority class rather than simply duplicating existing ones.
After applying SMOTE, the same models were trained again:
- **Logistic Regression**
- **Ridge Regression**
- **Lasso Regression**
- **XGBoost**
- **Random Forest**

This rebalancing improved the performance of the models, particularly the lasso regression model.

## Results
- **Before SMOTE**: the regression models, particularly the lasso model performance was hindered by the imbalanced dataset. XGBoost and Random Forest performed the best among the models.
- **After SMOTE**: XGBoost and Random Forest still achieved the best performance, with Balanced Accuracy of over 90%, meaning that they handle both fraudulent and non-fraudulent transactions well after the application of SMOTE. The lasso regression is the model that seems to have benefitted the most from applying SMOTE.

## Conclusion
This project demonstrates the importance of addressing class imbalance in fraud detection. While machine learning models like XGBoost and Random Forest are robust even with imbalanced data, applying techniques like SMOTE can improve model performance, making them more effective in detecting fraudulent transactions.

## Requirements
- **Libraries**: `tidyverse`, `skimr`, `pROC`, `caret`, `xgboost`, `randomForest`, `smotefamily`, `dplyr`,

# Credit-Card-Fraud-Detection
 This project detects fraudulent transactions using a Kaggle dataset. To address class imbalance, I applied downsampling and SMOTE oversampling techniques. I evaluated Logistic Regression, Random Forest, and Decision Tree classifiers on the original, downsampled, and oversampled datasets to improve fraud detection performance.


## Table of Contents

1. [Overview](#overview)
2. [Dataset](#dataset)
3. [Preprocessing](#preprocessing)
4. [Modeling](#modeling)
5. [Evaluation](#evaluation)
6. [Results](#results)
7. [Conclusion](#conclusion)

## Overview

This project aims to detect fraudulent credit card transactions using machine learning techniques. The dataset used is from Kaggle and contains various features related to credit card transactions, with the task of classifying each transaction as either fraudulent or legitimate.

## Dataset

The dataset contains anonymized features of credit card transactions. Due to confidentiality, the variables are hidden. The target variable (`Class`) represents the transaction type: `0` for normal transactions and `1` for fraudulent transactions.

## Preprocessing

The following steps were performed to prepare the dataset:

1. **Standardization**: The `Amount` column was standardized to improve model performance.
2. **Data Cleaning**: Duplicate rows were removed, and the `Time` column was dropped due to its irrelevance to the classification task.
3. **Class Distribution**: An imbalance between the classes (fraud vs. normal) was observed, leading to the application of class-balancing techniques.

## Modeling

To address the class imbalance, the following approaches were applied:

1. **Downsampling**: The normal class was downsampled to match the number of fraud transactions.
2. **Oversampling (SMOTE)**: Synthetic samples were generated for the fraud class to balance the dataset.

Three machine learning models were tested:

1. **Logistic Regression**
2. **Random Forest Classifier**
3. **Decision Tree Classifier**

Each classifier was trained on three different versions of the dataset: the original, downsampled, and oversampled datasets.

## Evaluation

The models were evaluated based on the following metrics:

- **Accuracy**: Proportion of correctly predicted transactions.
- **Precision**: The proportion of true positive fraud predictions out of all positive predictions.
- **Recall**: The proportion of actual fraudulent transactions correctly identified.
- **F1 Score**: The harmonic mean of precision and recall.

## Results

Each model was trained and tested on the original, downsampled, and oversampled datasets. The performance metrics for each model were recorded, and the model with the highest best metrics was selected as the best-performing model.

## Conclusion

This project demonstrates the application of machine learning for credit card fraud detection. Class imbalance was addressed using both downsampling and SMOTE oversampling techniques. The performance of multiple models was compared, and the best model was selected based on the evaluation metrics.

# Credit Card Fraud Detection

## Table of Contents

1. [Overview](#overview)
2. [Dataset](#dataset)
3. [Preprocessing](#preprocessing)
4. [Modeling](#modeling)
5. [Evaluation](#evaluation)
6. [Results](#results)
7. [Conclusion](#conclusion)
8. [Future Work](#future-work)

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

It’s worth noting that when trained on the original, imbalanced dataset, the models displayed a high accuracy but low precision, indicating that they were predominantly predicting the majority class (normal transactions). This demonstrates how class imbalance can create a false sense of performance, as the model may appear effective due to high accuracy but fail to identify the minority class (fraudulent transactions) adequately. Therefore, class-balancing techniques like SMOTE and downsampling were crucial to improving model performance, particularly for the detection of fraudulent transactions.

## Results

Each model was trained and tested on the original, downsampled, and oversampled datasets. The performance metrics for each model were recorded, and the model with the highest F1 score was selected as the best-performing model.

## Conclusion

This project demonstrates the application of machine learning for credit card fraud detection. One key observation from the results is that class imbalance can significantly affect model performance. On the original, imbalanced dataset, models showed relatively high accuracy, but this was misleading as precision was notably low. This suggests that the model was primarily predicting the majority class (normal transactions) and failing to detect the minority class (fraudulent transactions), which is the primary goal of the model. 

By addressing the class imbalance through techniques like downsampling and SMOTE oversampling, the models' ability to detect fraud was significantly improved. The performance of multiple models was compared, and the best model was selected based on the evaluation metrics.

This highlights the importance of addressing class imbalance in machine learning tasks, especially in fraud detection, where false positives (normal transactions identified as fraudulent) are much more costly than false negatives.


## Future Work

While this project provides a good foundation for detecting credit card fraud, there are several potential areas for improvement:

1. **Model Optimization**: Further tuning of hyperparameters using techniques like GridSearchCV or RandomizedSearchCV could improve model performance.
2. **Deep Learning Models**: Exploring deep learning models, such as neural networks, could yield better results, especially for large datasets.
3. **Ensemble Methods**: Combining multiple models using techniques like boosting or bagging (e.g., XGBoost, AdaBoost) could improve accuracy and robustness.
4. **Anomaly Detection Techniques**: Investigating unsupervised learning methods or anomaly detection algorithms could be helpful, especially when labels are limited.
5. **Real-Time Fraud Detection**: Implementing the model for real-time fraud detection in a production environment could be a valuable next step.
6. **Feature Engineering**: More advanced feature engineering and feature selection techniques could be explored to enhance model accuracy.

These areas offer opportunities to further improve fraud detection accuracy and adapt the solution for more complex, real-world scenarios.


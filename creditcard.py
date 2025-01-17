import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the dataset from a CSV file
try:
    df = pd.read_csv("C:\\Users\\anmit\\Documents\\Projects\\Python\\Credit Card Fraud Detection\\creditcard.csv")
    print("File loaded successfully!")
except FileNotFoundError as e:
    print(f"FileNotFoundError: {e}")
except Exception as e:
    print(f"An error occurred: {e}")

# Plot histograms for all the columns to understand their distribution
df.hist(df.columns, figsize=(20, 20), bins=40)
plt.show()

# Display summary statistics for the dataset
df.describe()

from sklearn.preprocessing import StandardScaler

# Standardize the 'Amount' column to normalize it for better modeling performance
df['Amount'] = StandardScaler().fit_transform(pd.DataFrame(df['Amount']))

# Drop the 'Time' column as it might not add value to the classification task
data = df.drop(['Time'], axis=1)
data.head()

# Check for and count duplicate rows in the dataset
data.duplicated().count()

# Remove duplicate rows to avoid redundant data
data.drop_duplicates(inplace=True)

# Count the occurrences of each class (0: Normal, 1: Fraud)
count_classes = pd.value_counts(df['Class'], sort=True).sort_index()
print(count_classes)

# Plot a bar chart to visualize the distribution of the target variable (Class)
count_classes.plot(kind='bar')
plt.title("Histogram for Class")
plt.xlabel("Class")
plt.ylabel("Frequency")
plt.show()

# Separate the data into normal (Class = 0) and fraud (Class = 1) transactions
normal = data[data['Class'] == 0]
fraud = data[data['Class'] == 1]

# Downsample the normal transactions to balance the dataset
normal_downsample = normal.sample(n=473)  # Same number as fraud samples
downsample_data = pd.concat([normal_downsample, fraud], ignore_index=True)

# Check the distribution of the downsampled data
downsample_data['Class'].value_counts()

# Separate features (X) and target (y) for the downsampled data
X_down = downsample_data.drop('Class', axis=1)
y_down = downsample_data['Class']

# Separate features (X) and target (y) for the original data
X = data.drop('Class', axis=1)
y = data['Class']

# Use SMOTE (Synthetic Minority Oversampling Technique) to balance the dataset
from imblearn.over_sampling import SMOTE
X_over, y_over = SMOTE().fit_resample(X, y)

# Verify the balanced class distribution after SMOTE
y_over.value_counts()

# Split the downsampled data into training and testing sets
from sklearn.model_selection import train_test_split
X_train_down, X_test_down, y_train_down, y_test_down = train_test_split(
    X_down, y_down, test_size=0.2, random_state=42
)

# Split the SMOTE-oversampled data into training and testing sets
X_train_over, X_test_over, y_train_over, y_test_over = train_test_split(
    X_over, y_over, test_size=0.2, random_state=42
)

# Split the original data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Import classifiers and performance metrics
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

# Define a dictionary of classifiers to test
classifier = {
    "Logistic Regression": LogisticRegression(),
    "RandomForestClassifier": RandomForestClassifier(),
    "Decision Tree Classifier": DecisionTreeClassifier()
}

# Iterate through each classifier and evaluate performance
for name, clf in classifier.items():
    print(f"\n=========={name}===========")
    
    # Train the classifier on SMOTE-oversampled data
    clf.fit(X_train_over, y_train_over)
    
    # Make predictions on the test set
    y_pred_over = clf.predict(X_test_over)
    
    # Evaluate performance metrics on SMOTE-oversampled test data
    print(f"\n Accuracy (Over): {accuracy_score(y_test_over, y_pred_over)}")
    print(f"\n Precision (Over): {precision_score(y_test_over, y_pred_over)}")
    print(f"\n Recall (Over): {recall_score(y_test_over, y_pred_over)}")
    print(f"\n F1 Score (Over): {f1_score(y_test_over, y_pred_over)}")
    
    # Evaluate performance metrics on the downsampled test data
    print(f"\n==========Downsampled===========")
    y_pred_down = clf.predict(X_test_down)
    print(f"\n Accuracy (Downsampled): {accuracy_score(y_test_down, y_pred_down)}")
    print(f"\n Precision (Downsampled): {precision_score(y_test_down, y_pred_down)}")
    print(f"\n Recall (Downsampled): {recall_score(y_test_down, y_pred_down)}")
    print(f"\n F1 Score (Downsampled): {f1_score(y_test_down, y_pred_down)}")
    
    # Evaluate performance metrics on the original data test set
    print(f"\n==========Original Data===========")
    y_pred_orig = clf.predict(X_test)
    print(f"\n Accuracy (Original): {accuracy_score(y_test, y_pred_orig)}")
    print(f"\n Precision (Original): {precision_score(y_test, y_pred_orig)}")
    print(f"\n Recall (Original): {recall_score(y_test, y_pred_orig)}")
    print(f"\n F1 Score (Original): {f1_score(y_test, y_pred_orig)}")



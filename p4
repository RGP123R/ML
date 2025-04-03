import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
import matplotlib.pyplot as plt

# Load the dataset from an Excel file
df = pd.read_excel("irisdataset.xlsx")

# Print first few rows to ensure data is loaded correctly
print(df.head())

# Assuming the last column contains target labels and others are features
X = df.iloc[:, :-1].values  # Features
y = df.iloc[:, -1].values   # Target labels

# Encode categorical labels if they are not numeric
if isinstance(y[0], str):
    encoder = LabelEncoder()
    y = encoder.fit_transform(y)

# Split the dataset into training and testing sets (70% train, 30% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# Standardize the features (important for distance-based classifiers like k-NN)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Function to evaluate k-NN with different k values
def evaluate_knn(X_train, X_test, y_train, y_test, k_values):
    for k in k_values:
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train, y_train)
        y_pred = knn.predict(X_test)

        # Calculate accuracy and F1-score
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')

        print(f"k = {k}:")
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  F1-score: {f1:.4f}")
        print("-" * 30)

# Test with k values 1, 3, 5
evaluate_knn(X_train, X_test, y_train, y_test, k_values=[1, 3, 5])

# Function to evaluate weighted k-NN (distance-based)
def evaluate_weighted_knn(X_train, X_test, y_train, y_test, k_values):
    for k in k_values:
        knn_weighted = KNeighborsClassifier(n_neighbors=k, weights='distance')
        knn_weighted.fit(X_train, y_train)
        y_pred = knn_weighted.predict(X_test)

        # Calculate accuracy and F1-score
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')

        print(f"Weighted k = {k}:")
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  F1-score: {f1:.4f}")
        print("-" * 30)

# Test with k values 1, 3, 5 for weighted k-NN
evaluate_weighted_knn(X_train, X_test, y_train, y_test, k_values=[1, 3, 5])
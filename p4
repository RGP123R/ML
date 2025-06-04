import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_excel("irisdataset.xlsx")

# Separate features and target variable
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# Standardize the feature values
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

def evaluate_knn(X_train, X_test, y_train, y_test, k_values):
    """Evaluate KNN classifier with different values of k."""
    for k in k_values:
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train, y_train)
        y_pred = knn.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')

        print(f"k = {k}:")
        print(f" Accuracy: {accuracy:.4f}")
        print(f" F1-score: {f1:.4f}")
        print("-" * 30)

# Evaluate standard KNN
evaluate_knn(X_train, X_test, y_train, y_test, k_values=[1, 3, 5])

def evaluate_weighted_knn(X_train, X_test, y_train, y_test, k_values):
    """Evaluate weighted KNN classifier with different values of k."""
    for k in k_values:
        knn_weighted = KNeighborsClassifier(n_neighbors=k, weights='distance')
        knn_weighted.fit(X_train, y_train)
        y_pred = knn_weighted.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')

        print(f"Weighted k = {k}:")
        print(f" Accuracy: {accuracy:.4f}")
        print(f" F1-score: {f1:.4f}")
        print("-" * 30)

# Evaluate weighted KNN
evaluate_weighted_knn(X_train, X_test, y_train, y_test, k_values=[1, 3, 5])

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load the Boston Housing Dataset
boston_df = pd.read_csv("BostonHousing.csv")
X_boston = boston_df[['rm']]  # Adjust if your dataset has a different column name
y_boston = boston_df['medv']   # Ensure this matches the target column name in your dataset

X_train, X_test, y_train, y_test = train_test_split(X_boston, y_boston, test_size=0.2, random_state=42)

# Train Linear Regression model
linear_reg = LinearRegression()
linear_reg.fit(X_train, y_train)
y_pred = linear_reg.predict(X_test)

# Evaluate model
print("Linear Regression - Custom Boston Housing Dataset")
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))

# Visualization
plt.scatter(X_test, y_test, color='blue', label='Actual')
plt.plot(X_test, y_pred, color='red', linewidth=2, label='Predicted')
plt.xlabel("Number of Rooms (RM)")
plt.ylabel("Median House Price (MEDV)")
plt.title("Linear Regression on Custom Boston Housing Data")
plt.legend()
plt.show()

# Linear Regression for House Price Prediction

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error, r2_score

# Step 1: Create a sample dataset (can be replaced with a real one)
data = pd.read_csv('Housing.csv')

df = pd.DataFrame(data)
print(df)

# Step 2: Features and Target
X = df[['area', 'bedrooms', 'stories']]
y = df['price']

# Step 3: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10 ,random_state=42)
print(X_train)
print()
print(X_test)


# # Step 4: Train Linear Regression Model
model = LinearRegression()
model.fit(X_train, y_train)
  
# Step 5: Predict and Evaluate
y_pred = model.predict(X_test)
print("Predictions:", y_pred)
print("Actual:", y_test.values)

# Step 6: Evaluation Metrics
print(f"\nMean Squared Error: {root_mean_squared_error(y_test, y_pred):.2f}")
print(f"R^2 Score: {r2_score(y_test, y_pred):.2f}")

# Step 7: Display model coefficients
print("\nModel Coefficients:")
print(f"Intercept: {model.intercept_:.2f}")
print(f"Area Coefficient: {model.coef_[0]:.2f}")
print(f"Bedrooms Coefficient: {model.coef_[1]:.2f}")
print(f"Age Coefficient: {model.coef_[2]:.2f}")

# Step 8: Visualize predicted vs actual
plt.scatter(y_test, y_pred, color='blue')
plt.plot([y.min(), y.max()], [y.min(), y.max()], '--', color='red')
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted House Prices")
plt.grid(True)
plt.show()

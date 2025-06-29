import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Step 1: Prepare data
years = np.array([2010, 2012, 2014, 2016, 2018, 2020, 2022]).reshape(-1, 1)
prices = np.array([20, 25, 30, 36, 45, 53, 60])  # Prices in Lakhs

# Step 2: Train model
model = LinearRegression()
model.fit(years, prices)

# Step 3: Predict future price
future_year = np.array([[2024]])  # change this to any year
predicted_price = model.predict(future_year)

print(f"Predicted price in {future_year[0][0]}: ₹{predicted_price[0]:.2f} Lakhs")

# Step 4: Visualize
plt.scatter(years, prices, color='blue', label='Actual Prices')
plt.plot(years, model.predict(years), color='red', label='Regression Line')
plt.scatter(future_year, predicted_price, color='green', s=100, label='Predicted')
plt.xlabel("Year")
plt.ylabel("Price (Lakhs)")
plt.title("House Price Trend Over Years")
plt.legend()
plt.grid(True)
plt.show()

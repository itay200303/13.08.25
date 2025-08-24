import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

X1 = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]).reshape(-1, 1)
y1 = np.array([3.1, 5.0, 7.2, 8.9, 11.2, 12.8, 15.2, 17.1, 19.2, 21.0])

X2 = np.array([-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5]).reshape(-1, 1)
y2 = np.array([-20.5, -15.2, -7.6, -0.8, 3.2, 2.1, 4.7, 6.0, 4.9, 1.6, -3.8])

def regression_metrics(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    return mse, mae, rmse, r2

def adjusted_r2(r2, n, k):
    return 1 - (1 - r2) * (n - 1) / (n - k - 1)

lin_model = LinearRegression()
lin_model.fit(X1, y1)
y1_pred = lin_model.predict(X1)

mse1, mae1, rmse1, r2_1 = regression_metrics(y1, y1_pred)

X2_train, X2_test, y2_train, y2_test = train_test_split(X2, y2, test_size=0.2, random_state=42)

poly = PolynomialFeatures(degree=2)
X2_train_poly = poly.fit_transform(X2_train)
X2_test_poly = poly.transform(X2_test)

poly_model = LinearRegression()
poly_model.fit(X2_train_poly, y2_train)

y2_train_pred = poly_model.predict(X2_train_poly)
y2_test_pred = poly_model.predict(X2_test_poly)

mse2_train, mae2_train, rmse2_train, r2_train = regression_metrics(y2_train, y2_train_pred)
mse2_test, mae2_test, rmse2_test, r2_test = regression_metrics(y2_test, y2_test_pred)
adj_r2_train = adjusted_r2(r2_train, len(y2_train), X2_train_poly.shape[1] - 1)
adj_r2_test = adjusted_r2(r2_test, len(y2_test), X2_test_poly.shape[1] - 1)

a, b, c = poly_model.coef_[2], poly_model.coef_[1], poly_model.intercept_
optimal_x = -b / (2 * a)
optimal_y = poly_model.predict(poly.transform([[optimal_x]]))[0]


plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.scatter(X1, y1, color='blue', label='Actual')
plt.plot(X1, y1_pred, color='red', label='Linear Fit')
plt.title('Dataset 1: Gym Hours vs Muscle Gain')
plt.xlabel('Hours/week')
plt.ylabel('Muscle Gain (kg)')
plt.legend()
plt.grid(True)

X2_range = np.linspace(X2.min(), X2.max(), 200).reshape(-1, 1)
X2_range_poly = poly.transform(X2_range)
y2_range_pred = poly_model.predict(X2_range_poly)

plt.subplot(1, 2, 2)
plt.scatter(X2, y2, color='green', label='Actual')
plt.plot(X2_range, y2_range_pred, color='orange', label='Polynomial Fit')
plt.scatter(optimal_x, optimal_y, color='red', s=100, label='Vertex (Optimal Point)')
plt.title('Dataset 2: Load vs Strength Score')
plt.xlabel('Load (kg)')
plt.ylabel('Strength Score')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

print("----- Dataset 1 (Linear Regression) -----")
print(f"MSE: {mse1:.3f}, MAE: {mae1:.3f}, RMSE: {rmse1:.3f}, R²: {r2_1:.3f}")
print(f"Model: y = {lin_model.intercept_:.2f} + {lin_model.coef_[0]:.2f}x")

print("----- Dataset 2 (Polynomial Regression Degree 2) -----")
print("Train Set:")
print(f"MSE: {mse2_train:.3f}, MAE: {mae2_train:.3f}, RMSE: {rmse2_train:.3f}, R²: {r2_train:.3f}, Adjusted R²: {adj_r2_train:.3f}")
print("Test Set:")
print(f"MSE: {mse2_test:.3f}, MAE: {mae2_test:.3f}, RMSE: {rmse2_test:.3f}, R²: {r2_test:.3f}, Adjusted R²: {adj_r2_test:.3f}")

print(f"Optimal Load (Vertex) for Max Strength: {optimal_x:.2f} kg")
print(f"Max Predicted Strength Score: {optimal_y:.2f}")
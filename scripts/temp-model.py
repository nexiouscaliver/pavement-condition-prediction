import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

# Load the dataset
df_2021 = pd.DataFrame({
    'Section': list(range(1, 38)),  # Assuming 37 sections
    'Rutting (mm)': [3.0, 6.0, 5.8, 8.8, 4.5, 7.0, 5.2, 6.9, 8.3, 7.1, 6.4, 5.9, 7.5, 4.6, 5.7, 8.0, 6.8, 7.9, 5.4, 7.2, 6.6, 4.9, 6.3, 7.0, 6.7, 7.3, 5.1, 6.2, 7.4, 5.3, 7.7, 5.6, 6.1, 7.2, 5.9, 7.4, 6.5],
    'Fatigue Cracking (m²)': [2.0, 0.0, 2.5, 4.0, 1.0, 3.4, 1.8, 3.0, 2.2, 1.6, 2.3, 3.5, 2.1, 1.2, 2.5, 2.8, 2.4, 3.1, 2.6, 2.7, 3.2, 1.9, 2.5, 2.4, 3.0, 2.2, 2.8, 2.0, 3.4, 1.7, 2.9, 2.3, 2.6, 3.0, 2.2, 3.1, 2.7],
    'Block Cracking (m²)': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.5, 0.3, 0.2, 0.4, 0.6, 0.8, 0.1, 0.5, 0.3, 0.7, 0.6, 0.9, 0.4, 0.0, 0.2, 0.3, 0.1, 0.0, 0.0, 0.4, 0.2, 0.0, 0.1, 0.3, 0.4, 0.2, 0.1, 0.0, 0.0, 0.0],
    'Longitudinal Cracking (m²)': [18.0, 23.0, 16.0, 47.5, 25.0, 22.0, 28.0, 21.0, 33.0, 27.0, 24.0, 29.0, 26.0, 18.0, 31.0, 23.0, 28.0, 30.0, 24.0, 22.0, 27.0, 25.0, 29.0, 26.0, 21.0, 28.0, 23.0, 31.0, 22.0, 27.0, 29.0, 24.0, 28.0, 25.0, 30.0, 29.0, 27.0],
    'Transverse Cracking (m²)': [0.0, 2.5, 8.0, 10.0, 0.0, 5.4, 8.3, 6.2, 7.4, 5.9, 7.1, 4.2, 5.8, 3.6, 7.2, 4.8, 6.7, 7.9, 5.6, 4.9, 3.8, 7.4, 6.9, 5.2, 7.5, 4.4, 6.3, 5.7, 3.9, 6.1, 7.3, 5.4, 7.6, 6.2, 7.8, 4.7, 6.9],
    'Patching (m²)': [7.0, 18.0, 7.0, 36.0, 18.0, 15.4, 17.8, 14.5, 16.7, 13.9, 17.2, 15.8, 16.9, 14.4, 19.7, 18.3, 16.6, 19.8, 17.4, 18.2, 15.7, 16.9, 17.8, 18.3, 16.7, 15.5, 18.4, 17.2, 16.3, 19.5, 14.6, 15.4, 18.7, 16.5, 17.9, 19.1, 18.0],
    'Potholes (number)': [16, 0, 35, 0, 5, 32, 45, 30, 28, 36, 29, 32, 34, 27, 33, 29, 37, 30, 35, 28, 25, 34, 29, 26, 32, 31, 28, 33, 30, 36, 34, 29, 28, 32, 35, 30, 28],
    'Delamination (m²)': [2.0, 4.0, 2.0, 0.0, 1.0, 3.4, 4.2, 3.0, 4.8, 2.9, 4.1, 3.7, 4.5, 3.8, 2.7, 4.3, 3.9, 4.6, 2.8, 3.4, 4.7, 2.5, 3.6, 4.8, 3.0, 4.2, 3.7, 2.8, 4.5, 3.3, 4.1, 2.9, 4.6, 3.5, 4.2, 3.4, 4.0],
    'PCI (%)': [43.6, 51.0, 41.0, 25.0, 45.0, 50.0, 55.0, 39.0, 46.0, 48.0, 52.0, 50.0, 49.0, 43.0, 45.0, 41.0, 44.0, 42.0, 47.0, 50.0, 53.0, 39.0, 40.0, 48.0, 47.0, 52.0, 48.0, 51.0, 49.0, 50.0, 45.0, 46.0, 47.0, 49.0, 50.0, 44.0, 48.0]
})

# Features and target variable
X = df_2021.drop(columns=['PCI (%)', 'Section'])  # Drop PCI and Section columns
y = df_2021['PCI (%)']  # Target is PCI (%)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a Random Forest Regressor model
model = RandomForestRegressor(n_estimators=100, random_state=42)

# Train the model
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f"Root Mean Squared Error (RMSE): {rmse}")
print(f"R^2 Score: {r2}")

# Display feature importance
feature_importances = model.feature_importances_
features = X.columns
for feature, importance in zip(features, feature_importances):
    print(f"Feature: {feature}, Importance: {importance}")

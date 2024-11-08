import pandas as pd
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error
import numpy as np
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

# Load the test data
test_data = pd.read_csv(r'../datasets/htest.csv')
y_test = pd.read_csv(r'../datasets/hy_test.csv')

# Load the CatBoost model
model = CatBoostRegressor()
model.load_model(r'../models/trialmodels/best_catboost.cbm')

# Make predictions
# testdata(Rutting (mm),Fatigue_Cracking (m²),Block_Cracking (m²),Longitudinal_Cracking (m²),Transverse_Cracking (m²),Patching (m²),Potholes (Number),Delamination (m²),Severity_Rating,Traffic_Volume (vehicles/day),Temperature_C,Precipitation_mm,Maintenance_History
#          0.661290322580645,0.33333333333333337,0.6,0.427272727272727,0.9433962264150944,0.029629629629629617,0.1428571428571428,0.6999999999999997,0.5,0.15384615384615383,0.4130434782608695,0.21999999999999997,1.0)

#make test data pd dataframe
#test_data_input = pd.DataFrame([[0.661290322580645,0.33333333333333337,0.6,0.427272727272727,0.9433962264150944,0.029629629629629617,0.1428571428571428,0.6999999999999997,0.5,0.15384615384615383,0.4130434782608695,0.21999999999999997,1.0]], columns=test_data.columns)
#test_data_input.to_csv(r'../datasets/test_data_input.csv', index=False)

# Ask for input from user for each field
input_data = []
fields = test_data.columns
for field in fields:
    value = float(input(f"Enter value for {field}: "))
    input_data.append(value)

# Convert input data to DataFrame
input_df = pd.DataFrame([input_data], columns=fields)

label_encoder = LabelEncoder()
label_encoder.fit(["Low", "Medium", "High"])
input_df['Severity_Rating'] = label_encoder.fit_transform(input_df['Severity_Rating'])
label_encoder2 = LabelEncoder()
label_encoder2.fit(["None", "Minor repairs", "Major repairs"])
input_df['Maintenance_History'] = label_encoder2.fit_transform(input_df['Maintenance_History'])

scaler = MinMaxScaler()
numeric_columns = ['Rutting (mm)', 'Fatigue_Cracking (m²)', 'Block_Cracking (m²)', 
                   'Longitudinal_Cracking (m²)', 'Transverse_Cracking (m²)', 
                   'Patching (m²)', 'Potholes (Number)', 'Delamination (m²)', 
                   'Traffic_Volume (vehicles/day)', 'Temperature_C', 'Precipitation_mm','Severity_Rating','Maintenance_History']

# input_df[numeric_columns] = scaler.fit_transform(input_df[numeric_columns])

print(input_df)
input_df.to_csv(r'../datasets/input_df.csv', index=False)

# Make prediction
# prediction = model.predict(input_df)
# print(f'Prediction: {prediction[0]}')

# predictions = model.predict(test_data)

# Evaluate the model
# mse = mean_squared_error(y_test, predictions)
# print(f'Mean Squared Error: {mse}')
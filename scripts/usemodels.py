import pandas as pd
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error
import numpy as np
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import joblib
import xgboost as xgb

#Rutting (mm),Fatigue_Cracking (m²),Block_Cracking (m²),Longitudinal_Cracking (m²),Transverse_Cracking (m²),Patching (m²),Potholes (Number),Delamination (m²),Severity_Rating,Traffic_Volume (vehicles/day),Temperature_C,Precipitation_mm,Maintenance_History

orignaldb = pd.read_csv(r'../datasets/htrain.csv')   
orignaly = pd.read_csv(r'../datasets/hy_train.csv')   

rmax = orignaldb.max()
rmin = orignaldb.min()

range_data = pd.DataFrame([rmax, rmin])
ogdf = pd.read_csv(r'../datasets/pci_data_50.csv')
ogy = ogdf['PCI (%)']
ogdf = ogdf.drop(columns=['PCI (%)']) 
# rmax = ogdf.max()
# rmin = ogdf.min()
# range_data = pd.DataFrame([rmax, rmin])
ymax = ogy.max()
ymin = ogy.min()
yrange = pd.DataFrame([ymax, ymin])

def predict_all_models(Rutting ,Fatigue_Cracking ,Block_Cracking ,Longitudinal_Cracking ,Transverse_Cracking ,Patching ,Potholes ,Delamination ,Severity_Rating,Traffic_Volume ,Temperature_C,Precipitation_mm,Maintenance_History):
    # Load the data
    input_df = pd.DataFrame({
        'Rutting (mm)': [Rutting],
        'Fatigue_Cracking (m²)': [Fatigue_Cracking],
        'Block_Cracking (m²)': [Block_Cracking],
        'Longitudinal_Cracking (m²)': [Longitudinal_Cracking],
        'Transverse_Cracking (m²)': [Transverse_Cracking],
        'Patching (m²)': [Patching],
        'Potholes (Number)': [Potholes],
        'Delamination (m²)': [Delamination],
        'Severity_Rating': [Severity_Rating],
        'Traffic_Volume (vehicles/day)': [Traffic_Volume],
        'Temperature_C': [Temperature_C],
        'Precipitation_mm': [Precipitation_mm],
        'Maintenance_History': [Maintenance_History]
    })

    # Load the models
    model_catboost = CatBoostRegressor()
    model_catboost.load_model(r"../models/trialmodels/best_catboost.cbm")

    model_dnn = load_model(r"../models/trialmodels/model_dnn.keras")
    
    model_lgbm = joblib.load(r"../models/trialmodels/best_lgbm.pkl")

    model_xgb = xgb.Booster()
    model_xgb.load_model(r"../models/trialmodels/best_xgb_model.json")

    model_tf_1 = load_model(r"../models/trialmodels/model1.keras")

    model_tf_2 = load_model(r"../models/trialmodels/model2.keras")

    model_tf_3 = load_model(r"../models/trialmodels/model3.keras")

    model_tf_4 = load_model(r"../models/trialmodels/model4.keras")

    # preprocess the input

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

    # print(range_data[numeric_columns])
    scaler.fit(range_data[numeric_columns])

    # print(scaler.transform(input_df[numeric_columns]))
    input_df[numeric_columns] = scaler.transform(input_df[numeric_columns])
    
    #remove header
    # input_df_headless = input_df.values
    input_df_headless = input_df.values
    print('input_df_headless = ',input_df_headless)

    # Predict using the models
    pred_catboost = model_catboost.predict(input_df)
    pred_dnn = model_dnn.predict(input_df_headless)
    pred_lgbm = model_lgbm.predict(input_df)
    pred_xgb = model_xgb.predict(xgb.DMatrix(input_df))
    pred_tf_1 = model_tf_1.predict(input_df_headless)
    pred_tf_2 = model_tf_2.predict(input_df_headless)
    pred_tf_3 = model_tf_3.predict(input_df_headless)
    pred_tf_4 = model_tf_4.predict(input_df_headless)

    print('Catboost:', pred_catboost)
    print('DNN:', pred_dnn)
    print('LGBM:', pred_lgbm)
    print('XGB:', pred_xgb)
    print('TF1:', pred_tf_1)
    print('TF2:', pred_tf_2)
    print('TF3:', pred_tf_3)
    print('TF4:', pred_tf_4)


    # pred_catboost = pred_catboost * (ymax - ymin) + ymin
    # pred_dnn = pred_dnn * (ymax - ymin) + ymin
    # pred_lgbm = pred_lgbm * (ymax - ymin) + ymin
    # pred_xgb = pred_xgb * (ymax - ymin) + ymin
    # pred_tf_1 = pred_tf_1 * (ymax - ymin) + ymin
    # pred_tf_2 = pred_tf_2 * (ymax - ymin) + ymin
    # pred_tf_3 = pred_tf_3 * (ymax - ymin) + ymin
    # pred_tf_4 = pred_tf_4 * (ymax - ymin) + ymin

    # Unscale the predictions
    scaler2 = MinMaxScaler()
    scaler2.fit([[ymin], [ymax]])
    pred_catboost = scaler2.inverse_transform([pred_catboost])
    pred_dnn = scaler2.inverse_transform(pred_dnn)
    pred_lgbm = scaler2.inverse_transform([pred_lgbm])
    pred_xgb = scaler2.inverse_transform([pred_xgb])
    pred_tf_1 = scaler2.inverse_transform(pred_tf_1)
    pred_tf_2 = scaler2.inverse_transform(pred_tf_2)
    pred_tf_3 = scaler2.inverse_transform(pred_tf_3)
    pred_tf_4 = scaler2.inverse_transform(pred_tf_4)


    return pred_catboost[0], pred_dnn[0][0], pred_lgbm[0], pred_xgb[0], pred_tf_1[0][0], pred_tf_2[0][0], pred_tf_3[0][0], pred_tf_4[0][0]

# test data

ans = predict_all_models(3.9, 1.7, 0.3, 18.2, 3.5, 50.0, 6, 13.5, 'Low', 8200, 10.1, 102.0, 'Major repairs')
print('FINAL PREDICTIONS :')
print('Catboost:', ans[0])
print('DNN:', ans[1])
print('LGBM:', ans[2])
print('XGB:', ans[3])
print('TF1:', ans[4])
print('TF2:', ans[5])
print('TF3:', ans[6])
print('TF4:', ans[7])
print('Actual:', 39.0)

#make a plot
# Plot the predictions
models = ['Catboost', 'DNN', 'LGBM', 'XGB', 'TF1', 'TF2', 'TF3', 'TF4']
predictions = [ans[0][0], ans[1], ans[2][0], ans[3][0], ans[4], ans[5], ans[6], ans[7]]
actual_value = 39.0

plt.figure(figsize=(10, 6))
plt.plot(models, predictions, marker='o', label='Predictions')
plt.axhline(y=actual_value, color='r', linestyle='--', label='Actual Value')
plt.xlabel('Models')
plt.ylabel('PCI (%)')
plt.title('Model Predictions vs Actual Value')
plt.legend()
plt.grid(True)
plt.show()
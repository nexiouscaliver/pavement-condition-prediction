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
import base64

#Rutting (mm),Fatigue_Cracking (m²),Block_Cracking (m²),Longitudinal_Cracking (m²),Transverse_Cracking (m²),Patching (m²),Potholes (Number),Delamination (m²),Severity_Rating,Traffic_Volume (vehicles/day),Temperature_C,Precipitation_mm,Maintenance_History
orignaldb = pd.read_csv(r'../datasets/htrain.csv')   
orignaly = pd.read_csv(r'../datasets/hy_train.csv')   

rmax = orignaldb.max()
rmin = orignaldb.min()

range_data = pd.DataFrame([rmax, rmin])

# ogdf = pd.read_csv(r'../datasets/pci_data_50.csv')
# ogy = ogdf['PCI (%)']
# ogdf = ogdf.drop(columns=['PCI (%)']) 
# rmax = ogdf.max()
# rmin = ogdf.min()
# range_data = pd.DataFrame([rmax, rmin])

# ymax = ogy.max()
# ymin = ogy.min()
# yrange = pd.DataFrame([ymax, ymin])

df = pd.read_csv(r'../datasets/pci_data_50.csv')
y = df['PCI (%)']
df = df.drop(columns=['PCI (%)'])
label_encoder = LabelEncoder()
label_encoder.fit(["Low", "Medium", "High"])
df['Severity_Rating'] = label_encoder.fit_transform(df['Severity_Rating'])
label_encoder2 = LabelEncoder()
label_encoder2.fit(["None", "Minor repairs", "Major repairs"])
df['Maintenance_History'] = label_encoder2.fit_transform(df['Maintenance_History'])

rmin1 = df.min()
rmax1 = df.max()
ymax = y.max()
ymin = y.min()
yrange = pd.DataFrame([ymax, ymin])
range_data2 = pd.DataFrame([rmax1, rmin1])


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

    rf_model = joblib.load(r'../models/trialmodels/random_forest_model.pkl')

    stacking_regressor = joblib.load(r'../models/trialmodels/stacking_regressor_model.pkl')

    gb_model_loaded = joblib.load(r'../models/trialmodels/gradient_boosting_model.pkl')

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
    scaler.fit(range_data2[numeric_columns])

    # print(scaler.transform(input_df[numeric_columns]))
    input_df[numeric_columns] = scaler.transform(input_df[numeric_columns])
    
    #remove header
    # input_df_headless = input_df.values
    input_df_headless = input_df.values
    # print('input_df_headless = ',input_df_headless)

    # Predict using the models
    pred_catboost = model_catboost.predict(input_df)
    pred_dnn = model_dnn.predict(input_df_headless)
    pred_lgbm = model_lgbm.predict(input_df)
    pred_xgb = model_xgb.predict(xgb.DMatrix(input_df))
    pred_tf_1 = model_tf_1.predict(input_df_headless)
    pred_tf_2 = model_tf_2.predict(input_df_headless)
    pred_tf_3 = model_tf_3.predict(input_df_headless)
    pred_tf_4 = model_tf_4.predict(input_df_headless)
    pred_rf = rf_model.predict(input_df)
    pred_stacking = stacking_regressor.predict(input_df)
    pred_gb = gb_model_loaded.predict(input_df)

    print('Catboost:', pred_catboost)
    print('DNN:', pred_dnn)
    print('LGBM:', pred_lgbm)
    print('XGB:', pred_xgb)
    print('TF1:', pred_tf_1)
    print('TF2:', pred_tf_2)
    print('TF3:', pred_tf_3)
    print('TF4:', pred_tf_4)
    print('RF:', pred_rf)
    print('Stacking:', pred_stacking)
    print('GB:', pred_gb)

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
    pred_rf = scaler2.inverse_transform([pred_rf])
    pred_stacking = scaler2.inverse_transform([pred_stacking])
    pred_gb = scaler2.inverse_transform([pred_gb])


    return pred_catboost[0], pred_dnn[0][0], pred_lgbm[0], pred_xgb[0], pred_tf_1[0][0], pred_tf_2[0][0], pred_tf_3[0][0], pred_tf_4[0][0], pred_rf[0], pred_stacking[0], pred_gb[0]

def plot_predictions(ans):
    # Plot the predictions
    models = ['Catboost', 'DNN', 'LGBM', 'XGB', 'TF1', 'TF2', 'TF3', 'TF4', 'RF', 'Stacking', 'GB']
    predictions = [ans[0][0], ans[1], ans[2][0], ans[3][0], ans[4], ans[5], ans[6], ans[7], ans[8][0], ans[9][0], ans[10][0]]

    plt.figure(figsize=(10, 6))
    plt.plot(models, predictions, marker='o', label='Predictions')
    # plt.axhline(y=actual_value, color='r', linestyle='--', label='Actual Value')
    plt.xlabel('Models')
    plt.ylabel('PCI (%)')
    plt.title('Model Predictions vs Actual Value')
    plt.legend()
    plt.grid(True)
    # plt.show()
    plt.savefig(r'../model_predictions_temp.png', dpi=400, bbox_inches='tight')
    return r'../model_predictions_temp.png'

def encode_base64_image(image_path):
    with open(image_path, "rb") as img_file:
        my_string = base64.b64encode(img_file.read())
    return my_string

def generate_prediction_json(input_list):
    ans = predict_all_models(*input_list)
    image_path = plot_predictions(ans)
    image_base64 = encode_base64_image(image_path)
    return {
        'predictions': {
            'catboost': str(str(ans[0])[1:-1]),
            'dnn': str(ans[1]),
            'lgbm': str(str(ans[2])[1:-1]),
            'xgb': str(str(ans[3])[1:-1]),
            'tf1': str(ans[4]),
            'tf2': str(ans[5]),
            'tf3': str(ans[6]),
            'tf4': str(ans[7]),
            'rf': str(str(ans[8])[1:-1]),
            'stacking': str(str(ans[9])[1:-1]),
            'gb': str(str(ans[10])[1:-1])
        },
        'image_base64': (image_base64)
    }


# test data

#print(generate_prediction_json([3.9, 1.7, 0.3, 18.2, 3.5, 50.0, 6, 13.5, 'Low', 8200, 10.1, 102.0, 'Major repairs']))

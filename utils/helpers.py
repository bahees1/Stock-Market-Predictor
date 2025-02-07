import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

#Preprocess data(feature selection, scaling, etc...)
def preprocess_data(data):
    #feature selection: We can get things like High, Low, Volume, etc...
    data['Volume'] = data['Volume'].fillna('0').replace({'K': '*1e3', 'M': '*1e6', 'B': '*1e9'}, regex=True).map(pd.eval).astype(float)
    features = data[['Open', 'High', 'Low', 'Volume']].values
    
    target = data['Price'].values

    #Normalize features 
    scaler = MinMaxScaler(feature_range=(0,1))
    features_scaled = scaler.fit_transform(features)

    #reshape for LSTM (samples, time steps, features)
    features_scaled = np.reshape(features_scaled, (features_scaled.shape[0], 1, features_scaled.shape[1]))

    return features_scaled, target

#function that predicts prices using the trained model
def predict_stock_prices(model, features):
    #make prediction
    predicions = model.predict(features)
    return predicions.ravel()




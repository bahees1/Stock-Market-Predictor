import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


#function that predicts prices using the trained model
def predict_stock_prices(model, features):
    #make prediction
    predicions = model.predict(features)
    return predicions




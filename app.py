import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler


from utils.helpers import preprocess_data, predict_stock_prices

#Setup the home page
st.title('Stock Predictor App')
st.sidebar.header("Options")

# Load the trained model
model = load_model('model/model.h5')

#upload the CSV file with the stock data
uploaded_file = st.sidebar.file_uploader("Upload your stock data CSV", type={"csv"})

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.write("Stock Data", data.tail())

    #preprocess data for prediction
    features, target = preprocess_data(data)

    #make prediction
    predictions = predict_stock_prices(model, features)

    #plot actual vs predicted prices
    st.subheader("Actual Stock Price vs Predicted Price")
    st.line_chart(data={'Actual': target, 'Predicted': predictions})

    #give option for user to download the prediction results as csv
    st.download_button(
        label = "Download Predicted Results",
        data=predictions.to_csv(index = False),
        file_name = "stock_predictions.csv",
        mime = "text/csv"
    )
    

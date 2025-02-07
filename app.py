import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model
from utils.helpers import preprocess_data
from sklearn.preprocessing import MinMaxScaler


from utils.helpers import preprocess_data, predict_stock_prices

#Setup the home page
st.title('Stock Predictor App')
st.sidebar.header("Options")

# Load the trained model
model = load_model('model/model/model.h5')

#upload the CSV file with the stock data
uploaded_file = st.sidebar.file_uploader("Upload your stock data CSV", type={"csv"})

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.write("Stock Data", data.tail())

    if 'Date' in data.columns:
        data['Date'] = pd.to_datetime(data['Date'])

        # Extract the year from the 'Date' column for plotting purposes
        data['Year'] = data['Date'].dt.year

    #preprocess data for prediction
    features, target = preprocess_data(data)

    #make prediction
    predictions = predict_stock_prices(model, features)

    #need to convert to pandas series
    target_series = pd.Series(target, name="Actual")
    prediction_series = pd.Series(predictions, name= "Predicted")

   

    #plot actual vs predicted prices
    st.subheader("Actual Stock Price vs Predicted Price")
    # Create a DataFrame for plotting, using 'Date' as the index for the x-axis
    plot_data = pd.DataFrame({'Date': data['Date'], 'Actual': target_series, 'Predicted': prediction_series})
    # Set the 'Date' column as the index
    plot_data.set_index('Date', inplace=True)

    st.line_chart(plot_data)

    #give option for user to download the prediction results as csv
    st.download_button(
        label = "Download Predicted Results",
        data=prediction_series.to_csv(index = False),
        file_name = "stock_predictions.csv",
        mime = "text/csv"
    )


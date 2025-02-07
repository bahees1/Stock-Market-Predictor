import streamlit as st
import matplotlib.pyplot as plt

from utils.helpers import get_stock_data, prepare_data, train_model

st.title('Stock Predictor App')

ticker = st.text_input('Enter Stock Ticker', 'AAPL')

if ticker:
    stock_data = get_stock_data(ticker)

    if not stock_data.empty:
        processed_data = prepare_data(stock_data)

        model = train_model(processed_data)

        #plot historical prices
        st.subheader(f'{ticker} Stock Price Trend')
        plt.figure(figsize=(10, 6))
        plt.plot(stock_data['Close'], label = 'Historical Prices')
        plt.title(f'{ticker} Stock Prices')
        plt.xlabel('Data')
        plt.ylabel('Prices')
        plt.legend()
        st.pyplot(plt)
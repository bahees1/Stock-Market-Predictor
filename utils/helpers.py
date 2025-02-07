import yfinance as yf
import pandas as pd
from sklearn.linear_model import LinearRegression

#this function gets the stock data for the ticker that the user inputs
def get_stock_data(ticker, start = "2015-01-01", end = "2024-01-01"):
    stock_data = yf.download(ticker, start = start, end = end)
    return stock_data

def prepare_data(stock_data):
    stock_data['Return'] = stock_data['Close'].pct_change() #identifies daily returns
    stock_data['SMA_10'] = stock_data['Close'].rolling(window=10).mean() #defining a 10-day moving average
    stock_data.dropna(inplace = True) # Drop missing values

    return stock_data

def train_model(stock_data):
    X = stock_data[['SMA_10', 'Return']].values
    y = stock_data['Close'].values
    model = LinearRegression()
    model.fit(X, y)
    
    return model



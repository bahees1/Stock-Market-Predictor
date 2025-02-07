import pandas as pd
import numpy as np
from keras.models import Sequential, load_model
from keras.layers import Dense, LSTM
from sklearn.model_selection import train_test_split
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


#load the given stock data
data = pd.read_csv('C:/Users/Sarujan/Desktop/VSCODE PROJECTS/Stock-Market-Predictor/data/stock_data.csv')

#start by preprocessing data
X, y = preprocess_data(data)

#split the data into train and testing sets for the model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

#build the LTSM model
model = Sequential()
model.add(LSTM(units = 50, return_sequences = True, input_shape=(1, X_train.shape[2])))
model.add(LSTM(units=50, return_sequences=False))
model.add(Dense(units=25, activation='relu'))
model.add(Dense(units=1)) #Output layer

#compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

#Train the model
model.fit(X_train, y_train, epochs = 50, batch_size =32, validation_split=0.2)

#save the model 
model.save('model/model.h5')

# importing the libraries and frameworks
import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential,load_model
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras.optimizers import RMSprop
from sklearn.preprocessing import MinMaxScaler
from selenium import webdriver
from selenium.webdriver.common.by import By
import wget
import os

cname_of = str(input().upper()) # eth of btc

def get_data(cn="BTC"): # function for a bitcoin-price scraping
    try:
        options = webdriver.ChromeOptions()
        options.add_argument('--headless')

        driver = webdriver.Chrome(options=options)
        driver.get(f"https://finance.yahoo.com/quote/{cn}-USD/history")

        get_text = lambda obj: obj.text
        els = driver.find_elements(By.TAG_NAME, 'a')
        data = list(map(get_text, els))
        try:
            os.remove(f"{cname_of}-USD.csv")
        except:
            pass
        wget.download(els[data.index('Download')].get_attribute('href'))
    except:
        return False
    return True

if not get_data(cname_of):
    get_data(cname_of)

# data preprocessing
dataset = pd.read_csv(f"{cname_of}-USD.csv")
dataset = dataset.drop(['Date', 'Open', 'High', 'Low', 'Adj Close', 'Volume'], axis=1)

scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(np.array(dataset))

# creating a test and train data
train_size = int(len(dataset)*0.85)
train_data, test_data = dataset[:train_size], dataset[train_size:]

def create_ds(dataset, time_step): # function for creating a dataset
    dataX, dataY = [], []
    for i in range(len(dataset)-time_step-1): # adding the data for a X(values) and Y(values for checking)
        frame = dataset[i:i+time_step, 0]
        dataX.append(frame)
        dataY.append(dataset[i+time_step, 0])
    return np.array(dataX), np.array(dataY)

# creating datasets for training and testing
X_train, y_train = create_ds(train_data, 15)
X_test, y_test = create_ds(test_data, 15)

# resizing
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

# creating model's structure
model = Sequential([
    LSTM(32, input_shape=(None, 1), activation='relu'), # 32 and , return_sequences=True
    Dropout(0.2),
    Dense(16, activation='relu'),
    Dense(1)
])

model.compile(loss='mse', optimizer=RMSprop(0.02)) # model's compiling
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=64, verbose=1) # training

# using neural network
dlen = len(dataset)
last_prices = np.reshape(dataset[-dlen:], (1, dlen, 1))
res = model.predict(last_prices)
res = scaler.inverse_transform(res)[0, 0]
print(res)

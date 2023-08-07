#%%
import numpy as np
import pandas as pd
import datetime
import matplotlib.pyplot as plt
from pandas_datareader import data as web
import yfinance as yf
import seaborn as sb
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense
from pydbr import DBNRegressor


# %%
#Obtendo cotacoes
tickers = ["TAEE11.SA"]
yf.pdr_override()
start_date = "2020-01-01"
end_date = "2023-07-14"
data = web.get_data_yahoo(tickers, start=start_date, end=end_date)
data.head()
data.shape[0]
# %%
#Segmentando Dataframe para obter valores de interesse
data_close = pd.DataFrame(data["Adj Close"])
data_close.head()

#%%
#Plotando variavel de interesse ao longo do tempo
sb.set()
data_close.plot()

#%%
data_close = data_close.values.reshape(-1,1)

#%%
# Split the data into training and testing sets
split_ratio = 0.8
split_index = int(split_ratio * len(data_close))
X_train, X_test = data_close[:split_index], data_close[split_index:]

# Create and train the Deep Belief Network (DBN) model
dbn_model = DBNRegressor(hidden_layers=[100, 50], epochs=100, learning_rate=0.01)
dbn_model.fit(X_train, X_train)

# Make predictions using the trained model
y_pred = dbn_model.predict(X_test)

# Visualize the results (actual vs. predicted)
plt.figure(figsize=(12, 6))
plt.plot(data.index[-len(y_pred):], data_close[split_index:], label='Actual Prices')
plt.plot(data.index[-len(y_pred):], y_pred, label='Predicted Prices', linestyle='dashed')
plt.xlabel('Date')
plt.ylabel('Stock Price')
plt.title('Stock Market Price Prediction using Deep Belief Networks (DBN)')
plt.legend()
plt.show()
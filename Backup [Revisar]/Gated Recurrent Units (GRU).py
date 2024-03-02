#%%
import numpy as np
import pandas as pd
import datetime
import matplotlib.pyplot as plt
from pandas_datareader import data as web
import yfinance as yf
import seaborn as sb
import tensorflow
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense
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
# %%
# Define the number of time steps and features
time_steps = 3  # Number of previous days' prices to consider for prediction
n_features = 1   # We are only using the 'Close' price as a feature
data_close = data_close.values.reshape(-1,1)
#%%
# Prepare the data in sequences for training
def create_sequences(data, time_steps):
    X, y = [], []
    for i in range(len(data) - time_steps):
        X.append(data[i:i + time_steps])
        y.append(data[i + time_steps])
    return np.array(X), np.array(y)

X, y = create_sequences(data_close, time_steps)

#%%
# Split the data into training and testing sets
split_ratio = 0.8
split_index = int(split_ratio * len(X))
X_train, X_test = X[:split_index], X[split_index:]
y_train, y_test = y[:split_index], y[split_index:]

# Build the GRU model
model = Sequential()
model.add(GRU(64, activation='relu', input_shape=(time_steps, n_features)))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mean_squared_error')

#%%
# Train the model
history = model.fit(X_train, y_train, epochs=50, batch_size=16, validation_split=0.2, verbose=1)

# Predict the stock prices using the trained model
y_pred = model.predict(X_test)

# Visualize the results
plt.figure(figsize=(12, 6))
plt.plot(data.index[-len(y_test):], y_test, label='Actual Prices')
plt.plot(data.index[-len(y_test):], y_pred, label='Predicted Prices', linestyle='dashed')
plt.xlabel('Date')
plt.ylabel('Stock Price')
plt.title('GRU Model for Stock Price Prediction')
plt.legend()
plt.show()

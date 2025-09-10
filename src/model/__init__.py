import yfinance as yf
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt


# Get data
ticker = yf.Ticker("GOOGL")
data = ticker.history(period="1y", interval="1h")

# Feature engineering
data["Return"] = data["Close"].pct_change()  # daily returns
data["MA5"] = data["Close"].rolling(5).mean()
data["MA10"] = data["Close"].rolling(10).mean()
data["Target"] = data["Close"].shift(-1)  # next day close

# Drop NA values from rolling windows
data = data.dropna()
print(data.head())

X = data[["Return", "MA5", "MA10"]]
y = data["Target"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

model = LinearRegression()
model.fit(X_train, y_train)

preds = model.predict(X_test)
mse = mean_squared_error(y_test, preds)

print("MSE:", mse)

scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data[["Close"]])

# Prepare sequences
X_seq, y_seq = [], []
seq_length = 60  # 60 days
for i in range(len(scaled_data) - seq_length):
    X_seq.append(scaled_data[i:i+seq_length])
    y_seq.append(scaled_data[i+seq_length])

X_seq, y_seq = np.array(X_seq), np.array(y_seq)

# Train/test split
split = int(0.8 * len(X_seq))
X_train, X_test = X_seq[:split], X_seq[split:]
y_train, y_test = y_seq[:split], y_seq[split:]

# Build model
model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], 1)),
    Dropout(0.2),
    LSTM(50),
    Dense(1)
])

model.compile(optimizer="adam", loss="mse")
model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_test, y_test))

# --- Scaling (double-check) ---
scaler = MinMaxScaler(feature_range=(0,1))
scaled = scaler.fit_transform(data[["Close"]])

# Inverse-transform predictions correctly
preds = model.predict(X_test)
preds = scaler.inverse_transform(preds)
actual = scaler.inverse_transform(y_test.reshape(-1,1))




rmse = np.sqrt(mean_squared_error(actual, preds))
print("RMSE:", rmse)

plt.figure(figsize=(12,6))
plt.plot(actual, color="black", label="Actual Price")
plt.plot(preds, color="green", label="Predicted Price")
plt.legend()
plt.show()

import yfinance as yf
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import joblib


ticker = yf.Ticker("TSLA")
data = ticker.history(period="1y", interval="1h")

data["Return"] = data["Close"].pct_change()
data["MA5"] = data["Close"].rolling(5).mean()
data["MA10"] = data["Close"].rolling(10).mean()
data["Target"] = data["Close"].shift(-1)

data = data.dropna()
print(data.head())


X = data[["Return", "MA5", "MA10"]]
y = data["Target"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

lin_model = LinearRegression()
lin_model.fit(X_train, y_train)

preds = lin_model.predict(X_test)
mse = mean_squared_error(y_test, preds)
print("Linear Regression MSE:", mse)

# Save Linear Regression model (optional)
joblib.dump(lin_model, "linear_model.pkl")


scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data[["Close"]])

X_seq, y_seq = [], []
seq_length = 60  # 60 timesteps
for i in range(len(scaled_data) - seq_length):
    X_seq.append(scaled_data[i:i+seq_length])
    y_seq.append(scaled_data[i+seq_length])

X_seq, y_seq = np.array(X_seq), np.array(y_seq)

split = int(0.8 * len(X_seq))
X_train, X_test = X_seq[:split], X_seq[split:]
y_train, y_test = y_seq[:split], y_seq[split:]

lstm_model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], 1)),
    Dropout(0.2),
    LSTM(50),
    Dense(1)
])

lstm_model.compile(optimizer="adam", loss="mse")
lstm_model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_test, y_test)) 


lstm_model.save("lstm_model.keras")
joblib.dump(scaler, "scaler.pkl")
print("LSTM model saved as lstm_model.keras")
print("Scaler saved as scaler.pkl")


preds = lstm_model.predict(X_test)
preds = scaler.inverse_transform(preds)
actual = scaler.inverse_transform(y_test.reshape(-1, 1))

rmse = np.sqrt(mean_squared_error(actual, preds))
print("LSTM RMSE:", rmse)

plt.figure(figsize=(12, 6))
plt.plot(actual, color="black", label="Actual Price")
plt.plot(preds, color="green", label="Predicted Price")
plt.legend()
plt.show()


last_window = scaled_data[-60:] 
last_window = np.reshape(last_window, (1, 60, 1))
next_pred = lstm_model.predict(last_window)
next_price = scaler.inverse_transform(next_pred)[0][0]
print("Next predicted price:", next_price)

future_days = 30
future_preds = []
window = scaled_data[-60:].tolist()

for _ in range(future_days):
    X_input = np.array(window[-60:]).reshape(1, 60, 1)
    pred = lstm_model.predict(X_input, verbose=0)
    future_preds.append(pred[0][0])
    window.append([pred[0][0]])

future_preds = scaler.inverse_transform(np.array(future_preds).reshape(-1, 1))

# plt.figure(figsize=(12, 6))
# plt.plot(data.index, data["Close"], label="Historical Price")
# plt.plot(pd.date_range(data.index[-1], periods=future_days+1, freq="B")[1:],
#          future_preds, label="Future Prediction", color="red")
# plt.legend()
# plt.show()

# print(lin_model.predict(5))

lin_model = joblib.load('linear_model.pkl')
ltsm_model = load_model('lstm_model.keras')
scaler = joblib.load("scaler.pkl")

sample_features = [[0.01, 250.5, 252.1]]  
lin_pred = lin_model.predict(sample_features)
print("Linear Regression Prediction:", lin_pred[0])
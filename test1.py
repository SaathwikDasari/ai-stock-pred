import yfinance as yf
from flask import Flask, jsonify, request
from flask_cors import CORS
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from datetime import timedelta
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)
CORS(app)

items = []

@app.route('/api/receive_data', methods=['GET', 'POST'])
def receive_data():
    """
    API endpoint to receive stock ticker and return historical and predicted prices.
    """
    try:
        # Getting JSON
        if request.method == 'POST':
            posted_data = request.json
        else: 
            posted_data = request.args
            
        if not posted_data or "ticker" not in posted_data:
            return jsonify({"error": "No stock ticker provided in the request"}), 400
        
        ticker = posted_data["ticker"]
        print(f"Received request for ticker: {ticker}")

        p_data = run_stock_prediction(ticker)

        if p_data is None:
            return jsonify({"error": f"Could not process prediction for {ticker}"}), 500

        return jsonify({"message": "Data processed successfully", "data": p_data}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


class LSTMStockPredictor:
    def __init__(self, symbol, period='5y'):
        """
        Initialize the LSTM stock predictor
        
        Args:
            symbol (str): Stock symbol (e.g., 'AAPL', 'MSFT')
            period (str): Data period ('1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max')
        """
        self.symbol = symbol
        self.period = period
        self.data = None
        self.model = None
        self.scaler = MinMaxScaler()
        self.sequence_length = 60
        
    def fetch_data(self):
        """Fetch stock data using yfinance"""
        try:
            stock = yf.Ticker(self.symbol)
            self.data = stock.history(period=self.period)
            if self.data.empty:
                print(f"No data found for {self.symbol}")
                return False
            print(f"Successfully fetched {len(self.data)} days of data for {self.symbol}")
            return True
        except Exception as e:
            print(f"Error fetching data for {self.symbol}: {e}")
            return False
    
    def prepare_lstm_data(self):
        """Prepare data for LSTM model"""
        if self.data is None:
            print("No data available. Please fetch data first.")
            return None, None
            
        data = self.data['Close'].values.reshape(-1, 1)
        
        scaled_data = self.scaler.fit_transform(data)
        
        X, y = [], []
        for i in range(self.sequence_length, len(scaled_data)):
            X.append(scaled_data[i-self.sequence_length:i, 0])
            y.append(scaled_data[i, 0])
        
        return np.array(X), np.array(y)
    
    def build_lstm_model(self, input_shape):
        """Build LSTM model"""
        model = Sequential([
            LSTM(50, return_sequences=True, input_shape=input_shape),
            Dropout(0.2),
            LSTM(50, return_sequences=True),
            Dropout(0.2),
            LSTM(50),
            Dropout(0.2),
            Dense(25),
            Dense(1)
        ])
        
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
        return model
    
    def train_model(self, epochs=50, validation_split=0.2):
        """Train LSTM model"""
        print(f"\nPreparing LSTM data for {self.symbol}...")
        X, y = self.prepare_lstm_data()
        
        if X is None or y is None:
            print("Failed to prepare data")
            return False
        
        train_size = int(len(X) * (1 - validation_split))
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]
        
        X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
        X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
        
        self.model = self.build_lstm_model((X_train.shape[1], 1))
        
        print(f"Training LSTM model for {epochs} epochs...")
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=epochs,
            batch_size=32,
            verbose=1
        )
        
        self.evaluate_model(X_test, y_test)
        
        return True
    
    def evaluate_model(self, X_test, y_test):
        """Evaluate the trained model"""
        if self.model is None:
            print("No trained model available")
            return
        
        predictions = self.model.predict(X_test, verbose=0)
        
        predictions = self.scaler.inverse_transform(predictions)
        y_test_actual = self.scaler.inverse_transform(y_test.reshape(-1, 1))
        
        mse = mean_squared_error(y_test_actual, predictions)
        mae = mean_absolute_error(y_test_actual, predictions)
        r2 = r2_score(y_test_actual, predictions)
        rmse = np.sqrt(mse)
        
        print(f"\n{'='*50}")
        print(f"LSTM MODEL EVALUATION FOR {self.symbol}")
        print(f"{'='*50}")
        print(f"Mean Squared Error (MSE):  {mse:.4f}")
        print(f"Root Mean Squared Error:   {rmse:.4f}")
        print(f"Mean Absolute Error (MAE): {mae:.4f}")
        print(f"R² Score:                  {r2:.4f}")
        
        self.evaluatiton_results = {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'predictions': predictions.flatten(),
            'actual': y_test_actual.flatten()
        }
    
    def predict_next_prices(self, days_ahead=1):
        """Predict future stock prices"""
        if self.model is None:
            print("No trained model available. Please train the model first.")
            return None
        
        if self.data is None or len(self.data) < self.sequence_length:
            print("Insufficient data for prediction")
            return None
        
        predictions = []
        
        last_sequence = self.data['Close'].tail(self.sequence_length).values.reshape(-1, 1)
        scaled_sequence = self.scaler.transform(last_sequence)
        
        current_sequence = scaled_sequence.copy()
        
        for day in range(days_ahead):
            X_pred = current_sequence.reshape(1, self.sequence_length, 1)
            
            next_price_scaled = self.model.predict(X_pred, verbose=0)
            
            next_price = self.scaler.inverse_transform(next_price_scaled)[0][0]
            predictions.append(next_price)
            
            current_sequence = np.append(current_sequence[1:], next_price_scaled[0])
        
        return predictions
    
    def get_prediction_summary(self, days_ahead=5):
        """Get summary of predictions"""
        if self.data is None:
            return "No data available"
        
        current_price = self.data['Close'].iloc[-1]
        predictions = self.predict_next_prices(days_ahead)
        
        if predictions is None:
            return "Unable to make predictions"
        
        summary = f"\n{'='*60}\n"
        summary += f"STOCK PREDICTION SUMMARY FOR {self.symbol}\n"
        summary += f"{'='*60}\n"
        summary += f"Current Price: ${current_price:.2f}\n"
        summary += f"{'='*60}\n"
        
        for i, pred in enumerate(predictions, 1):
            change = pred - current_price
            change_pct = (change / current_price) * 100
            summary += f"Day +{i}: ${pred:.2f} ({change:+.2f}, {change_pct:+.2f}%)\n"
        
        return summary


def run_stock_prediction(symbol='AAPL', period='2y', epochs=30, days_ahead=5):
    """
    Complete workflow for LSTM stock prediction and returns structured data.
    """
    print(f"Starting LSTM stock prediction for {symbol}")
    print("="*60)
    
    predictor = LSTMStockPredictor(symbol, period)
    
    if not predictor.fetch_data():
        return None
    
    if not predictor.train_model(epochs=epochs):
        return None
    
    historical_data = predictor.data['Close'].tail(30)
    historical_prices = [
        {"date": str(date.date()), "price": float(price)}
        for date, price in historical_data.items()
    ]

    predictions = predictor.predict_next_prices(days_ahead)
    
    if predictions is None:
        return None

    last_date = historical_data.index[-1].date()

    predicted_prices = []
    for i, pred_price in enumerate(predictions, 1):
        future_date = last_date + timedelta(days=i)
        predicted_prices.append({
            "date": str(future_date),
            "price": float(pred_price)
        })

    result = {
        "ticker": symbol,
        "historical_prices": historical_prices,
        "predicted_prices": predicted_prices
    }
    
    return result

if __name__ == "__main__":
    app.run(host='0.0.0.0', debug=True)

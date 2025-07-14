from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import joblib
import yfinance as yf
import ta

app = Flask(__name__)
CORS(app)

MODEL_PATH = "models/trade_model.pkl"
model = joblib.load(MODEL_PATH)

FEATURES = [
    "RSI_14", "MACD", "MACD_signal", "MACD_diff",
    "SMA_20", "SMA_50", "EMA_20", "EMA_50"
]

def calculate_indicators(df):
    df["RSI_14"] = ta.momentum.RSIIndicator(df["Close"], window=14).rsi()
    macd = ta.trend.MACD(df["Close"])
    df["MACD"] = macd.macd()
    df["MACD_signal"] = macd.macd_signal()
    df["MACD_diff"] = macd.macd_diff()
    df["SMA_20"] = df["Close"].rolling(window=20).mean()
    df["SMA_50"] = df["Close"].rolling(window=50).mean()
    df["EMA_20"] = df["Close"].ewm(span=20, adjust=False).mean()
    df["EMA_50"] = df["Close"].ewm(span=50, adjust=False).mean()
    return df

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    symbol = data.get("symbol")
    if not symbol:
        return jsonify({"error": "No symbol provided"}), 400

    try:
        df = yf.download(symbol, period="1y", interval="1d", group_by="ticker")
        df.reset_index(inplace=True)

        # 🔥 Fix multi-indexed DataFrame if present
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [col[1] if col[1] else col[0] for col in df.columns]

        # ✅ Now ensure 'Close' is numeric
        if "Close" not in df.columns:
            return jsonify({"error": "'Close' column not found in data"}), 400

        df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
        df.dropna(inplace=True)

        df = calculate_indicators(df)
        df.dropna(inplace=True)

        if df.empty:
            return jsonify({"error": "Not enough data after indicators"}), 400

        latest = df.iloc[-1]
        input_data = np.array([[latest[feat] for feat in FEATURES]])

        prediction = model.predict(input_data)[0]
        label_map = {0: "Sell", 1: "Hold", 2: "Buy"}
        return jsonify({"prediction": label_map[prediction]})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(port=5055, debug=True)

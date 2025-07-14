import yfinance as yf
import pandas as pd
import numpy as np
import joblib
import warnings
warnings.filterwarnings("ignore")

from ta.trend import MACD, SMAIndicator, EMAIndicator
from ta.momentum import RSIIndicator

FEATURES = [
    "RSI_14", "MACD", "MACD_signal", "MACD_diff",
    "SMA_20", "SMA_50", "EMA_20", "EMA_50"
]

def calculate_indicators(df):
    df["RSI_14"] = RSIIndicator(close=df["Close"], window=14).rsi()
    macd = MACD(close=df["Close"])
    df["MACD"] = macd.macd()
    df["MACD_signal"] = macd.macd_signal()
    df["MACD_diff"] = macd.macd_diff()
    df["SMA_20"] = SMAIndicator(close=df["Close"], window=20).sma_indicator()
    df["SMA_50"] = SMAIndicator(close=df["Close"], window=50).sma_indicator()
    df["EMA_20"] = EMAIndicator(close=df["Close"], window=20).ema_indicator()
    df["EMA_50"] = EMAIndicator(close=df["Close"], window=50).ema_indicator()
    return df

def main():
    symbol = input("📈 Enter stock symbol (e.g., TCS.NS): ").upper()
    if not symbol.endswith(".NS"):
        symbol += ".NS"

    print(f"\n📥 Fetching latest data for {symbol}...")
    df = yf.download(symbol, period="1y", interval="1d", group_by="ticker")

    if df.empty:
        print("❌ Failed to fetch stock data.")
        return

    df.reset_index(inplace=True)

    print("🔍 Available columns after reset:")
    print(df.columns.tolist())  # For debugging

    # Flatten if necessary (e.g., if columns are multi-level)
    if isinstance(df.columns[1], tuple):
        df.columns = df.columns.get_level_values(1)

    if "Close" not in df.columns:
        print("❌ 'Close' column not found. Columns are:", df.columns)
        return

    df["Close"] = pd.to_numeric(df["Close"], errors="coerce")

    print(f"✅ Fetched {len(df)} rows before indicators.")
    df = calculate_indicators(df)
    df.dropna(inplace=True)
    print(f"📊 Rows after indicators: {len(df)}")

    if len(df) < 1:
        print("⚠️ Not enough data after indicator calculation.")
        return

    latest = df.iloc[-1]
    features_dict = {feat: latest[feat] for feat in FEATURES}

    print("🧪 Feature values:")
    for k, v in features_dict.items():
        print(f"   {k}: {v}")

    if any(v is None or pd.isna(v) or np.isinf(v) for v in features_dict.values()):
        print("❌ One or more features are missing or invalid. Cannot proceed with prediction.")
        return

    input_data = np.array([list(features_dict.values())])

    try:
        model = joblib.load("models/trade_model.pkl")
    except FileNotFoundError:
        print("❌ Model file not found: models/trade_model.pkl")
        return

    prediction = model.predict(input_data)[0]

    labels = {0: "Sell", 1: "Hold", 2: "Buy"}
    print(f"\n📌 Prediction for {symbol}: {labels[prediction]}")

if __name__ == "__main__":
    main()

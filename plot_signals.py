import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

# Load model
model = joblib.load("models/trade_model.pkl")

# Features used during training
FEATURES = [
    "RSI_14", "MACD", "MACD_signal", "MACD_diff",
    "SMA_20", "SMA_50", "EMA_20", "EMA_50"
]

# Load stock data
df = pd.read_csv("data/processed/TCS.csv").dropna()

# Drop last row from df and X to match shape
df = df.iloc[:-1].copy()
X = df[FEATURES]

# Predict
predictions = model.predict(X)
df["Prediction"] = predictions

# Plotting
plt.figure(figsize=(14, 7))
plt.plot(df["Date"], df["Close"], label="Close Price", color="black")

# Buy (2), Sell (0)
buy_signals = df[df["Prediction"] == 2]
sell_signals = df[df["Prediction"] == 0]

# Plot buy/sell points
plt.scatter(buy_signals["Date"], buy_signals["Close"], label="Buy", marker="^", color="green", s=100)
plt.scatter(sell_signals["Date"], sell_signals["Close"], label="Sell", marker="v", color="red", s=100)

plt.title("Buy/Sell Signals - TCS")
plt.xlabel("Date")
plt.ylabel("Price")
plt.xticks(rotation=45)
plt.legend()
plt.grid(True)
plt.tight_layout()

plt.show()

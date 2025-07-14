import pandas as pd
import joblib
import numpy as np
from sklearn.metrics import classification_report

# Load the trained model
model = joblib.load("models/trade_model.pkl")

# Define input features
FEATURES = [
    "RSI_14", "MACD", "MACD_signal", "MACD_diff",
    "SMA_20", "SMA_50", "EMA_20", "EMA_50"
]

# Human-readable label mapping (optional)
LABELS = {0: "Sell", 1: "Hold", 2: "Buy"}

# Load processed data for a single stock (TCS)
df = pd.read_csv("data/processed/TCS.csv").dropna()

# Generate actual labels from real price movement
df["Real"] = np.where(df["Close"].shift(-1) > df["Close"], 2,
              np.where(df["Close"].shift(-1) < df["Close"], 0, 1))

# Remove last row (no next-day price)
df = df[:-1]

# Extract features and true labels
X = df[FEATURES]
y_true = df["Real"]
y_pred = model.predict(X)

# Add predictions to DataFrame for reference
df["Prediction"] = y_pred

# Print accuracy report
print("\n📊 Backtest Accuracy Report (TCS):\n")
print(classification_report(
    y_true,
    y_pred,
    labels=[0, 1, 2],
    target_names=["Sell", "Hold", "Buy"]
))

# Save result for manual analysis or future plotting
df[["Date", "Close", "Real", "Prediction"]].to_csv("TCS_backtest.csv", index=False)
print("✅ Saved prediction report to TCS_backtest.csv")

import os
import pandas as pd
import joblib

# Load trained model
model = joblib.load("models/trade_model.pkl")

# Features used during training
FEATURES = [
    "RSI_14", "MACD", "MACD_signal", "MACD_diff",
    "SMA_20", "SMA_50", "EMA_20", "EMA_50"
]

processed_folder = "data/processed"
results = []

for file in os.listdir(processed_folder):
    if not file.endswith(".csv"):
        continue

    symbol = file.replace(".csv", "")
    print(f"\n🔍 Checking: {symbol}")

    path = os.path.join(processed_folder, file)

    try:
        df = pd.read_csv(path)
    except Exception as e:
        print(f"❌ Error reading {symbol}: {e}")
        continue

    df.dropna(inplace=True)

    if len(df) < 20:
        print(f"⚠️ Skipping {symbol} — Not enough rows ({len(df)} rows)")
        continue

    missing = [col for col in FEATURES if col not in df.columns]
    if missing:
        print(f"⚠️ Skipping {symbol} — Missing columns: {missing}")
        continue

    df = df.iloc[:-1].copy()  # exclude last row
    X = df[FEATURES]

    try:
        preds = model.predict(X)
    except Exception as e:
        print(f"❌ Skipping {symbol} due to prediction error: {e}")
        continue

    df["Prediction"] = preds

    balance = 10000.0
    quantity = 0
    trades = 0

    for i in range(len(df)):
        price = df.iloc[i]["Close"]
        signal = df.iloc[i]["Prediction"]

        if signal == 2 and balance >= price:
            qty = int(balance // price)
            balance -= qty * price
            quantity += qty
            trades += 1

        elif signal == 0 and quantity > 0:
            balance += quantity * price
            quantity = 0
            trades += 1

    # Force sell at end
    if quantity > 0:
        final_price = df.iloc[-1]["Close"]
        balance += quantity * final_price
        quantity = 0

    final_value = balance
    net_profit = final_value - 10000

    results.append({
        "Symbol": symbol,
        "Final Value (INR)": round(final_value, 2),
        "Net Profit (INR)": round(net_profit, 2),
        "Total Trades": trades
    })

# Convert results to DataFrame
result_df = pd.DataFrame(results)

# 🔍 If no stocks were processed, exit
if result_df.empty:
    print("\n⚠️ No stock files processed. Make sure 'data/processed/' has valid .csv files with indicator columns.")
    exit()

# Sort by net profit
result_df.sort_values(by="Net Profit (INR)", ascending=False, inplace=True)

# Save to CSV
result_df.to_csv("batch_backtest_results.csv", index=False)
print("\n✅ All done! Results saved to: batch_backtest_results.csv")

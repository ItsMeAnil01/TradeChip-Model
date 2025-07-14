import pandas as pd
import joblib
import matplotlib.pyplot as plt

# Load trained model
model = joblib.load("models/trade_model.pkl")

# Features used during training
FEATURES = [
    "RSI_14", "MACD", "MACD_signal", "MACD_diff",
    "SMA_20", "SMA_50", "EMA_20", "EMA_50"
]

# Load stock data (change to any processed stock)
df = pd.read_csv("data/processed/TCS.csv").dropna()

# Prepare data
df = df.iloc[:-1].copy()
X = df[FEATURES]
preds = model.predict(X)
df["Prediction"] = preds

# Simulation variables
balance = 10000.0
quantity = 0
portfolio_values = []
trade_log = []

# Simulate trading
for i in range(len(df)):
    date = df.iloc[i]["Date"]
    price = df.iloc[i]["Close"]
    signal = df.iloc[i]["Prediction"]

    if signal == 2 and balance >= price:
        qty = int(balance // price)
        cost = qty * price
        balance -= cost
        quantity += qty
        trade_log.append(f"🟢 BUY {qty} @ ₹{price:.2f} on {date} | Balance: ₹{balance:.2f}")

    elif signal == 0 and quantity > 0:
        sale = quantity * price
        balance += sale
        trade_log.append(f"🔴 SELL {quantity} @ ₹{price:.2f} on {date} | Balance: ₹{balance:.2f}")
        quantity = 0

    portfolio_values.append(balance + quantity * price)

# Force sell at end if still holding
if quantity > 0:
    final_price = df.iloc[-1]["Close"]
    balance += quantity * final_price
    trade_log.append(f"⚠️ FINAL SELL {quantity} @ ₹{final_price:.2f} (forced)")
    quantity = 0
    portfolio_values[-1] = balance

# Add portfolio to DataFrame
df["Portfolio"] = portfolio_values

# Plot result
plt.figure(figsize=(14, 7))
plt.plot(df["Date"], df["Portfolio"], label="Portfolio Value", color="blue")
plt.title("Simulated Trading - TCS")
plt.xlabel("Date")
plt.ylabel("₹ Value")
plt.grid(True)
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()
plt.show()

# Print results
final_value = portfolio_values[-1]
net_profit = final_value - 10000
print("\n📋 Trade Log:")
print("\n".join(trade_log))

print(f"\n📊 Final portfolio value: ₹{final_value:,.2f}")
print(f"💰 Net Profit: ₹{net_profit:,.2f}")
print(f"🔁 Total Trades: {len([t for t in trade_log if 'BUY' in t or 'SELL' in t])}")

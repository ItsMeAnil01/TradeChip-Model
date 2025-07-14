import os
import yfinance as yf
from datetime import datetime, timedelta

# Folder & input setup
symbol_file = "nse_stock_list.txt"
output_folder = "data/raw"
os.makedirs(output_folder, exist_ok=True)

# 🗓 Date range: last 3 years (approx. 1100 days)
start_date = (datetime.now() - timedelta(days=1100)).strftime("%Y-%m-%d")
end_date = datetime.now().strftime("%Y-%m-%d")

# Read stock symbols (one per line, e.g., RELIANCE.NS)
with open(symbol_file, "r") as f:
    symbols = [line.strip() for line in f if line.strip()]

# Loop through each symbol and fetch data
for symbol in symbols:
    print(f"📥 Fetching data for: {symbol}")
    try:
        df = yf.download(
            symbol,
            start=start_date,
            end=end_date,
            interval="1d",
            auto_adjust=True
        )
        if df.empty:
            print(f"⚠️ No data for {symbol}")
            continue

        df.reset_index(inplace=True)
        output_path = os.path.join(output_folder, f"{symbol}.csv")
        df.to_csv(output_path, index=False)
        print(f"✅ Saved: {symbol}")
    except Exception as e:
        print(f"❌ Failed: {symbol} — {e}")

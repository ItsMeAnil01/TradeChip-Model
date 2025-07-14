import pandas as pd
import os
import ta  # Technical Analysis library

RAW_DATA_DIR = "data/raw"
PROCESSED_DATA_DIR = "data/processed"

def add_indicators(df):
    # Convert price columns to numeric, ignore errors (e.g., strings)
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Drop rows where Close is NaN after conversion
    df.dropna(subset=["Close"], inplace=True)

    # Add RSI
    df["RSI_14"] = ta.momentum.RSIIndicator(df["Close"], window=14).rsi()

    # Add MACD
    macd = ta.trend.MACD(df["Close"])
    df["MACD"] = macd.macd()
    df["MACD_signal"] = macd.macd_signal()
    df["MACD_diff"] = macd.macd_diff()

    # Add SMA and EMA
    df["SMA_20"] = df["Close"].rolling(window=20).mean()
    df["SMA_50"] = df["Close"].rolling(window=50).mean()
    df["EMA_20"] = df["Close"].ewm(span=20, adjust=False).mean()
    df["EMA_50"] = df["Close"].ewm(span=50, adjust=False).mean()

    return df

def process_file(file_path, output_path):
    try:
        df = pd.read_csv(file_path)
        if df.empty or 'Close' not in df.columns:
            print(f"❌ Skipped: {file_path} (invalid or empty)")
            return

        df = add_indicators(df)
        df.dropna(inplace=True)  # Drop any remaining NaNs from indicators

        df.to_csv(output_path, index=False)
        print(f"✅ Processed and saved: {output_path}")

    except Exception as e:
        print(f"❌ Error processing {file_path}: {e}")

def main():
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)

    for file_name in os.listdir(RAW_DATA_DIR):
        if file_name.endswith(".csv"):
            input_path = os.path.join(RAW_DATA_DIR, file_name)
            output_path = os.path.join(PROCESSED_DATA_DIR, file_name)
            process_file(input_path, output_path)

if __name__ == "__main__":
    main()

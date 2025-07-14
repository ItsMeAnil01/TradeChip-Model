import os
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib

PROCESSED_DATA_DIR = "data/processed"
MODEL_PATH = "models/trade_model.pkl"

def generate_labels(df):
    df["Target"] = np.where(df["Close"].shift(-1) > df["Close"], 2,
                    np.where(df["Close"].shift(-1) < df["Close"], 0, 1))
    return df

def load_data():
    all_data = []
    for file in os.listdir(PROCESSED_DATA_DIR):
        if file.endswith(".csv"):
            df = pd.read_csv(os.path.join(PROCESSED_DATA_DIR, file))
            df = generate_labels(df)
            all_data.append(df)

    if not all_data:
        print("❌ No CSV files found in data/processed/")
        exit()

    full_df = pd.concat(all_data, ignore_index=True)
    full_df.dropna(inplace=True)
    return full_df

def train_model(df):
    feature_cols = [
        "RSI_14", "MACD", "MACD_signal", "MACD_diff",
        "SMA_20", "SMA_50", "EMA_20", "EMA_50"
    ]

    X = df[feature_cols]
    y = df["Target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    print("\n📊 Classification Report:\n")
    print(classification_report(y_test, preds))

    joblib.dump(model, MODEL_PATH)
    print(f"\n✅ Model saved to {MODEL_PATH}")

def main():
    print("📥 Loading processed data...")
    df = load_data()

    print("🧠 Training model...")
    train_model(df)

if __name__ == "__main__":
    main()

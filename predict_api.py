from flask import Flask, request, jsonify
import numpy as np
import xgboost as xgb
import yfinance as yf
import pandas_ta as ta

app = Flask(__name__)

# Define feature order
FEATURES = [
    "RSI_14", "MACD", "MACD_signal", "MACD_diff",
    "SMA_20", "SMA_50", "EMA_20", "EMA_50"
]

# Reverse map for readability
LABEL_MAP = {0: "Sell", 1: "Hold", 2: "Buy"}

# Load the trained XGBoost model from JSON
booster = xgb.Booster()
booster.load_model("models/trade_model.json")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        # Ensure all required features are present
        if not all(feat in data for feat in FEATURES):
            return jsonify({"error": "Missing required features"}), 400

        # Extract features in correct order
        input_data = np.array([[data[feat] for feat in FEATURES]])
        dmatrix = xgb.DMatrix(input_data, feature_names=FEATURES)

        # Predict using the booster
        probs = booster.predict(dmatrix)
        prediction = int(np.argmax(probs, axis=1)[0])
        result = LABEL_MAP.get(prediction, "Unknown")

        return jsonify({"prediction": result}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/", methods=["GET"])
def home():
    return "✅ TradeChip Prediction API is running."


if __name__ == "__main__":
    app.run(port=5050, debug=True)

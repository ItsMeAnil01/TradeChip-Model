import xgboost as xgb
import joblib

# Load model from joblib file
model = joblib.load("models/trade_model.pkl")

# Convert to Booster (if it’s an XGBClassifier)
booster = model.get_booster()

# Save as JSON
booster.save_model("models/trade_model.json")

print("✅ Model saved as JSON successfully!")

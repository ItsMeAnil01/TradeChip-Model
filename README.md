# 📊 TradeChip Model + Prediction API

This project contains the machine learning model and REST API for predicting stock actions: **Buy**, **Hold**, or **Sell** based on technical indicators.

---

## ✅ Requirements

- Python 3.9 or above
- `pip` installed

---

## 🔧 Setup Instructions

### 1. Clone the repository

```bash
git clone https://github.com/ItsMeAnil01/TradeChip-Model.git
cd TradeChip-Model



2. Install dependencies
pip install -r requirements.txt



🧠 Train the Model
Make sure you have preprocessed stock CSV files inside:
data/processed/

run this:
python train_model.py
models/trade_model.pkl



🔍 Run Live Prediction (CLI)
Test real-time prediction using live stock data:
python predict_live.py


🌐 Run the Flask API
To start the API server:

cd api
python app.py

The API will be running at:
http://localhost:5050/predict



eg:
POST /predict

Body
{
  "symbol": "TCS.NS"
}



🏷️ Prediction Labels
0 → Sell

1 → Hold

2 → Buy\





📁 Project Structure
TradeChip-Model/
├── data/processed/           # Training CSV files
├── models/                   # Saved ML model
├── train_model.py            # Training script
├── predict_live.py           # Live prediction via yfinance
├── api/
│   └── app.py                # Flask API
├── requirements.txt
└── README.md


🧑‍💻 Author
Made with 💻 by @ItsMeAnil01


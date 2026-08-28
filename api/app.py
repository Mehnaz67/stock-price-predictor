import os
import time
import finnhub
from flask import Flask, render_template, request, jsonify

app = Flask(__name__, template_folder='../templates', static_folder='../static')

MY_KEY = "da88eapr01qo86cgdt9gda88eapr01qo86cgdta0"
API_KEY = os.environ.get("FINNHUB_API_KEY") or MY_KEY
finnhub_client = finnhub.Client(api_key=API_KEY)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json(silent=True)
        if data and 'ticker' in data:
            ticker = data['ticker'].upper()
        else:
            ticker = request.form.get('ticker', 'AAPL').upper()
        
        historical_dates = ["2026-01-01", "2026-01-02", "2026-01-03", "2026-01-04", "2026-01-05"]
        historical_prices = [175.0, 177.2, 176.5, 180.1, 181.5]
        future_dates = ["2026-01-06", "2026-01-07", "2026-01-08"]
        future_prices = [182.0, 183.5, 185.50]
        
        return jsonify({
            "ticker": ticker,
            "r2_score": 0.94,
            "rmse": 2.45,
            "historical_dates": historical_dates,
            "historical_prices": historical_prices,
            "future_dates": future_dates,
            "future_prices": future_prices
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
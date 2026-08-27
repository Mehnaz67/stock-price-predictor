from datetime import date, timedelta
import time
from flask import Flask, render_template, request, jsonify
import numpy as np
import pandas as pd
import finnhub
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

app = Flask(__name__)

finnhub_client = finnhub.Client(api_key="da84s01r01qo86cga720da84s01r01qo86cga72g")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json() or {}
    ticker = data.get('ticker', 'AAPL').strip().upper()
    days = int(data.get('days', 30))
    
    period_days = min(int(data.get('period', 365)), 365)

    end_date = date.today()
    start_date = end_date - timedelta(days=period_days)
    
    from_ts = int(time.mktime(start_date.timetuple()))
    to_ts = int(time.mktime(end_date.timetuple()))

    try:
        res = finnhub_client.stock_candles(ticker, 'D', from_ts, to_ts)
        if not res or res.get('s') != 'ok':
            return jsonify({'error': 'Invalid ticker symbol or no data found.'}), 400
        
        df = pd.DataFrame({
            'Close': res['c']
        }, index=pd.to_datetime(res['t'], unit='s'))

    except Exception as e:
        return jsonify({'error': 'Invalid ticker symbol or no data found.'}), 400

    df = df.dropna()
    df['Days'] = np.arange(len(df))
    
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    df.dropna(inplace=True)
    
    if df.empty:
        return jsonify({'error': 'Insufficient market data to compute moving averages.'}), 400

    X = df[['Days', 'SMA_20', 'SMA_50']]
    y = df['Close']
    model = LinearRegression()
    model.fit(X, y)
    y_pred = model.predict(X)
    r2 = r2_score(y, y_pred)
    rmse = np.sqrt(mean_squared_error(y, y_pred))

    last_day = df['Days'].iloc[-1]
    last_sma_20 = df['SMA_20'].iloc[-1]
    last_sma_50 = df['SMA_50'].iloc[-1]
    future_days = np.array([last_day + i for i in range(1, days + 1)]).reshape(-1, 1)
    future_feature = np.column_stack((future_days, np.full(days, last_sma_20), np.full(days, last_sma_50)))
    future_preds = model.predict(future_feature)
    
    historical_dates = df.index.strftime('%Y-%m-%d').tolist()
    historical_prices = df['Close'].values.flatten().tolist()
    
    future_dates = [
        (end_date + timedelta(days=int(i))).strftime('%Y-%m-%d')
        for i in range(1, days + 1)
    ]
    
    return jsonify({
        'ticker': ticker,
        'historical_dates': historical_dates,
        'historical_prices': historical_prices,
        'future_dates': future_dates,
        'future_prices': future_preds.flatten().tolist(),
        'r2_score': round(float(r2), 4),
        'rmse': round(float(rmse), 4),
    })

if __name__ == '__main__':
    app.run()
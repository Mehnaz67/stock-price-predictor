from datetime import date,timedelta
from flask import Flask,render_template,request,jsonify
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,r2_score
app = Flask(__name__)
@app.route('/')
def index():
    return render_template('index.html')
@app.route('/predict',methods = ['POST'])
def predict():
    data = request.get_json() or {}
    ticker = data.get('ticker', 'AAPL').strip().upper()
    days = int(data.get('days',30))
    period_days = int(data.get('period', 730))

    end_date = date.today()
    fetch_days = max(period_days, 365)
    start_date = end_date - timedelta(days = fetch_days)
    df = yf.download(ticker, start = start_date, end = end_date, progress = False)
    if df.empty:
        return jsonify({'error': 'Invalid ticker symbol or no data found.'}),400
    
    if isinstance(df.columns, pd.MultiIndex):
        df = df['Close']
    else:
        df = df[['Close']]
    if isinstance(df, pd.DataFrame):
        df = df.iloc[:, 0]
    df = df.to_frame(name='Close').dropna()
    df['Days'] = np.arange(len(df))
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    df['SMA_200'] = df['Close'].rolling(window=200).mean()
    df.dropna(inplace=True)
    if df.empty:
        return jsonify({'error': 'Insufficient market data to compute 200-day moving averages.'}), 400

    X =df[['Days', 'SMA_50', 'SMA_200']]
    y = df['Close']
    model = LinearRegression()
    model.fit(X,y)
    y_pred = model.predict(X)
    r2 = r2_score(y, y_pred)
    rmse = np.sqrt(mean_squared_error(y, y_pred))

    last_day = df['Days'].iloc[-1]
    last_sma_50 = df['SMA_50'].iloc[-1]
    last_sma_200 = df['SMA_200'].iloc[-1]
    future_days = np.array([last_day + i for i in range(1,days+1)]).reshape(-1,1)
    future_feature = np.column_stack((future_days, np.full(days,last_sma_50), np.full(days,last_sma_200)))
    future_preds = model.predict(future_feature)
    historical_dates = df.index.strftime('%Y-%m-%d').tolist()
    historical_prices = df['Close'].values.flatten().tolist()
    historical_dates = historical_dates[-period_days:]
    historical_prices = historical_prices[-period_days:]
    future_dates = [
        (end_date + timedelta(days =int(i))).strftime('%Y-%m-%d')
        for i in range(1,days+1)
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
    app.run(debug = True)


    

import ccxt
import pandas as pd
from datetime import datetime, timedelta
import os

# Initialize Binance exchange
exchange = ccxt.binance()

# Function to fetch all klines for a given symbol
def fetch_klines(symbol, timeframe, since, limit=1000):
    klines = []
    while True:
        data = exchange.fetch_ohlcv(symbol, timeframe, since, limit)
        if not data:
            break
        klines.extend(data)
        since = data[-1][0] + 1
    df = pd.DataFrame(klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    return df


# Function to fetch and store kline data for eligible USDT trading pairs
def fetch_all_klines(timeframe, since, only, limit=1000):
    markets = exchange.load_markets()
    usdt_pairs = [symbol for symbol in markets if markets[symbol]['symbol'].endswith('/USDT') and markets[symbol]['info']['status'] == 'TRADING']
    usdt_pairs = sorted(usdt_pairs)

    if not os.path.exists('kline_data'):
        os.makedirs('kline_data')
    
    for symbol in usdt_pairs:
        if len(only) > 0 and symbol not in only:
            continue
        kline_data = fetch_klines(symbol, timeframe, since, limit)
        if not kline_data.empty:
            file_path = f'kline_data/{symbol.replace("/", "_")}.csv'
            kline_data.to_csv(file_path, index=False)
            print(f'Kline data saved to {file_path}')

length = 3000
only = []
timeframe = '1d'  # 1-day intervals
since = exchange.parse8601((datetime.now() - timedelta(days=length)).isoformat())

# Fetch and store kline data for eligible USDT trading pairs
fetch_all_klines(timeframe, since, only)

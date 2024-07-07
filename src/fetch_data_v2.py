import ccxt
import pandas as pd
import time
import json
import os
from datetime import datetime, timedelta

# bitmart - does not support 12h timeframe
# bitmex {"error":{"message":"binSize is invalid.","name":"ValidationError"}} - 12h not supported ?
# mexc for AES/USDT: mexc {"msg":"Invalid interval.","code":-1121,"_extend":null} - 12h not supported ?
# upbit {"error":{"message":"no Route matched with those values","name":"not_found"}} - 12h not supported ?
# kraken for EURT/USDT: kraken {"error":["EGeneral:Invalid arguments"]} - 12h not supported ?
exchanges = [
'binance', 'bitmex', 'bybit', 'bingx', 
'bitmart', 'bitget',
'coinbase', 'coinbaseinternational', 'coinex', 'cryptocom', 
'gate', 'kucoin', 'mexc', 'okx', 'upbit', 'kraken',
'bitstamp', 'bithumb', 'gemini', 'bitfinex',
]

# Timeframe for OHLCV data
timeframe = '4h'

# File to store fetched symbols
fetched_symbols_file = 'fetched_symbols.json'
# File to log errors
error_log_file = 'error_log.txt'

# Load seen coins from the file if it exists
seen_coins = set()
if os.path.exists(fetched_symbols_file):
    with open(fetched_symbols_file, 'r') as f:
        seen_coins = set(json.load(f))

def get_sleep_period(exchange_id: str) -> float:
    if exchange_id == 'bitfinex':
        return 4
    return 0.5

def format_symbol(symbol: str, exchange_id: str) -> str:
    return symbol

def log_error(message):
    """Log an error message to the error log file."""
    with open(error_log_file, 'a') as f:
        f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - {message}\n")

def get_since(exchange_id: str)->str:
    if exchange_id == 'bitget' or exchange_id == 'bingx':
        current_date = datetime.utcnow()
        one_year_ago = current_date - timedelta(days=365)
        formatted_date = one_year_ago.strftime('%Y-%m-%dT%H:%M:%SZ')
        return formatted_date
    return '2017-01-01T00:00:00Z'

def fetch_all_ohlcv(exchange, symbol, timeframe):
    """Fetch all available OHLCV data from a specific exchange."""
    all_data = []
    since = exchange.parse8601(get_since(exchange.id))
    while True:
        try:
            data = exchange.fetch_ohlcv(format_symbol(symbol, exchange.id), timeframe, since)
            if not data:
                break
            all_data.extend(data)
            since = data[-1][0] + 1  # move to the next timestamp to avoid duplication
            time.sleep(get_sleep_period(exchange.id))  # respect rate limit
        except Exception as e:
            log_error(f"Error fetching data from {exchange.id} for {symbol}: {e}")
            break
    return all_data

def save_to_csv(df, coin_name):
    """Save DataFrame to CSV file."""
    filename = f"{coin_name}.csv"
    df.to_csv(filename, index=False)
    print(f"Data saved to {filename}")

def fetch_and_save_data(exchange, symbol, coin_name):
    """Fetch and save data for a specific symbol if it hasn't been seen before."""
    global seen_coins
    if coin_name not in seen_coins:
        print(f"Fetching data for {coin_name} from {exchange.id}...")
        data = fetch_all_ohlcv(exchange, symbol, timeframe)
        if data and len(data) >= 500:  # Save only if there are at least 500 candles
            df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            save_to_csv(df, coin_name)
            seen_coins.add(coin_name)
            # Update the fetched symbols file
            with open(fetched_symbols_file, 'w') as f:
                json.dump(list(seen_coins), f)
        else:
            print(f'Not enough data of {coin_name}.')
    else:
        print(f"Skipping {coin_name}, already fetched.")

def main():
    for exchange_id in exchanges:
        exchange_class = getattr(ccxt, exchange_id)
        exchange = exchange_class()
        try:
            markets = exchange.load_markets()
            for symbol in markets.keys():
                if symbol.endswith('USDT'):
                    base_asset = markets[symbol]['base'].lower()
                    fetch_and_save_data(exchange, symbol, base_asset)
        except Exception as e:
            log_error(f"Error loading markets for {exchange_id}: {e}")

if __name__ == "__main__":
    main()

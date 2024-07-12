import ccxt
import pandas as pd
import time
import datetime
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from data_preparation import (
    FeaturesConfig, 
    TrainingDataConfig,
    add_features,
    normalize_data,
    load_scaler,
)
from modeling import load_model
from utils import NON_FEATURE_COLUMNS
from evaluation import create_prediction_data

features_config = FeaturesConfig(
    sma_lengths=[],
    ema_lengths=[],
    rsi_lengths=[],
    stoch_k_lengths=[],
    stoch_d_lengths=[],
    stoch_k_smooth=[],
    mfi_lengths=[],
    adx_lengths=[],
    atr_lengths=[],
    std_lengths=[],
    bb_lengths=[],
    bb_stds=[],
    ichimoku=[],
)
training_data_config = TrainingDataConfig(
    sequence_length=25,
    future_candles_count=6,
    pct_increase=2,
)

CANDLE_LIMIT = 30
PREDICTION_THRESHOLD = 0.7
PROFIT_THRESHOLD = 0.02
MAX_CANDLES = 6
SLEEP_INTERVAL = 1
MONITOR_INTERVAL = 10
TRADES_FILE = './src/backtest/trades.csv'
TIMEFRAME = '4h'

MODEL_PATH = './src/backtest/model.h5'
model = load_model(model_path=MODEL_PATH)

SCALER_PATH = './src/backtest/scaler.pkl'
scaler = load_scaler(scaler_file=SCALER_PATH)

exchange = ccxt.binance()

markets = exchange.load_markets()
active_usdt_coins = [coin for coin in markets if markets[coin]['active'] and coin.endswith('/USDT')]

def log_error(message):
    print(f"{datetime.datetime.now()} - ERROR - {message}\n")
    with open("./src/backtest/trading_errors.log", "a") as log_file:
        log_file.write(f"{datetime.datetime.now()} - ERROR - {message}\n")

def process_data(candles):
    df = pd.DataFrame(candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    
    df = add_features(dfs=[df], config=features_config)[0]
    feature_columns = [col for col in df if col not in NON_FEATURE_COLUMNS]

    df = normalize_data(dfs=[df], scaler=scaler, columns_to_normalize=feature_columns)[0]

    sequences = create_prediction_data(
        df=df, 
        feature_columns=feature_columns, 
        sequence_length=training_data_config.sequence_length,
    )

    return sequences

def should_enter(data):
    prediction = model.predict(x=data, verbose=0)
    return prediction[0][0] > PREDICTION_THRESHOLD

try:
    trades_df = pd.read_csv(TRADES_FILE)
except FileNotFoundError:
    trades_df = pd.DataFrame(columns=[
        'coin', 
        'buy_price', 
        'buy_date', 
        'sell_price', 
        'sell_date', 
        'last_close_price', 
        'opening_candle_date',
        'profit_percentage', 
        'percentage_change', 
        'close_reason', 
        'candles_passed', 
        'last_monitor_date',
    ])


def save_trades():
    if len(trades_df) > 0:
        trades_df.to_csv(TRADES_FILE, index=False)


def start_trading():
    global trades_df
    while True:
        try:
            for coin in active_usdt_coins:
                # Check if there is already an open trade for this coin
                if not trades_df[(trades_df['coin'] == coin) & (trades_df['sell_date'].isnull())].empty:
                    continue
                
                candles = exchange.fetch_ohlcv(coin, timeframe=TIMEFRAME, limit=CANDLE_LIMIT)
                processed_data = process_data(candles)           
                is_enter = should_enter(processed_data)
                
                current_price = candles[-1][4]
                last_close_price = candles[-2][4]
                opening_candle_date = datetime.datetime.fromtimestamp(candles[-1][0] / 1000, datetime.UTC).strftime("%Y-%m-%d %H:%M:%S")
                is_below_prev_close = current_price < last_close_price
                
                if is_enter and is_below_prev_close:
                    current_date = datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%d %H:%M:%S")
                    
                    new_trade = {
                        'coin': coin,
                        'buy_price': current_price,
                        'sell_price': 0.0,
                        'buy_date': current_date,
                        'sell_date': None,
                        'last_close_price': last_close_price,
                        'profit_percentage': 0.0,
                        'percentage_change': 0.0,
                        'close_reason': None,
                        'candles_passed': 0,
                        'opening_candle_date': opening_candle_date,
                        'last_monitor_date': None,
                    }
                    trades_df.loc[len(trades_df)] = new_trade
                    
                    save_trades()
                
                time.sleep(SLEEP_INTERVAL)

        except Exception as e:
            log_error(f"Error in start_trading: {e}")
        
        time.sleep(60)


def monitor_trades():
    global trades_df
    while True:
        try:
            current_time = datetime.datetime.now(datetime.UTC)
            for idx, trade in trades_df.iterrows():
                if not pd.isnull(trade['sell_date']):
                    continue
                
                coin = trade['coin']
                entry_price = trade['buy_price']
                last_close_price = trade['last_close_price']
                entry_time = datetime.datetime.strptime(trade['buy_date'], "%Y-%m-%d %H:%M:%S")
                entry_time = entry_time.replace(tzinfo=datetime.timezone.utc).timestamp() * 1000
                
                candles = exchange.fetch_ohlcv(coin, timeframe=TIMEFRAME, limit=MAX_CANDLES)

                candles_since_entry = [candle for candle in candles if candle[0] > entry_time]
                
                current_price = candles[-1][4]
                profit_percentage = (current_price - entry_price) / entry_price * 100
                percentage_change = (current_price - last_close_price) / last_close_price * 100
                candles_passed = len(candles_since_entry) + 1
                last_monitor_date = current_time.strftime("%Y-%m-%d %H:%M:%S")
                
                trades_df.at[idx, 'candles_passed'] = candles_passed
                trades_df.at[idx, 'last_monitor_date'] = last_monitor_date
                trades_df.at[idx, 'percentage_change'] = round(percentage_change, 2)
                trades_df.at[idx, 'profit_percentage'] = round(profit_percentage, 2)
                
                if current_price > (last_close_price * (1 + PROFIT_THRESHOLD)):
                    trades_df.at[idx, 'sell_price'] = round(current_price, 2)
                    trades_df.at[idx, 'sell_date'] = last_monitor_date
                    trades_df.at[idx, 'close_reason'] = 'Profit Reached'
                elif candles_passed > MAX_CANDLES:
                    trades_df.at[idx, 'sell_price'] = round(current_price, 2)
                    trades_df.at[idx, 'sell_date'] = last_monitor_date
                    trades_df.at[idx, 'close_reason'] = 'Candles Exceeded'

            save_trades()
            time.sleep(MONITOR_INTERVAL)

        except Exception as e:
            log_error(f"Error in monitor_trades: {e}")


# Run trading and monitoring in separate threads
import threading
trading_thread = threading.Thread(target=start_trading)
monitor_thread = threading.Thread(target=monitor_trades)

trading_thread.start()
monitor_thread.start()

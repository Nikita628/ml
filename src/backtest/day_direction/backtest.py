import ccxt
import pandas as pd
import time
import datetime
import sys
import os
import threading

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from data_preparation import (
    FeaturesConfig, 
    TrainingDataConfig,
    add_features,
    normalize_data,
    load_scaler,
)
from modeling import load_model
from utils import NON_FEATURE_COLUMNS, PredictionType
from evaluation import create_prediction_data

class Config:
    FEATURES_CONFIG = FeaturesConfig(
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
    
    TRAINING_DATA_CONFIG = TrainingDataConfig(
        sequence_length=25,
        prediction_type=PredictionType.next_close_direction
    )

    CANDLE_LIMIT = 30
    DECREASE_RELATIVE_TO_LAST_CLOSE = 0.01
    TRADE_MAX_CANDLES = 1
    TIMEFRAME = '1d'
    
    class Paths:
        TRADES_FILE = './src/backtest/day_direction/trades.csv'
        POTENTIAL_TRADES_FILE = './src/backtest/day_direction/potential_trades.csv'
        MODEL_PATH = './src/backtest/day_direction/model.h5'
        SCALER_PATH = './src/backtest/day_direction/scaler.pkl'
        ERROR_LOG_FILE_PATH = "./src/backtest/day_direction/trading_errors.log"
    
    class Intervals:
        COIN_CHECK = 1
        TRADES_MONITOR = 10
        TRADING = 60
        POTENTIAL_TRADES_MONITOR = 120
    
    class ProbabilityThresholds:
        TRADE = 0.75
        POTENTIAL_TRADE = 0.7

config = Config()

model = load_model(model_path=config.Paths.MODEL_PATH)
scaler = load_scaler(scaler_file=config.Paths.SCALER_PATH)

exchange = ccxt.binance()

def log_error(message):
    print(f"{datetime.datetime.now()} - ERROR - {message}\n")
    with open(config.Paths.ERROR_LOG_FILE_PATH, "a") as log_file:
        log_file.write(f"{datetime.datetime.now()} - ERROR - {message}\n")

def initialize_trades():
    global trades_df
    try:
        trades_df = pd.read_csv(config.Paths.TRADES_FILE)
        trades_df.rename(columns={"last_close_price": "previous_close_price"}, inplace=True)
    except FileNotFoundError:
        trades_df = pd.DataFrame(columns=[
            'coin', 
            'buy_price', 
            'buy_date', 
            'sell_price', 
            'sell_date', 
            'previous_close_price', 
            'opening_candle_date',
            'profit_percentage', 
            'percentage_change', 
            'close_reason', 
            'candles_passed', 
            'last_monitor_date',
        ])

def initialize_potential_trades():
    global potential_trades_df
    try:
        potential_trades_df = pd.read_csv(config.Paths.POTENTIAL_TRADES_FILE)
    except FileNotFoundError:
        potential_trades_df = pd.DataFrame(columns=[
            'coin', 
            'opening_candle_date', 
            'probability',
            'previous_close_price',
            'close_price',
            'close_direction',
            'last_monitor_date',
            'potential_profit_%'
        ])

def initialize_active_usdt_symbols():
    global active_usdt_coins
    markets = exchange.load_markets()
    active_usdt_coins = [coin for coin in markets if markets[coin]['active'] and coin.endswith('/USDT')]

def save_trades():
    if len(trades_df) > 0:
        trades_df.to_csv(config.Paths.TRADES_FILE, index=False)

def save_potential_trades():
    if len(potential_trades_df) > 0:
        potential_trades_df.to_csv(config.Paths.POTENTIAL_TRADES_FILE, index=False)

def create_prediction_sequence(candles):
    df = pd.DataFrame(candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    
    df = add_features(dfs=[df], config=config.FEATURES_CONFIG)[0]
    feature_columns = [col for col in df if col not in NON_FEATURE_COLUMNS]

    df = normalize_data(dfs=[df], scaler=scaler, columns_to_normalize=feature_columns)[0]

    sequences = create_prediction_data(
        df=df, 
        feature_columns=feature_columns, 
        sequence_length=config.TRAINING_DATA_CONFIG.sequence_length,
    )

    return sequences

def log_potential_trade(coin, opening_candle_date, probability, previous_close_price):
    global potential_trades_df

    if not ((potential_trades_df['coin'] == coin) & (potential_trades_df['opening_candle_date'] == opening_candle_date)).any():
        new_entry = {
            'coin': coin,
            'opening_candle_date': opening_candle_date,
            'probability': round(probability, 2),
            'previous_close_price': previous_close_price,
            'close_price': None,
            'close_direction': None,
            'last_monitor_date': None,
            'potential_profit_%': None
        }
        potential_trades_df.loc[len(potential_trades_df)] = new_entry
        save_potential_trades()

def predict(data):
    prediction = model.predict(x=data, verbose=0)
    y_pred = prediction[0][0]
    return y_pred

def start_trading():
    global trades_df
    while True:
        try:
            for coin in active_usdt_coins:
                # Check if there is already an open trade for this coin
                if not trades_df[(trades_df['coin'] == coin) & (trades_df['sell_date'].isnull())].empty:
                    continue
                
                candles = exchange.fetch_ohlcv(coin, timeframe=config.TIMEFRAME, limit=config.CANDLE_LIMIT)
                processed_data = create_prediction_sequence(candles)
                opening_candle_date = datetime.datetime.fromtimestamp(candles[-1][0] / 1000, datetime.UTC).strftime("%Y-%m-%d %H:%M:%S")      
                prediction_probability = predict(processed_data)
                current_price = candles[-1][4]
                previous_close_price = candles[-2][4]

                # current price dipped below last close by a certain %
                is_below_prev_close = (current_price - previous_close_price) / previous_close_price < (-config.DECREASE_RELATIVE_TO_LAST_CLOSE) 
                
                if prediction_probability > config.ProbabilityThresholds.POTENTIAL_TRADE:
                    log_potential_trade(coin, opening_candle_date, prediction_probability, previous_close_price)
                
                if prediction_probability > config.ProbabilityThresholds.TRADE and is_below_prev_close:
                    current_date = datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%d %H:%M:%S")
                    
                    new_trade = {
                        'coin': coin,
                        'buy_price': current_price,
                        'sell_price': 0.0,
                        'buy_date': current_date,
                        'sell_date': None,
                        'previous_close_price': previous_close_price,
                        'profit_percentage': 0.0,
                        'percentage_change': 0.0,
                        'close_reason': None,
                        'candles_passed': 0,
                        'opening_candle_date': opening_candle_date,
                        'last_monitor_date': None,
                    }
                    trades_df.loc[len(trades_df)] = new_trade
                    
                    save_trades()
                
                time.sleep(config.Intervals.COIN_CHECK)

        except Exception as e:
            log_error(f"Error in start_trading: {e}")
        
        time.sleep(config.Intervals.TRADING)

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
                previous_close_price = trade['previous_close_price']
                entry_time = datetime.datetime.strptime(trade['buy_date'], "%Y-%m-%d %H:%M:%S")
                entry_time = entry_time.replace(tzinfo=datetime.timezone.utc).timestamp() * 1000
                
                candles = exchange.fetch_ohlcv(coin, timeframe=config.TIMEFRAME, limit=config.TRADE_MAX_CANDLES)

                candles_since_entry = [candle for candle in candles if candle[0] > entry_time]
                
                current_price = candles[-1][4]
                profit_percentage = (current_price - entry_price) / entry_price * 100
                percentage_change = (current_price - previous_close_price) / previous_close_price * 100
                candles_passed = len(candles_since_entry) + 1
                last_monitor_date = current_time.strftime("%Y-%m-%d %H:%M:%S")
                
                trades_df.at[idx, 'candles_passed'] = candles_passed
                trades_df.at[idx, 'last_monitor_date'] = last_monitor_date
                trades_df.at[idx, 'percentage_change'] = round(percentage_change, 2)
                trades_df.at[idx, 'profit_percentage'] = round(profit_percentage, 2)
                
                if current_price > previous_close_price:
                    trades_df.at[idx, 'sell_price'] = round(current_price, 2)
                    trades_df.at[idx, 'sell_date'] = last_monitor_date
                    trades_df.at[idx, 'close_reason'] = 'Profit Reached'
                elif candles_passed > config.TRADE_MAX_CANDLES:
                    trades_df.at[idx, 'sell_price'] = round(current_price, 2)
                    trades_df.at[idx, 'sell_date'] = last_monitor_date
                    trades_df.at[idx, 'close_reason'] = 'Candles Exceeded'

            save_trades()
            time.sleep(config.Intervals.TRADES_MONITOR)

        except Exception as e:
            log_error(f"Error in monitor_trades: {e}")

def monitor_potential_trades():
    global potential_trades_df
    while True:
        try:
            current_time = datetime.datetime.now(datetime.UTC)
            
            for idx, trade in potential_trades_df[potential_trades_df['close_price'].isnull()].iterrows():
                coin = trade['coin']
                opening_candle_date = trade['opening_candle_date']
                previous_close_price = trade['previous_close_price']
                
                # Fetch the candle with the opening_candle_date
                candles = exchange.fetch_ohlcv(coin, timeframe=config.TIMEFRAME, limit=config.CANDLE_LIMIT)
                candle_dates = [datetime.datetime.fromtimestamp(candle[0] / 1000, datetime.UTC).strftime("%Y-%m-%d %H:%M:%S") for candle in candles]
                
                try:
                    candle_idx = candle_dates.index(opening_candle_date)
                except ValueError:
                    potential_trades_df.at[idx, 'last_monitor_date'] = current_time.strftime("%Y-%m-%d %H:%M:%S")
                    continue

                if candle_idx + 1 < len(candles):  # Ensure there is a next candle
                    closing_candle = candles[candle_idx]
                    next_candle = candles[candle_idx + 1]
                    closing_price = closing_candle[4]
                    close_direction = 1 if next_candle[4] > previous_close_price else 0
                    potential_profit_percentage = ((closing_price - previous_close_price) / previous_close_price) * 100

                    potential_trades_df.at[idx, 'close_price'] = closing_price
                    potential_trades_df.at[idx, 'close_direction'] = close_direction
                    potential_trades_df.at[idx, 'potential_profit_%'] = round(potential_profit_percentage, 2)
                
                potential_trades_df.at[idx, 'last_monitor_date'] = current_time.strftime("%Y-%m-%d %H:%M:%S")

            save_potential_trades()

            time.sleep(config.Intervals.POTENTIAL_TRADES_MONITOR)
        
        except Exception as e:
            log_error(f"Error in monitor_potential_trades: {e}")

# Initialize dataframes and active USDT coins
initialize_trades()
initialize_potential_trades()
initialize_active_usdt_symbols()

# Run trading and monitoring in separate threads
trading_thread = threading.Thread(target=start_trading)
monitor_thread = threading.Thread(target=monitor_trades)
monitor_potential_trades_thread = threading.Thread(target=monitor_potential_trades)

trading_thread.start()
monitor_thread.start()
monitor_potential_trades_thread.start()

import ccxt
import pandas as pd
import time
import datetime
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

# Constants
CANDLE_LIMIT = 30
PREDICTION_THRESHOLD = 0.7
PROFIT_THRESHOLD = 0.02  # Example: 2%
MAX_CANDLES = 6  # Example: 6 candles to evaluate profit
SLEEP_INTERVAL = 2  # Sleep interval between API calls to respect rate limits
MONITOR_INTERVAL = 10  # Interval to monitor trades
TRADES_FILE = 'trades.csv'
TIMEFRAME = '4h'  # 4-hour timeframe

# Load Keras model
MODEL_PATH = './src/model.h5'
model = load_model(model_path=MODEL_PATH)

# Load StandardScaler
SCALER_PATH = './src/scaler.pkl'
scaler = load_scaler(scaler_file=SCALER_PATH)

# Initialize Binance API
exchange = ccxt.binance()

# Fetch the list of all active tradable coins against USDT
markets = exchange.load_markets()
active_usdt_coins = [coin for coin in markets if markets[coin]['active'] and coin.endswith('/USDT')]

# Function stubs
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

def predict(data):
    prediction = model.predict(data)
    return prediction[0][0]

try:
    trades_df = pd.read_csv(TRADES_FILE)
except FileNotFoundError:
    trades_df = pd.DataFrame(columns=[
        'coin', 'buy_price', 'last_close_price', 'sell_price', 'buy_date', 'sell_date', 'predicted', 'actual', 'profit_percentage', 'percentage_change'
    ])

def save_trades():
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
                prediction = predict(processed_data)
                
                current_price = candles[-1][4]
                last_close_price = candles[-2][4]
                profit_condition = current_price < (last_close_price * (1 + PROFIT_THRESHOLD / 2))
                
                if prediction > PREDICTION_THRESHOLD and profit_condition:
                    current_date = datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
                    
                    new_trade = {
                        'coin': coin,
                        'buy_price': current_price,
                        'last_close_price': last_close_price,
                        'sell_price': 0,
                        'buy_date': current_date,
                        'sell_date': None,
                        'predicted': 1,
                        'actual': 0,
                        'profit_percentage': 0,
                        'percentage_change': 0
                    }
                    trades_df.loc[len(trades_df)] = new_trade
                    
                    save_trades()
                
                # Sleep to respect rate limits
                time.sleep(SLEEP_INTERVAL)

        except Exception as e:
            print(f"Error: {e}")
        
        time.sleep(60)  # Wait 1 minute before next iteration

# Monitor trades
def monitor_trades():
    global trades_df
    while True:
        current_time = datetime.datetime.utcnow()
        for index, trade in trades_df.iterrows():
            if not pd.isnull(trade['sell_date']):
                continue
            
            coin = trade['coin']
            entry_price = trade['buy_price']
            last_close_price = trade['last_close_price']
            entry_time = datetime.datetime.strptime(trade['buy_date'], "%Y-%m-%d %H:%M:%S")
            entry_time = entry_time.replace(tzinfo=datetime.timezone.utc).timestamp() * 1000
            
            # Fetch latest candles since trade entry
            candles = exchange.fetch_ohlcv(coin, timeframe=TIMEFRAME, limit=MAX_CANDLES)

            # Check the number of candles since the trade was opened
            candles_since_entry = [candle for candle in candles if candle[0] > entry_time]
            
            current_price = candles[-1][4]
            profit_percentage = (current_price - entry_price) / entry_price
            percentage_change = (current_price - last_close_price) / last_close_price * 100
            
            if current_price > (last_close_price * (1 + PROFIT_THRESHOLD)):
                trades_df.at[index, 'sell_price'] = current_price
                trades_df.at[index, 'sell_date'] = current_time.strftime("%Y-%m-%d %H:%M:%S")
                trades_df.at[index, 'actual'] = 1
                trades_df.at[index, 'profit_percentage'] = profit_percentage * 100
                trades_df.at[index, 'percentage_change'] = percentage_change
                save_trades()
            elif len(candles_since_entry) + 1 > MAX_CANDLES:
                # Close the trade if MAX_CANDLES have passed and profit threshold is not reached
                trades_df.at[index, 'sell_price'] = current_price  # Closing at the last candle's close price
                trades_df.at[index, 'sell_date'] = current_time.strftime("%Y-%m-%d %H:%M:%S")
                trades_df.at[index, 'actual'] = 0
                trades_df.at[index, 'profit_percentage'] = profit_percentage * 100
                trades_df.at[index, 'percentage_change'] = percentage_change
                save_trades()

        time.sleep(MONITOR_INTERVAL)  # Wait before next monitoring iteration

# Run trading and monitoring in separate threads
import threading
trading_thread = threading.Thread(target=start_trading)
monitor_thread = threading.Thread(target=monitor_trades)

trading_thread.start()
monitor_thread.start()

from typing import List, Any
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import os
import pickle
import pandas_ta as ta
from utils import PredictionType
from numpy import ndarray
from collections import Counter

def calculate_label_percentages(labels) -> dict[Any, float]:
    total_count = len(labels)
    label_counts = Counter(labels)
    label_percentages = {label: (count / total_count) * 100 for label, count in label_counts.items()}
    
    return label_percentages


REQUIRED_COLUMNS = ['timestamp', 'open', 'close', 'high', 'low', 'volume']

def load_data(directory_path: str, min_len=500) -> List[pd.DataFrame]:
    dataframes = []
    for filename in os.listdir(directory_path):
        if filename.endswith('.csv'):
            file_path = os.path.join(directory_path, filename)
            df = pd.read_csv(file_path, parse_dates=['timestamp'])

            if not all(column in df.columns for column in REQUIRED_COLUMNS):
                raise ValueError(f"File {filename} is missing one or more required columns: {REQUIRED_COLUMNS}")
        
            if len(df) < min_len:
                continue

            df['file'] = filename
            dataframes.append(df)
    return dataframes


class FeaturesConfig:
    def __init__(
            self,
            sma_lengths=[2, 5, 10, 20, 50, 100, 200],
            ema_lengths=[2, 5, 10, 20, 50],
            rsi_lengths=[2, 5, 10, 15, 20],
            stoch_lengths=[2, 5, 10, 15, 20],
            stoch_d_lengths=[3, 5, 7, 9],
            mfi_lengths=[2, 5, 10, 15, 20],
            adx_lengths=[2, 5, 10, 15, 20],
            atr_lengths=[2, 5, 10, 15, 20],
            std_lengths=[2, 5, 10, 15, 20],
            is_pct=True,
            is_slopes=False,
            is_order=False,
        ):
        self.sma_lengths = sma_lengths
        self.ema_lengths = ema_lengths
        self.rsi_lengths = rsi_lengths
        self.stoch_lengths = stoch_lengths
        self.stoch_ds = stoch_d_lengths
        self.mfi_lengths = mfi_lengths
        self.adx_lengths = adx_lengths
        self.atr_lengths = atr_lengths
        self.std_lengths = std_lengths
        self.is_pct = is_pct
        self.is_slopes = is_slopes
        self.is_order = is_order

    def __str__(self):
        return (
            f"FeaturesConfig(\n"
            f"  sma_lengths={self.sma_lengths},\n"
            f"  ema_lengths={self.ema_lengths},\n"
            f"  rsi_lengths={self.rsi_lengths},\n"
            f"  stoch_lengths={self.stoch_lengths},\n"
            f"  stoch_ds={self.stoch_ds},\n"
            f"  mfi_lengths={self.mfi_lengths},\n"
            f"  adx_lengths={self.adx_lengths},\n"
            f"  atr_lengths={self.atr_lengths},\n"
            f"  std_lengths={self.std_lengths}\n"
            f"  is_pct={self.is_pct}\n"
            f"  is_slopes={self.is_slopes}\n"
            f"  is_order={self.is_order}\n"
            f")"
        )


def add_features(dfs: List[pd.DataFrame], config: FeaturesConfig) -> List[pd.DataFrame]:
    result = []

    for df in dfs:
        new_columns = {}

        # tech indicators
        for length in config.sma_lengths:
            new_columns[f'SMA_{length}'] = ta.sma(df['close'], length=length)

        for length in config.ema_lengths:
            new_columns[f'EMA_{length}'] = ta.ema(df['close'], length=length)

        for length in config.rsi_lengths:
            new_columns[f'RSI_{length}'] = ta.rsi(df['close'], length=length)

        for length in config.atr_lengths:
            atr = ta.atr(df['high'], df['low'], df['close'], length=length)
            new_columns[f'ATR_{length}_pct'] = (atr / df['close'])

        for length in config.mfi_lengths:
            new_columns[f'MFI_{length}'] = ta.mfi(df['high'], df['low'], df['close'], df['volume'], length=length)

        for length in config.stoch_lengths:
            for d in config.stoch_ds:
                stoch = ta.stoch(df['high'], df['low'], df['close'], k=length, d=d)
                new_columns[f'STOCHk_{length}_{d}'] = stoch[f'STOCHk_{length}_{d}_3']
                new_columns[f'STOCHd_{length}_{d}'] = stoch[f'STOCHd_{length}_{d}_3']
        
        for length in config.adx_lengths:
            adx = ta.adx(df['high'], df['low'], df['close'], length=length)
            new_columns[f'ADX_{length}'] = adx[f'ADX_{length}']
            new_columns[f'DMP_{length}'] = adx[f'DMP_{length}']
            new_columns[f'DMN_{length}'] = adx[f'DMN_{length}']

        # custom features
        for length in config.std_lengths:
            new_columns[f'STD_{length}'] = df['close'].rolling(window=length).std()

        if config.is_pct:
            new_columns['close_pct_change'] = df['close'].pct_change(periods=1)
            new_columns['volume_pct_change'] = df['volume'].pct_change(periods=1)
            new_columns['pct_diff_close_open'] = ((df['close'] - df['open']) / df['open'])
            new_columns['pct_diff_close_high'] = ((df['close'] - df['high']) / df['high'])
            new_columns['pct_diff_close_low'] = ((df['close'] - df['low']) / df['low'])
            for length in config.sma_lengths:
                new_columns[f'pct_diff_close_SMA_{length}'] = ((df['close'] - new_columns[f'SMA_{length}']) / new_columns[f'SMA_{length}'])
            for length in config.ema_lengths:
                new_columns[f'pct_diff_close_EMA_{length}'] = ((df['close'] - new_columns[f'EMA_{length}']) / new_columns[f'EMA_{length}'])

        new_columns_df = pd.DataFrame(new_columns)

        dependent_columns = {}

        if config.is_slopes:
            for length in config.sma_lengths:
                dependent_columns[f'SMA_{length}_slope'] = (new_columns_df[f'SMA_{length}'] > new_columns_df[f'SMA_{length}'].shift(1)).astype(int)
                dependent_columns[f'SMA_{length}_below_close'] = (df['close'] > new_columns_df[f'SMA_{length}']).astype(int)

            for length in config.ema_lengths:
                dependent_columns[f'EMA_{length}_slope'] = (new_columns_df[f'EMA_{length}'] > new_columns_df[f'EMA_{length}'].shift(1)).astype(int)
                dependent_columns[f'EMA_{length}_below_close'] = (df['close'] > new_columns_df[f'EMA_{length}']).astype(int)

            for length in config.rsi_lengths:
                dependent_columns[f'RSI_{length}_slope'] = (new_columns_df[f'RSI_{length}'] > new_columns_df[f'RSI_{length}'].shift(1)).astype(int)

            for length in config.mfi_lengths:
                dependent_columns[f'MFI_{length}_slope'] = (new_columns_df[f'MFI_{length}'] > new_columns_df[f'MFI_{length}'].shift(1)).astype(int)

            for length in config.stoch_lengths:
                for d in config.stoch_ds:
                    dependent_columns[f'STOCHk_{length}_{d}_slope'] = (new_columns_df[f'STOCHk_{length}_{d}'] > new_columns_df[f'STOCHk_{length}_{d}'].shift(1)).astype(int)
                    dependent_columns[f'STOCHd_{length}_{d}_slope'] = (new_columns_df[f'STOCHd_{length}_{d}'] > new_columns_df[f'STOCHd_{length}_{d}'].shift(1)).astype(int)

        if config.is_order:
            for i in range(len(config.sma_lengths) - 1):
                dependent_columns[f'SMA_ordered_{config.sma_lengths[i]}_{config.sma_lengths[i+1]}'] = (new_columns_df[f'SMA_{config.sma_lengths[i]}'] > new_columns_df[f'SMA_{config.sma_lengths[i+1]}']).astype(int)
            
            for i in range(len(config.ema_lengths) - 1):
                dependent_columns[f'EMA_ordered_{config.ema_lengths[i]}_{config.ema_lengths[i+1]}'] = (new_columns_df[f'EMA_{config.ema_lengths[i]}'] > new_columns_df[f'EMA_{config.ema_lengths[i+1]}']).astype(int)

        df_with_features = pd.concat([df, new_columns_df, pd.DataFrame(dependent_columns)], axis=1)
        df_with_features.dropna(inplace=True)

        # drop unnecessary columns
        drop_cols = [f'SMA_{length}' for length in config.sma_lengths] + [f'EMA_{length}' for length in config.ema_lengths]
        df_with_features.drop(columns=drop_cols, inplace=True)
        
        result.append(df_with_features)

    return result


def create_scaler(dfs, columns_to_normalize, scaler_path='scaler.pkl') -> MinMaxScaler:
    combined_df = pd.concat(dfs)
    scaler = MinMaxScaler(clip=True)
    combined_df[columns_to_normalize] = scaler.fit_transform(combined_df[columns_to_normalize])
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    return scaler


def normalize_data(dfs: List[pd.DataFrame], scaler: MinMaxScaler, columns_to_normalize: List[str]) -> List[pd.DataFrame]:
    for df in dfs:
        df[columns_to_normalize] = scaler.transform(df[columns_to_normalize])
    return dfs


def load_scaler(scaler_file='scaler.pkl'):
    if os.path.exists(scaler_file):
        with open(scaler_file, 'rb') as f:
            scaler = pickle.load(f)
    else:
        raise Exception(f'{scaler_file} does not exist')
    return scaler


class TrainingDataConfig:
    def __init__(
            self,
            sequence_length=5,
            future_candles_count=3,
            atr_multiple=1.0,
            atr_length=7,
            pct_increase=2.5,
            prediction_type=PredictionType.pct_increase,
        ):
        self.sequence_length = sequence_length
        self.future_candles_count = future_candles_count
        self.atr_multiple = atr_multiple
        self.atr_length = atr_length
        self.pct_increase = pct_increase
        self.prediction_type = prediction_type

    def __str__(self):
        return (
            f"SequencesLabelsConfig(\n"
            f"  sequence_length={self.sequence_length},\n"
            f"  future_candles_count={self.future_candles_count},\n"
            f"  atr_multiple={self.atr_multiple},\n"
            f"  atr_length={self.atr_length},\n"
            f"  pct_increase={self.pct_increase},\n"
            f"  prediction_type={self.prediction_type},\n"
            f")"
        )


def create_training_data(
        dfs: List[pd.DataFrame], 
        feature_columns: List[str], 
        config: TrainingDataConfig, 
    ) -> tuple[List[ndarray], List[int], List]:

    if config.prediction_type == PredictionType.atr_mult:
        return create_training_data_atr_percentage_increase(
            dfs=dfs,
            feature_columns=feature_columns,
            sequence_length=config.sequence_length,
            atr_multiple=config.atr_multiple,
            atr_length=config.atr_length,
            future_candles=config.future_candles_count,
        )
    elif config.prediction_type == PredictionType.next_close_direction:
        return create_training_data_next_close_direction(
            dfs=dfs,
            feature_columns=feature_columns,
            sequence_length=config.sequence_length,
        )
    elif config.prediction_type == PredictionType.pct_increase:
        return create_training_data_percentage_increase(
            dfs=dfs,
            feature_columns=feature_columns,
            future_candles=config.future_candles_count,
            percentage_increase=config.pct_increase,
            sequence_length=config.sequence_length,
        )
    
    raise Exception(f'Invalid prediction type {config.prediction_type}')


def create_training_data_atr_percentage_increase(
        dfs: List[pd.DataFrame],
        feature_columns: List[str],
        sequence_length=5, 
        future_candles=3,
        atr_multiple=1.0,
        atr_length=14
    ) -> tuple[List[ndarray], List[int], List]:

    sequences = []
    labels = []
    combined_data = []

    for df in dfs:
        df[f'ATR_label_{atr_length}'] = ta.atr(df['high'], df['low'], df['close'], length=atr_length)
        for i in range(len(df) - sequence_length - future_candles):
            sequence = df[feature_columns].iloc[i:i+sequence_length].values
            future_closes = df['close'].iloc[i+sequence_length:i+sequence_length+future_candles]
            current_close = df['close'].iloc[i+sequence_length-1]
            current_atr = df[f'ATR_label_{atr_length}'].iloc[i+sequence_length-1]
            increased_close = current_close + atr_multiple * current_atr
            label = int((future_closes > increased_close).any())
            sequences.append(sequence)
            labels.append(label)
            sequence_start_date = df['timestamp'].iloc[i]
            sequence_end_date = df['timestamp'].iloc[i + sequence_length - 1]
            prediction_end_date = df['timestamp'].iloc[i + sequence_length + future_candles - 1]
            combined_row = [sequence_start_date, sequence_end_date, prediction_end_date] + sequence.flatten().tolist() + [label]
            combined_data.append(combined_row)
        df.drop(columns=[f'ATR_label_{atr_length}'], inplace=True)
       
    return sequences, labels, combined_data


def create_training_data_percentage_increase(
        dfs: List[pd.DataFrame], 
        sequence_length: int, 
        feature_columns: List[str], 
        future_candles=3, 
        percentage_increase=1.0
    ) -> tuple[List[ndarray], List[int], List]:

    sequences = []
    labels = []
    combined_data = []

    for df in dfs:
        for i in range(len(df) - sequence_length - future_candles):
            sequence = df[feature_columns].iloc[i:i+sequence_length].values
            future_closes = df['close'].iloc[i+sequence_length:i+sequence_length+future_candles]
            current_close = df['close'].iloc[i+sequence_length-1]
            increased_close = current_close * (1 + percentage_increase / 100)
            label = int((future_closes > increased_close).any())
            sequences.append(sequence)
            labels.append(label)
            sequence_start_date = df['timestamp'].iloc[i]
            sequence_end_date = df['timestamp'].iloc[i + sequence_length - 1]
            prediction_end_date = df['timestamp'].iloc[i + sequence_length + future_candles - 1]
            combined_row = [sequence_start_date, sequence_end_date, prediction_end_date] + sequence.flatten().tolist() + [label]
            combined_data.append(combined_row)
     
    return sequences, labels, combined_data


def create_training_data_next_close_direction(
        dfs: List[pd.DataFrame], 
        sequence_length: int, 
        feature_columns: List[str]
    ) -> tuple[List[ndarray], List[int], List]:

    sequences = []
    labels = []
    combined_data = []

    for df in dfs:
        for i in range(len(df) - sequence_length):
            sequence = df[feature_columns].iloc[i:i+sequence_length].values
            label = int(df['close'].iloc[i+sequence_length] > df['close'].iloc[i+sequence_length-1])
            sequences.append(sequence)
            labels.append(label)
            sequence_start_date = df['timestamp'].iloc[i]
            sequence_end_date = df['timestamp'].iloc[i + sequence_length - 1]
            prediction_end_date = df['timestamp'].iloc[i + sequence_length]
            combined_row = [sequence_start_date, sequence_end_date, prediction_end_date] + sequence.flatten().tolist() + [label]
            combined_data.append(combined_row)
  
    return sequences, labels, combined_data
 

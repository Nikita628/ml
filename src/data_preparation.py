from typing import List, Any, Tuple
import pandas as pd
import os
import pickle
import pandas_ta as ta
from utils import PredictionType
from numpy import ndarray
from collections import Counter
import random
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample

def calculate_label_percentages(labels) -> dict[str, float]:
    total_count = len(labels)
    label_counts = Counter(labels)
    label_percentages = {label: (count / total_count) * 100 for label, count in label_counts.items()}
    
    return label_percentages


def is_balanced(label_percentages: dict[str, float], threshold: float):
    for key, value in label_percentages.items():
        if value > threshold:
            return False
    return True


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

            df['high'] = df['high'].astype(float)
            df['low'] = df['low'].astype(float)
            df['close'] = df['close'].astype(float)
            df['volume'] = df['volume'].astype(float) 
            df['open'] = df['open'].astype(float) 
            df['file'] = filename
            df = prune_zeros(df)

            if len(df) >= min_len:
                dataframes.append(df)

    return dataframes


def prune_zeros(df: pd.DataFrame) -> pd.DataFrame:
    # Find the latest row where any of the columns contains 0.0
    pruned_df = df
    zero_mask = (df[['open', 'close', 'high', 'low', 'volume']] == 0.0).any(axis=1)
    if zero_mask.any():
        last_zero_index = zero_mask[::-1].idxmax()
        pruned_df = df.iloc[last_zero_index + 1:]
    return pruned_df


class FeaturesConfig:
    def __init__(
            self,
            sma_lengths=[2,5,7,10,15,20,30,50,100,200],
            ema_lengths=[2,5,7,10,15,20,30,50,100,200],
            rsi_lengths=[2,3,5,7,10,15,20,25,30],
            stoch_k_lengths=[2,3,5,7,10,15,20,25,30],
            stoch_d_lengths=[2,3,5,7,10,15,20,25,30],
            stoch_k_smooth=[2,3,5,7,10,15,20,25,30],
            mfi_lengths=[2,3,5,7,10,15,20,25,30],
            adx_lengths=[2,3,5,7,10,15,20,25,30],
            atr_lengths=[2,3,5,7,10,15,20,25,30],
            std_lengths=[2,3,5,7,10,15,20,25,30],
            bb_lengths=[2,3,5,7,10,15,20,25,30],
            bb_stds=[1.5, 2.0, 3.0],
            ichimoku=[(9, 26, 52),(5, 15, 30),(12, 24, 48),(10, 30, 60),(7, 22, 44)],
        ):
        self.sma_lengths = sma_lengths
        self.ema_lengths = ema_lengths
        self.rsi_lengths = rsi_lengths
        self.stoch_k_lengths = stoch_k_lengths
        self.stoch_d_lengths = stoch_d_lengths
        self.stoch_k_smooth = stoch_k_smooth
        self.mfi_lengths = mfi_lengths
        self.adx_lengths = adx_lengths
        self.atr_lengths = atr_lengths
        self.std_lengths = std_lengths
        self.bb_lengths = bb_lengths
        self.ichimoku = ichimoku
        self.bb_stds = bb_stds

    def __str__(self):
        return (
            f"FeaturesConfig(\n"
            f"  sma_lengths={self.sma_lengths},\n"
            f"  ema_lengths={self.ema_lengths},\n"
            f"  rsi_lengths={self.rsi_lengths},\n"
            f"  stoch_k_lengths={self.stoch_k_lengths},\n"
            f"  stoch_d_lengths={self.stoch_d_lengths},\n"
            f"  stoch_k_smooth={self.stoch_k_smooth},\n"
            f"  mfi_lengths={self.mfi_lengths},\n"
            f"  adx_lengths={self.adx_lengths},\n"
            f"  atr_lengths={self.atr_lengths},\n"
            f"  std_lengths={self.std_lengths}\n"
            f"  bb_lengths={self.bb_lengths}\n"
            f"  ichimoku={self.ichimoku}\n"
            f"  bb_stds={self.bb_stds}\n"
            f")"
        )


def add_ichimoku_components(df, tenkan, kijun, senkou, new_columns):
    ichimoku, _ = ta.ichimoku(high=df['high'], low=df['low'], close=df['close'], tenkan=tenkan, kijun=kijun, senkou=senkou)
    new_columns[f'ratio_ISA_{tenkan}'] = ichimoku[f'ISA_{tenkan}'] / df['close']
    new_columns[f'ratio_ISB_{kijun}'] = ichimoku[f'ISB_{kijun}'] / df['close']
    new_columns[f'ratio_ITS_{tenkan}'] = ichimoku[f'ITS_{tenkan}'] / df['close']
    new_columns[f'ratio_IKS_{kijun}'] = ichimoku[f'IKS_{kijun}'] / df['close']
    new_columns[f'ratio_ICS_{kijun}'] = ichimoku[f'ICS_{kijun}'] / df['close']


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

        for k in config.stoch_k_lengths:
            for d in config.stoch_d_lengths:
                for smooth in config.stoch_k_smooth:
                    stoch = ta.stoch(df['high'], df['low'], df['close'], k=k, d=d, smooth_k=smooth)
                    new_columns[f'STOCHk_{k}_{d}_{smooth}'] = stoch[f'STOCHk_{k}_{d}_{smooth}']
                    new_columns[f'STOCHd_{k}_{d}_{smooth}'] = stoch[f'STOCHd_{k}_{d}_{smooth}']
        
        for length in config.adx_lengths:
            adx = ta.adx(df['high'], df['low'], df['close'], length=length)
            new_columns[f'ADX_{length}'] = adx[f'ADX_{length}']
            new_columns[f'DMP_{length}'] = adx[f'DMP_{length}']
            new_columns[f'DMN_{length}'] = adx[f'DMN_{length}']

        for length in config.bb_lengths:
            for std in config.bb_stds:
                bbands = ta.bbands(close=df['close'], length=length, std=std)
                new_columns[f'ratio_BBL_{length}_{std}'] = bbands[f'BBL_{length}_{std}'] / df['close']
                new_columns[f'ratio_BBM_{length}_{std}'] = bbands[f'BBM_{length}_{std}'] / df['close']
                new_columns[f'ratio_BBU_{length}_{std}'] = bbands[f'BBU_{length}_{std}'] / df['close']
                new_columns[f'ratio_BBB_{length}_{std}'] = bbands[f'BBB_{length}_{std}'] / df['close']
                new_columns[f'ratio_BBP_{length}_{std}'] = bbands[f'BBP_{length}_{std}'] / df['close']

        for a,b,c in config.ichimoku:
            add_ichimoku_components(df, a, b, c, new_columns)

        # custom features
        for length in config.std_lengths:
            new_columns[f'std_pct_{length}'] = df['close'].rolling(window=length).std() / df['close'].rolling(window=length).mean()

        new_columns['close_pct_change'] = df['close'].pct_change()
        new_columns['high_pct_change'] = df['high'].pct_change()
        new_columns['low_pct_change'] = df['low'].pct_change()
        new_columns['open_pct_change'] = df['open'].pct_change()
        new_columns['volume_pct_change'] = df['volume'].pct_change()
        new_columns['ratio_open_close'] = df['open'] / df['close']
        new_columns['ratio_high_close'] = df['high'] / df['close']
        new_columns['ratio_low_close'] = df['low'] / df['close']
        new_columns['ratio_high_low'] = df['high'] / df['low']

        for length in config.sma_lengths:
            new_columns[f'ratio_SMA_{length}_close'] = new_columns[f'SMA_{length}'] / df['close']

        for length in config.ema_lengths:
            new_columns[f'ratio_EMA_{length}_close'] = new_columns[f'EMA_{length}'] / df['close']

        df_with_features = pd.concat([df, pd.DataFrame(new_columns)], axis=1)
        df_with_features.dropna(inplace=True)

        # drop unnecessary columns
        drop_cols = [f'SMA_{length}' for length in config.sma_lengths] + [f'EMA_{length}' for length in config.ema_lengths]
        df_with_features.drop(columns=drop_cols, inplace=True)
        
        result.append(df_with_features)

    return result


def create_scaler(dfs, columns_to_normalize, scaler_path='scaler.pkl') -> StandardScaler:
    combined_df = pd.concat(dfs)
    scaler = StandardScaler()
    scaler.fit(combined_df[columns_to_normalize])
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    return scaler

def create_regression_scaler(data: ndarray, scaler_path='scaler.pkl') -> StandardScaler:
    num_samples, sequence_length, num_features = data.shape
    data_2d = data.reshape(-1, num_features)
    scaler = StandardScaler()
    scaler.fit(data_2d)
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    return scaler

def normalize_regression_data(data: ndarray, scaler: StandardScaler) -> ndarray:
    num_samples, sequence_length, num_features = data.shape
    data_2d = data.reshape(-1, num_features)
    normalized = scaler.transform(data_2d)
    return normalized.reshape(num_samples, sequence_length, num_features)

def normalize_data(dfs: List[pd.DataFrame], scaler: StandardScaler, columns_to_normalize: List[str]) -> List[pd.DataFrame]:
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
        collect_combined_data: bool = False,
    ) -> Tuple[ndarray, ndarray, List]:

    if config.prediction_type == PredictionType.atr_mult:
        return create_training_data_atr_percentage_increase(
            dfs=dfs,
            feature_columns=feature_columns,
            sequence_length=config.sequence_length,
            atr_multiple=config.atr_multiple,
            atr_length=config.atr_length,
            future_candles=config.future_candles_count,
            collect_combined_data=collect_combined_data,
        )
    elif config.prediction_type == PredictionType.next_close_direction:
        return create_training_data_next_close_direction(
            dfs=dfs,
            feature_columns=feature_columns,
            sequence_length=config.sequence_length,
            collect_combined_data=collect_combined_data,
        )
    elif config.prediction_type == PredictionType.pct_increase:
        return create_training_data_percentage_increase(
            dfs=dfs,
            feature_columns=feature_columns,
            future_candles=config.future_candles_count,
            percentage_increase=config.pct_increase,
            sequence_length=config.sequence_length,
            collect_combined_data=collect_combined_data,
        )
    elif config.prediction_type == PredictionType.pct_change:
        return create_training_data_percentage_change(
            dfs=dfs,
            feature_columns=feature_columns,
            future_candles=config.future_candles_count,
            sequence_length=config.sequence_length,
            collect_combined_data=collect_combined_data,
        )
    
    raise Exception(f'Invalid prediction type {config.prediction_type}')


def create_training_data_atr_percentage_increase(
        dfs: List[pd.DataFrame],
        feature_columns: List[str],
        sequence_length=5, 
        future_candles=3,
        atr_multiple=1.0,
        atr_length=14,
        collect_combined_data: bool = False,
    ) -> Tuple[ndarray, ndarray, List]:

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
            if collect_combined_data:
                combined_row = [sequence_start_date, sequence_end_date, prediction_end_date] + sequence.flatten().tolist() + [label]
                combined_data.append(combined_row)
        df.drop(columns=[f'ATR_label_{atr_length}'], inplace=True)
       
    return np.array(sequences), np.array(labels), combined_data


def create_training_data_percentage_increase(
        dfs: List[pd.DataFrame], 
        sequence_length: int, 
        feature_columns: List[str], 
        future_candles=3, 
        percentage_increase=1.0,
        collect_combined_data: bool = False,
    ) -> Tuple[ndarray, ndarray, List]:

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
            if collect_combined_data:
                combined_row = [sequence_start_date, sequence_end_date, prediction_end_date] + sequence.flatten().tolist() + [label]
                combined_data.append(combined_row)
     
    return np.array(sequences), np.array(labels), combined_data

def create_training_data_percentage_change(
        dfs: List[pd.DataFrame], 
        sequence_length: int, 
        feature_columns: List[str], 
        future_candles=3, 
        collect_combined_data: bool = False,
    ) -> Tuple[ndarray, ndarray, List]:

    sequences = []
    labels = []
    combined_data = []

    for df in dfs:
        for i in range(len(df) - sequence_length - future_candles):
            sequence = df[feature_columns].iloc[i:i+sequence_length].values
            future_close = df['close'].iloc[i+sequence_length+future_candles-1]
            current_close = df['close'].iloc[i+sequence_length-1]
            percentage_change = ((future_close - current_close) / current_close) * 100
            sequences.append(sequence)
            labels.append(percentage_change)
            sequence_start_date = df['timestamp'].iloc[i]
            sequence_end_date = df['timestamp'].iloc[i + sequence_length - 1]
            prediction_end_date = df['timestamp'].iloc[i + sequence_length + future_candles - 1]
            if collect_combined_data:
                combined_row = [sequence_start_date, sequence_end_date, prediction_end_date] + sequence.flatten().tolist() + [percentage_change]
                combined_data.append(combined_row)
     
    return np.array(sequences), np.array(labels), combined_data


def create_training_data_next_close_direction(
        dfs: List[pd.DataFrame], 
        sequence_length: int, 
        feature_columns: List[str],
        collect_combined_data: bool = False,
    ) -> Tuple[ndarray, ndarray, List]:

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
            if collect_combined_data:
                combined_row = [sequence_start_date, sequence_end_date, prediction_end_date] + sequence.flatten().tolist() + [label]
                combined_data.append(combined_row)
  
    return np.array(sequences), np.array(labels), combined_data
 

def balance_dataset(X: ndarray, y: ndarray, method: str = 'over', max_pct_diff: float = 10) -> Tuple[ndarray, ndarray]:
    # Ensure inputs are numpy arrays
    X = np.array(X)
    y = np.array(y)

    # Get the counts of each class
    unique, counts = np.unique(y, return_counts=True)
    count_dict = dict(zip(unique, counts))
    
    if len(count_dict) != 2:
        raise ValueError("The function currently supports binary classification (0/1 labels).")
    
    # Determine the minority and majority classes
    minority_class = min(count_dict, key=count_dict.get)
    majority_class = max(count_dict, key=count_dict.get)
    
    minority_count = count_dict[minority_class]
    majority_count = count_dict[majority_class]
    
    # Calculate the percentage difference
    current_pct_diff = abs(majority_count - minority_count) / float(len(y)) * 100
    
    if current_pct_diff <= max_pct_diff:
        print("Dataset is already balanced within the specified max_pct_diff.")
        return X, y
    
    # Balance the dataset based on the method
    if method == 'over':
        # Over-sample the minority class
        n_samples_to_add = int((majority_count - minority_count) - (max_pct_diff / 100.0 * len(y) / 2))
        X_minority = X[y == minority_class]
        y_minority = y[y == minority_class]
        
        X_minority_upsampled = resample(X_minority, replace=True, n_samples=n_samples_to_add, random_state=42)
        y_minority_upsampled = np.full(n_samples_to_add, minority_class)
        
        X_balanced = np.concatenate((X, X_minority_upsampled))
        y_balanced = np.concatenate((y, y_minority_upsampled))
    
    elif method == 'under':
        # Under-sample the majority class
        n_samples_to_remove = int((majority_count - minority_count) - (max_pct_diff / 100.0 * len(y) / 2))
        X_majority = X[y == majority_class]
        y_majority = y[y == majority_class]
        
        X_majority_downsampled = resample(X_majority, replace=False, n_samples=majority_count - n_samples_to_remove, random_state=42)
        y_majority_downsampled = np.full(majority_count - n_samples_to_remove, majority_class)
        
        X_balanced = np.concatenate((X[y != majority_class], X_majority_downsampled))
        y_balanced = np.concatenate((y[y != majority_class], y_majority_downsampled))
    
    else:
        raise ValueError("Method must be either 'over' or 'under'.")
    
    return X_balanced, y_balanced
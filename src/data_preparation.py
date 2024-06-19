from typing import List, Any
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

            df['file'] = filename
            dataframes.append(df)
    return dataframes


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
            new_columns[f'variation_{length}'] = df['close'].rolling(window=length).std() / df['close'].rolling(window=length).mean()

        new_columns['close_pct_change'] = df['close'].pct_change()
        new_columns['high_pct_change'] = df['high'].pct_change()
        new_columns['low_pct_change'] = df['low'].pct_change()
        new_columns['open_pct_change'] = df['open'].pct_change()
        new_columns['volume_pct_change'] = df['volume'].pct_change()
        new_columns['ratio_open_close'] = df['open'] / df['close']
        new_columns['ratio_high_close'] = df['high'] / df['close']
        new_columns['ratio_low_close'] = df['low'] / df['close']

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
    combined_df[columns_to_normalize] = scaler.fit_transform(combined_df[columns_to_normalize])
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    return scaler


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
    ) -> tuple[List[ndarray], List[int], List]:

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
    
    raise Exception(f'Invalid prediction type {config.prediction_type}')


def create_training_data_atr_percentage_increase(
        dfs: List[pd.DataFrame],
        feature_columns: List[str],
        sequence_length=5, 
        future_candles=3,
        atr_multiple=1.0,
        atr_length=14,
        collect_combined_data: bool = False,
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
            if collect_combined_data:
                combined_row = [sequence_start_date, sequence_end_date, prediction_end_date] + sequence.flatten().tolist() + [label]
                combined_data.append(combined_row)
        df.drop(columns=[f'ATR_label_{atr_length}'], inplace=True)
       
    return sequences, labels, combined_data


def create_training_data_percentage_increase(
        dfs: List[pd.DataFrame], 
        sequence_length: int, 
        feature_columns: List[str], 
        future_candles=3, 
        percentage_increase=1.0,
        collect_combined_data: bool = False,
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
            if collect_combined_data:
                combined_row = [sequence_start_date, sequence_end_date, prediction_end_date] + sequence.flatten().tolist() + [label]
                combined_data.append(combined_row)
     
    return sequences, labels, combined_data


def create_training_data_next_close_direction(
        dfs: List[pd.DataFrame], 
        sequence_length: int, 
        feature_columns: List[str],
        collect_combined_data: bool = False,
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
            if collect_combined_data:
                combined_row = [sequence_start_date, sequence_end_date, prediction_end_date] + sequence.flatten().tolist() + [label]
                combined_data.append(combined_row)
  
    return sequences, labels, combined_data
 

def balance_dataset(
        sequences: List[np.ndarray], 
        labels: List[int], 
        method: str = 'downsample', 
        max_percentage_diff: float = 0.1
    ) -> tuple[List[np.ndarray], List[int]]:
    # Separate the sequences and labels into different classes
    class_0_indices = [i for i, label in enumerate(labels) if label == 0]
    class_1_indices = [i for i, label in enumerate(labels) if label == 1]
    
    # Determine minority and majority classes
    if len(class_0_indices) > len(class_1_indices):
        majority_class_indices = class_0_indices
        minority_class_indices = class_1_indices
    else:
        majority_class_indices = class_1_indices
        minority_class_indices = class_0_indices
    
    # Calculate the target size for balancing
    if method == 'downsample':
        # Maximum allowable size for the majority class
        max_majority_size = int(len(minority_class_indices) * (1 + max_percentage_diff))
        if len(majority_class_indices) > max_majority_size:
            majority_class_indices = random.sample(majority_class_indices, max_majority_size)
    elif method == 'upsample':
        # Target size for the minority class
        target_minority_size = int(len(majority_class_indices) / (1 + max_percentage_diff))
        minority_class_indices = minority_class_indices * (target_minority_size // len(minority_class_indices)) + random.choices(minority_class_indices, k=target_minority_size % len(minority_class_indices))
    else:
        raise ValueError("Method should be either 'downsample' or 'upsample'")
    
    # Combine the indices and shuffle them to mix the classes
    balanced_indices = majority_class_indices + minority_class_indices
    random.shuffle(balanced_indices)
    
    # Create balanced sequences and labels
    balanced_sequences = [sequences[i] for i in balanced_indices]
    balanced_labels = [labels[i] for i in balanced_indices]
    
    return balanced_sequences, balanced_labels
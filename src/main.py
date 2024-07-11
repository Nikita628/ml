from datetime import datetime
from typing import List
import os
from sklearn.model_selection import train_test_split
from utils import NON_FEATURE_COLUMNS, format_date
import numpy as np
from sklearn.metrics import accuracy_score, classification_report
from modeling import (
    create_model,
    train_model,
    save_model,
)
from evaluation import (
    test_model,
    test_model_on_unseen_data,
    write_test_report,
    test_ensemble_on_unseen_data,
)
from data_preparation import (
    FeaturesConfig, 
    TrainingDataConfig,
    calculate_label_percentages,
    load_data,
    add_features,
    create_training_data,
    create_scaler,
    normalize_data,
)


def prepare_model(
        data_path: str,
        unseen_path: str,
        features_config: FeaturesConfig, 
        training_data_config: TrainingDataConfig,
        prediction_thresholds: List[float]
    ):
    
    model_dir = format_date(datetime.now())
    model_dir_path = f'src/tested_models/{model_dir}'
    os.makedirs(model_dir_path, exist_ok=True)

    # data preprocessing
    dataframes = load_data(data_path)
    dataframes = add_features(dataframes, features_config)

    feature_columns = [col for col in dataframes[0].columns if col not in NON_FEATURE_COLUMNS]
    
    scaler_path = f'{model_dir_path}/scaler.pkl'
    scaler = create_scaler(
        dfs=dataframes, 
        columns_to_normalize=feature_columns, 
        scaler_path=scaler_path
    )

    dataframes = normalize_data(
        dfs=dataframes, 
        scaler=scaler, 
        columns_to_normalize=feature_columns
    )

    sequences, labels, _ = create_training_data(
        dfs=dataframes, 
        feature_columns=feature_columns, 
        config=training_data_config
    )

    label_percentages = calculate_label_percentages(labels=labels)
    print(f'label percentages: \n{label_percentages}')

    # modeling
    input_shape = (training_data_config.sequence_length, len(feature_columns))
    model = create_model(input_shape=input_shape)

    X_train, X_test, y_train, y_test = train_test_split(
        sequences, 
        labels, 
        test_size=0.2, 
        random_state=42, 
        stratify=labels
    )
    model = train_model(model=model, x_train=X_train, y_train=y_train)
    model_path = f'{model_dir_path}/model.h5'
    save_model(model=model, model_path=model_path)

    # evaluation
    model_predictions = test_model(model=model, x_test=X_test, y_test=y_test, prediction_thresholds=prediction_thresholds)

    write_test_report(
        title=f'data_path={data_path}\nunseen_path={unseen_path}',
        label_percentages=label_percentages,
        features_config=features_config,
        training_data_config=training_data_config,
        report_path=f'{model_dir_path}/test_report.txt',
        model_predictions=model_predictions,
    )

    test_model_on_unseen_data(
        model_path=model_path, 
        scaler_path=scaler_path, 
        unseen_path=unseen_path,
        report_path=f'{model_dir_path}',
        features_config=features_config,
        training_data_config=training_data_config,
        prediction_thresholds=prediction_thresholds,
    )


def prepare_model_ensemble(
        data_path: str,
        unseen_path: str,
        features_config: FeaturesConfig, 
        training_data_config: TrainingDataConfig,
        n_estimators: int = 10,
        test_size: float = 0.2,
        bootstrap_fraction: float = 0.8,
        random_state: int = 42,
    ):
    model_dir = format_date(datetime.now())
    model_dir_path = f'src/tested_models/{model_dir}'
    os.makedirs(model_dir_path, exist_ok=True)

    dataframes = load_data(data_path)
    dataframes = add_features(dataframes, features_config)

    feature_columns = [col for col in dataframes[0].columns if col not in NON_FEATURE_COLUMNS]

    scaler_path = f'{model_dir_path}/scaler.pkl'
    scaler = create_scaler(dfs=dataframes, columns_to_normalize=feature_columns, scaler_path=scaler_path)
    dataframes = normalize_data(dfs=dataframes, scaler=scaler, columns_to_normalize=feature_columns)

    sequences, labels, _ = create_training_data(dfs=dataframes, feature_columns=feature_columns, config=training_data_config)
    label_percentages = calculate_label_percentages(labels=labels)
    print(f'label percentages: \n{label_percentages}')

    X_train, X_test, y_train, y_test = train_test_split(
        sequences, 
        labels, 
        test_size=test_size, 
        random_state=random_state,
        stratify=labels
    )
    X_train = np.array(X_train)
    y_train = np.array(y_train)

    models = []
    bootstrap_size = int(len(X_train) * bootstrap_fraction)
    for i in range(n_estimators):
        bootstrap_indices = np.random.choice(np.arange(len(X_train)), size=bootstrap_size, replace=True)
        X_bootstrap = X_train[bootstrap_indices]
        y_bootstrap = y_train[bootstrap_indices]

        input_shape = (training_data_config.sequence_length, len(feature_columns))
        model = create_model(input_shape)
        model = train_model(model=model, x_train=X_bootstrap, y_train=y_bootstrap)
        
        model_path = os.path.join(model_dir_path, f'model_{i}.h5')
        save_model(model, model_path)
        models.append(model)

    model_reports = []
    for i, model in enumerate(models):
        accuracy, report, _ = test_model(model, X_test, y_test)
        model_reports.append((accuracy, report))

    predictions = [model.predict(np.array(X_test)) for model in models]
    ensemble_predictions = np.sum(predictions, axis=0) > (n_estimators / 2)
    ensemble_predictions = ensemble_predictions.astype(int)
    ensemble_accuracy = accuracy_score(np.array(y_test), ensemble_predictions)
    ensemble_report = classification_report(np.array(y_test), ensemble_predictions)
    
    with open(f'{model_dir_path}/by_model_ensemble_report.txt', 'a') as file:
        for accuracy, report in model_reports:
            file.write(f'{accuracy}\n{report}')
            file.write('\n\n')

    write_test_report(
        title=f'data_path={data_path}\nunseen_path={unseen_path}',
        accuracy=ensemble_accuracy,
        label_percentages=label_percentages,
        classification_report=ensemble_report,
        features_config=features_config,
        training_data_config=training_data_config,
        report_path=f'{model_dir_path}/test_ensemble_report.txt'
    )

    test_ensemble_on_unseen_data(
        models=models, 
        scaler_path=scaler_path, 
        unseen_path=unseen_path,
        report_path=f'{model_dir_path}',
        features_config=features_config,
        training_data_config=training_data_config,
        n_estimators=n_estimators,
    )


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
    sequence_length=2,
    future_candles_count=2,
    pct_increase=3,
)

if __name__ == '__main__':
    prepare_model(
        data_path='src/datasets_1d/small',
        unseen_path='src/datasets_1d/unseen',
        features_config=features_config,
        training_data_config=training_data_config,
        prediction_thresholds=[0.65, 0.7, 0.75, 0.8],
    )


# most optimal sequence length on 1d is between 15 - 20, given tech indicators lengths up to 10
# sensitivity (recall) drops down after making the model more complex (adding more layers)

# Imbalanced Dataset: 
# If there is a class imbalance, consider techniques like oversampling (duplicating instances of the minority class) 
# or undersampling (reducing instances of the majority class) to balance the dataset.
# NEED to measure 0/1 class balance. Supposedly need to make 50/50
        
# Class-Specific Weights: Modify the loss function to give higher weights to the misclassification of oranges, 
# which will force the model to focus more on getting oranges correct.

# Focal Loss: 
# Use focal loss to handle class imbalance by focusing more on hard-to-classify examples.


# Ideas
#
# multiclass classifier - increase 5% -> class 1, increase 10% -> class 2, etc.
# regression ? predict percentage change ?
# train only on recent data (1000 last candles)
# explore shorter timeframes

# predict
    # load model
    # load scaler
    # load data
    # add features
    # normalize
    # scale
    # predict
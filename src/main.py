from datetime import datetime
import os
import sys
from sklearn.model_selection import train_test_split
from utils import NON_FEATURE_COLUMNS, format_date
from modeling import (
    create_model,
    train_model,
)
from evaluation import (
    test_model,
    test_model_on_unseen_data,
    write_test_report,
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
        report_title: str,
        features_config: FeaturesConfig, 
        training_data_config: TrainingDataConfig
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

    sequences, labels, combined_data = create_training_data(
        dfs=dataframes, 
        feature_columns=feature_columns, 
        config=training_data_config
    )

    label_percentages = calculate_label_percentages(labels=labels)

    # modeling
    input_shape = (training_data_config.sequence_length, len(feature_columns))
    model = create_model(input_shape=input_shape)

    X_train, X_test, y_train, y_test = train_test_split(sequences, labels, test_size=0.2, random_state=42)
    model = train_model(model=model, x_train=X_train, y_train=y_train)
    model_path = f'{model_dir_path}/model.h5'
    model.save(model_path)

    # evaluation
    accuracy, report, y_pred = test_model(model=model, x_test=X_test, y_test=y_test)

    write_test_report(
        title=report_title,
        accuracy=accuracy,
        label_percentages=label_percentages,
        classification_report=report,
        features_config=features_config,
        training_data_config=training_data_config,
        report_path=f'{model_dir_path}/test_report.txt'
    )

    test_model_on_unseen_data(
        model_path=model_path, 
        scaler_path=scaler_path, 
        unseen_path=unseen_path,
        report_path=f'{model_dir_path}',
        features_config=features_config,
        training_data_config=training_data_config
    )



features_config = FeaturesConfig()
training_data_config = TrainingDataConfig(
    sequence_length=2,
    future_candles_count=2,
    pct_increase=2,
)

if __name__ == '__main__':
    if len(sys.argv) > 1:
        model_run = sys.argv[1]
        print(f'starting model run: {model_run}')
        prepare_model(
            data_path='src/datasets_4h/small',
            unseen_path='src/datasets_4h/unseen',
            features_config=features_config,
            training_data_config=training_data_config,
            report_title=model_run
        )
    else:
        prepare_model(
            data_path='src/datasets_4h/small',
            unseen_path='src/datasets_4h/unseen',
            features_config=features_config,
            training_data_config=training_data_config,
            report_title='4h small'
        )


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
# increase the thrashold of the binary classifier (> 0.5 -> > 0.7)
# multiclass classifier - increase 5% -> class 1, increase 10% -> class 2, etc.
# regression ? predict percentage change ?
# train only on recent data (1000 last candles)
# more momentum tech indicators
# explore shorter timeframes
# various sequence lengths

# predict
    # load model
    # load scaler
    # load data
    # add features
    # normalize
    # scale
    # predict
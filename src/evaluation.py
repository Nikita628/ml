import pandas as pd
import numpy as np
from typing import Any
from sklearn.metrics import accuracy_score, classification_report
import inspect
from modeling import create_model, load_model
from utils import NON_FEATURE_COLUMNS
from data_preparation import (
    load_data, 
    add_features,
    load_scaler, 
    normalize_data,
    create_training_data,
    TrainingDataConfig,
    FeaturesConfig,
)


def test_model(model, x_test, y_test):
    X = np.array(x_test)
    y_true = np.array(y_test)
    y_pred = model.predict(X)
    y_pred = (y_pred > 0.5).astype(int)
    accuracy = accuracy_score(y_true, y_pred)
    report = classification_report(y_true, y_pred)
    return accuracy, report, y_pred


def test_model_on_unseen_data(
        model_path: str, 
        scaler_path: str, 
        unseen_path: str, 
        report_path: str, 
        features_config: FeaturesConfig,
        training_data_config: TrainingDataConfig
    ):

    model = load_model(model_path)
    scaler = load_scaler(scaler_path)
    dataframes = load_data(unseen_path)
    dataframes = add_features(dataframes, features_config)

    feature_columns = [col for col in dataframes[0].columns if col not in NON_FEATURE_COLUMNS]
    reports = []

    for df in dataframes:
        df = normalize_data(
            dfs=[df], 
            scaler=scaler, 
            columns_to_normalize=feature_columns
        )[0]

        sequences, labels, combined_data = create_training_data(
            dfs=[df], 
            feature_columns=feature_columns, 
            config=training_data_config
        )

        accuracy, report, y_pred = test_model(model=model, x_test=sequences, y_test=labels)
        tested_file = df.iloc[0]['file']
        reports.append(f'{tested_file}\naccuracy={accuracy}\n{report}')

        report_file = f'{report_path}/{tested_file}_unseen_test_result.csv'
        flattened_feature_columns = [f'feature_{j}' for j in range(training_data_config.sequence_length * len(feature_columns))]
        columns = ['sequence_start_date', 'sequence_end_date', 'prediction_end_date'] \
            + flattened_feature_columns \
            + ['actual_label', 'predicted_label']
        
        for i, row in enumerate(combined_data):
            row.append(y_pred[i])

        combined_df = pd.DataFrame(combined_data, columns=columns)
        combined_df.drop(columns=flattened_feature_columns, inplace=True)
        combined_df.to_csv(report_file, index=False)

    with open(f'{report_path}/unseen_test_report.txt', 'a') as file:
        for report in reports:
            file.write(report)
            file.write('\n\n')


def write_test_report(
        title: str,
        accuracy: float,
        label_percentages: dict[Any, float],
        classification_report: str | dict, 
        training_data_config: TrainingDataConfig,
        features_config: FeaturesConfig,
        report_path: str,
    ):
    create_model_func_source = inspect.getsource(create_model)

    file_content = f"""
{title}

accuracy = {accuracy}

training data config:
{training_data_config}

label percentages:
{label_percentages}

classification report:
{classification_report}

features config:
{features_config}

model config:
{create_model_func_source}
"""

    with open(report_path, 'w') as file:
        file.write(file_content)
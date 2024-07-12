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
from typing import List, Any
from numpy import ndarray
import copy
from keras import models

class ModelPrediction:
    def __init__(self, accuracy: float, report: str | dict, y_pred: np.ndarray, prediction_threshold: float):
        self.accuracy = accuracy
        self.report = report
        self.y_pred = y_pred
        self.prediction_threshold = prediction_threshold

    def __str__(self) -> str:
        return (f"Prediction Threshold: {self.prediction_threshold}\n"
                f"Accuracy: {self.accuracy}\n"
                f"Classification Report:\n{self.report}\n")

def test_model(model: models.Sequential, x_test: ndarray, y_test: ndarray, prediction_thresholds: List[float]) -> List[ModelPrediction]:
    X = np.array(x_test)
    y_true = np.array(y_test)
    results = []
    
    for threshold in prediction_thresholds:
        y_pred = model.predict(X)
        y_pred = (y_pred > threshold).astype(int)
        accuracy = accuracy_score(y_true, y_pred)
        report = classification_report(y_true, y_pred)
        result = ModelPrediction(accuracy, report, y_pred, threshold)
        results.append(result)
    
    return results


def test_model_on_unseen_data(
        model_path: str, 
        scaler_path: str, 
        unseen_path: str, 
        report_path: str, 
        features_config: FeaturesConfig,
        training_data_config: TrainingDataConfig,
        prediction_thresholds: List[float],
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
            config=training_data_config,
            collect_combined_data=True,
        )

        model_predictions: List[ModelPrediction] = test_model(model=model, x_test=sequences, y_test=labels, prediction_thresholds=prediction_thresholds)

        for mp in model_predictions:
            combined_data_copy = copy.deepcopy(combined_data)
            tested_file = df.iloc[0]['file']
            reports.append(f'{tested_file}\nprediction_treshold={mp.prediction_threshold}\naccuracy={mp.accuracy}\n{mp.report}')

            report_file = f'{report_path}/{tested_file}_{mp.prediction_threshold}_unseen_test_result.csv'
            flattened_feature_columns = [f'feature_{j}' for j in range(training_data_config.sequence_length * len(feature_columns))]
            columns = ['sequence_start_date', 'sequence_end_date', 'prediction_end_date'] \
                + flattened_feature_columns \
                + ['actual_label', 'predicted_label']
            
            for i, row in enumerate(combined_data_copy):
                row.append(mp.y_pred[i])

            combined_df = pd.DataFrame(combined_data_copy, columns=columns)
            combined_df.drop(columns=flattened_feature_columns, inplace=True)
            combined_df.to_csv(report_file, index=False)

    with open(f'{report_path}/unseen_test_report.txt', 'a') as file:
        for report in reports:
            file.write(report)
            file.write('\n\n')


def test_ensemble_on_unseen_data(
        models, 
        scaler_path: str, 
        unseen_path: str, 
        report_path: str, 
        features_config: FeaturesConfig,
        training_data_config: TrainingDataConfig,
        n_estimators: int,
    ):
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
            config=training_data_config,
            collect_combined_data=True,
        )

        predictions = [model.predict(np.array(sequences)) for model in models]
        ensemble_predictions = np.sum(predictions, axis=0) > (n_estimators / 2)
        ensemble_predictions = ensemble_predictions.astype(int)
        ensemble_accuracy = accuracy_score(np.array(labels), ensemble_predictions)
        ensemble_report = classification_report(np.array(labels), ensemble_predictions)

        tested_file = df.iloc[0]['file']
        reports.append(f'{tested_file}\naccuracy={ensemble_accuracy}\n{ensemble_report}')

        report_file = f'{report_path}/{tested_file}_unseen_test_ensemble_result.csv'
        flattened_feature_columns = [f'feature_{j}' for j in range(training_data_config.sequence_length * len(feature_columns))]
        columns = ['sequence_start_date', 'sequence_end_date', 'prediction_end_date'] \
            + flattened_feature_columns \
            + ['actual_label', 'predicted_label']
        
        for i, row in enumerate(combined_data):
            row.append(ensemble_predictions[i])

        combined_df = pd.DataFrame(combined_data, columns=columns)
        combined_df.drop(columns=flattened_feature_columns, inplace=True)
        combined_df.to_csv(report_file, index=False)

    with open(f'{report_path}/unseen_test_ensemble_report.txt', 'a') as file:
        for report in reports:
            file.write(report)
            file.write('\n\n')


def format_model_predictions(predictions: List[ModelPrediction]) -> str:
    return '\n'.join(str(prediction) for prediction in predictions)

def write_test_report(
        title: str,
        label_percentages: dict[Any, float],
        training_data_config: TrainingDataConfig,
        features_config: FeaturesConfig,
        report_path: str,
        model_predictions: List[ModelPrediction]
    ):
    create_model_func_source = inspect.getsource(create_model)

    file_content = f"""
{title}

model predictions:
{format_model_predictions(model_predictions)}

training data config:
{training_data_config}

label percentages:
{label_percentages}

features config:
{features_config}

model config:
{create_model_func_source}
"""

    with open(report_path, 'w') as file:
        file.write(file_content)


def create_prediction_data(
        df: pd.DataFrame, 
        sequence_length: int, 
        feature_columns: List[str], 
    ) -> ndarray:

    sequences = []

    sequence = df[feature_columns].iloc[-sequence_length-1:-1].values
    sequences.append(sequence)
     
    return np.array(sequences)
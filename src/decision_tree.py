from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from data_preparation import (
    FeaturesConfig, 
    TrainingDataConfig,
    load_data,
    add_features,
    create_training_data,
    calculate_label_percentages,
)
from utils import NON_FEATURE_COLUMNS
import numpy as np
from sklearn.ensemble import BaggingClassifier

dataframes = load_data('src/datasets_12h/small')

features_config = FeaturesConfig()
dataframes = add_features(dataframes, features_config)

feature_columns = [col for col in dataframes[0].columns if col not in NON_FEATURE_COLUMNS]
training_data_config = TrainingDataConfig(
    sequence_length=2,
    future_candles_count=2,
    pct_increase=1,
)

sequences, labels, combined_data = create_training_data(
    dfs=dataframes, 
    feature_columns=feature_columns, 
    config=training_data_config
)

label_percentages = calculate_label_percentages(labels)
print(f'label percentages:\n{label_percentages}')

X_train, X_test, y_train, y_test = train_test_split(sequences, labels, test_size=0.2, random_state=42)
X_train = np.array(X_train).reshape(len(X_train), training_data_config.sequence_length * len(feature_columns))
y_train = np.array(y_train)
X_test = np.array(X_test).reshape(len(X_test), training_data_config.sequence_length * len(feature_columns))
y_test = np.array(y_test)

clf = BaggingClassifier(DecisionTreeClassifier(), n_estimators=500,
                        max_samples=0.4, n_jobs=-1, random_state=42)
clf.fit(X_train, y_train)

# clf = DecisionTreeClassifier(max_depth=20)
# clf.fit(X_train, y_train)

prediction = clf.predict(X_test)
report = classification_report(y_test, prediction)
score = clf.score(X_test, y_test)
print(f'{score}\n{report}')

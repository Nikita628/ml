from sklearn.datasets import load_iris
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
import matplotlib.pyplot as plt
from data_preparation import (
    FeaturesConfig, 
    TrainingDataConfig,
    load_data,
    add_features,
    create_training_data,
    create_scaler,
    normalize_data,
)
from utils import NON_FEATURE_COLUMNS
import numpy as np

dataframes = load_data('src/datasets_1d/large')

features_config = FeaturesConfig()
dataframes = add_features(dataframes, features_config)

feature_columns = [col for col in dataframes[0].columns if col not in NON_FEATURE_COLUMNS]

training_data_config = TrainingDataConfig(
    sequence_length=10,
    future_candles_count=10,
    pct_increase=5,
)

sequences, labels, combined_data = create_training_data(
    dfs=dataframes, 
    feature_columns=feature_columns, 
    config=training_data_config
)

original_array = np.random.rand(10, 5, 2)
reshaped_array = original_array.reshape(10, 5 * 2)

X_train, X_test, y_train, y_test = train_test_split(sequences, labels, test_size=0.2, random_state=42)
X_train = np.array(X_train).reshape(len(X_train), training_data_config.sequence_length * 92)
y_train = np.array(y_train)
X_test = np.array(X_test).reshape(len(X_test), training_data_config.sequence_length * 92)
y_test = np.array(y_test)

# Load dataset
# iris = load_iris()
# X, y = iris.data, iris.target

# Train the model
clf = DecisionTreeClassifier(max_depth=10)
clf.fit(X_train, y_train)

prediction = clf.predict(X_test)
report = classification_report(y_test, prediction)
score = clf.score(X_test, y_test)
print(score, report)

# Plot the tree
# plt.figure(figsize=(20,10))
# tree.plot_tree(clf, filled=True, feature_names=iris.feature_names, class_names=iris.target_names)
# plt.show()

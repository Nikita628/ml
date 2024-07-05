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
    normalize_data
)
from utils import NON_FEATURE_COLUMNS
import numpy as np
from sklearn.ensemble import BaggingClassifier
from sklearn.manifold import TSNE
import matplotlib.patheffects as PathEffects
import matplotlib as plt
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

dataframes = load_data('src/datasets_1d/small')

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
dataframes = add_features(dataframes, features_config)

feature_columns = [col for col in dataframes[0].columns if col not in NON_FEATURE_COLUMNS]
scaler = StandardScaler()
scaler.fit(pd.concat(dataframes)[feature_columns])
dataframes = normalize_data(dfs=dataframes, scaler=scaler, columns_to_normalize=feature_columns)

training_data_config = TrainingDataConfig(
    sequence_length=2,
    future_candles_count=2,
    pct_increase=3,
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

#lr = LogisticRegression(max_iter=500)
#lr.fit(X_train, y_train)
svm = SVC(kernel='rbf', random_state=1, gamma=0.10, C=10.0)
svm.fit(X_train, y_train)
y_pred = svm.predict(X_test)
y_pred = (y_pred > 0.7).astype(int)
report = classification_report(y_test, y_pred)
print(report)


# clf = BaggingClassifier(DecisionTreeClassifier(), n_estimators=500,
#                         max_samples=0.4, n_jobs=-1, random_state=42)
# clf.fit(X_train, y_train)
# clf = DecisionTreeClassifier(max_depth=20)
# clf.fit(X_train, y_train)
# prediction = clf.predict(X_test)
#report = classification_report(y_test, prediction)
# score = clf.score(X_test, y_test)
# print(f'{score}\n{report}')



# def plot_projection(x, colors):
#     f = plt.figure(figsize=(20, 20))
#     ax = plt.subplot(aspect='equal')
#     for i in range(2):
#         plt.scatter(x[colors == i, 0],
#                     x[colors == i, 1])

#     for i in range(2):
#         xtext, ytext = np.median(x[colors == i, :], axis=0)
#         txt = ax.text(xtext, ytext, str(i), fontsize=24)
#         txt.set_path_effects([
#             PathEffects.Stroke(linewidth=5, foreground="w"),
#             PathEffects.Normal()])

# tsne = TSNE(n_components=2, init='pca', random_state=123, learning_rate=1)
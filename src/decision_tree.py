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

dataframes = load_data('src/datasets_1d/prediction')

features_config = FeaturesConfig(
    sma_lengths=[2,3,5,7,10],
    ema_lengths=[2,3,5,7,10],
    rsi_lengths=[2,3,5,7,10],
    stoch_k_lengths=[2,3,5,7,10],
    stoch_d_lengths=[2,3,5,7,10],
    stoch_k_smooth=[2,3,5,7,10],
    mfi_lengths=[2,3,5,7,10],
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
    pct_increase=1,
)

sequences, labels, combined_data = create_training_data(
    dfs=dataframes, 
    feature_columns=feature_columns, 
    config=training_data_config
)

label_percentages = calculate_label_percentages(labels)
print(f'label percentages:\n{label_percentages}')

def plot_projection(x, colors):
    f = plt.figure(figsize=(20, 20))
    ax = plt.subplot(aspect='equal')
    for i in range(2):
        plt.scatter(x[colors == i, 0],
                    x[colors == i, 1])

    for i in range(2):
        xtext, ytext = np.median(x[colors == i, :], axis=0)
        txt = ax.text(xtext, ytext, str(i), fontsize=24)
        txt.set_path_effects([
            PathEffects.Stroke(linewidth=5, foreground="w"),
            PathEffects.Normal()])

tsne = TSNE(n_components=2, init='pca', random_state=123)
X_train = np.array(sequences).reshape(len(sequences), training_data_config.sequence_length * len(feature_columns))
X_digits_tsne = tsne.fit_transform(X_train)

plot_projection(X_digits_tsne, np.array(labels))
plt.show()
print()
# X_train, X_test, y_train, y_test = train_test_split(sequences, labels, test_size=0.2, random_state=42)
# X_train = np.array(X_train).reshape(len(X_train), training_data_config.sequence_length * len(feature_columns))
# y_train = np.array(y_train)
# X_test = np.array(X_test).reshape(len(X_test), training_data_config.sequence_length * len(feature_columns))
# y_test = np.array(y_test)

# clf = BaggingClassifier(DecisionTreeClassifier(), n_estimators=500,
#                         max_samples=0.4, n_jobs=-1, random_state=42)
# clf.fit(X_train, y_train)

# # clf = DecisionTreeClassifier(max_depth=20)
# # clf.fit(X_train, y_train)

# prediction = clf.predict(X_test)
# report = classification_report(y_test, prediction)
# score = clf.score(X_test, y_test)
# print(f'{score}\n{report}')

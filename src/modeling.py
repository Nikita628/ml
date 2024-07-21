import os
import numpy as np
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import KFold
from keras import models
from keras import layers
from keras import optimizers
from keras import callbacks
from sklearn.utils.class_weight import compute_class_weight
# import keras_tuner as kt

def create_model(input_shape: tuple[int, int]) -> models.Sequential:
    # add normalization layers
    model = models.Sequential()
    model.add(layers.Bidirectional(layers.LSTM(units=100, return_sequences=True), input_shape=input_shape))
    model.add(layers.Dropout(0.2))
    model.add(layers.Bidirectional(layers.LSTM(units=100, return_sequences=True)))
    model.add(layers.Dropout(0.2))
    model.add(layers.Bidirectional(layers.LSTM(units=100, return_sequences=True)))
    model.add(layers.Dropout(0.2))
    model.add(layers.Bidirectional(layers.LSTM(units=100)))
    model.add(layers.Dropout(0.2))
    model.add(layers.Dense(units=1, activation='sigmoid'))
    optimizer = optimizers.Adam(learning_rate=0.001, clipvalue=1.0)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    return model

# def create_model(input_shape: tuple[int, int]) -> models.Sequential:
#     model = models.Sequential()
#     model.add(layers.InputLayer(shape=input_shape))
#     model.add(layers.LSTM(units=100, return_sequences=True))
#     model.add(layers.Dropout(0.2))
#     model.add(layers.LSTM(units=100, return_sequences=True))
#     model.add(layers.Dropout(0.2))
#     model.add(layers.LSTM(units=100, return_sequences=True))
#     model.add(layers.Dropout(0.2))
#     model.add(layers.LSTM(units=100))
#     model.add(layers.Dropout(0.2))
#     model.add(layers.Dense(units=1, activation='sigmoid'))
#     optimizer = optimizers.Adam(learning_rate=0.001, clipvalue=1.0)
#     model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
#     return model


def get_callbacks():
    reduce_lr = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.0001)
    early_stopping = callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    return [reduce_lr, early_stopping]


def load_model(model_path='model.h5') -> models.Sequential:
    if os.path.exists(model_path):
        return models.load_model(model_path)
    else:
        raise Exception(f'{model_path} does not exist')


def train_model(model: models.Sequential, x_train, y_train, epochs=100, batch_size=32, balanced=False):
    class_weights_dict = None

    if balanced:
        class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
        class_weights_dict = {i: class_weights[i] for i in range(len(class_weights))}
        print(f'class weights: {class_weights_dict}')

    X = np.array(x_train)
    y = np.array(y_train)
    model.fit(X, y, epochs=epochs, batch_size=batch_size, 
              validation_split=0.1, callbacks=get_callbacks(), class_weight=class_weights_dict) 
    return model


def save_model(model: models.Sequential, model_path='model.h5'):
    model.save(model_path)


def cross_validate_model(model_builder, sequences, labels, n_splits=5):
    kf = KFold(n_splits=n_splits)
    accuracies = []
    reports = []

    for train_index, val_index in kf.split(sequences):
        X_train, X_val = np.array(sequences)[train_index], np.array(sequences)[val_index]
        y_train, y_val = np.array(labels)[train_index], np.array(labels)[val_index]
        
        model = model_builder(input_shape=(X_train.shape[1], X_train.shape[2]))
        model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_val, y_val), verbose=0)
        
        y_pred = (model.predict(X_val) > 0.5).astype(int)
        accuracy = accuracy_score(y_val, y_pred)
        report = classification_report(y_val, y_pred)
        
        accuracies.append(accuracy)
        reports.append(report)
        print(report)
    
    avg_accuracy = np.mean(accuracies)
    print(f'Average Accuracy: {avg_accuracy}')

    for report in reports:
        print(report)

    return accuracies, reports


# def create_tuned_model(X_train, y_train, input_shape, model_file='lstm_model.h5'):
#     X_train = np.array(X_train)
#     y_train = np.array(y_train)

#     def build_model(hp):
#         model = models.Sequential()
#         model.add(layers.Bidirectional(layers.LSTM(units=hp.Int('units_1', min_value=50, max_value=300, step=50), return_sequences=True), input_shape=input_shape))
#         model.add(layers.Dropout(hp.Float('dropout_1', min_value=0.2, max_value=0.5, step=0.1)))
        
#         model.add(layers.Bidirectional(layers.LSTM(units=hp.Int('units_2', min_value=50, max_value=300, step=50), return_sequences=True)))
#         model.add(layers.Dropout(hp.Float('dropout_2', min_value=0.2, max_value=0.5, step=0.1)))
        
#         model.add(layers.Bidirectional(layers.LSTM(units=hp.Int('units_3', min_value=50, max_value=300, step=50))))
#         model.add(layers.Dropout(hp.Float('dropout_3', min_value=0.2, max_value=0.5, step=0.1)))
        
#         model.add(layers.Dense(units=1, activation='sigmoid'))
        
#         model.compile(optimizer=optimizers.Adam(hp.Choice('learning_rate', values=[1e-5, 1e-4, 1e-3])),
#                       loss='binary_crossentropy',
#                       metrics=['accuracy'])
#         return model

#     tuner = kt.Hyperband(build_model,
#                          objective='val_accuracy',
#                          max_epochs=50,
#                          factor=3,
#                          directory='tuner',
#                          project_name='tuning_nn')

#     stop_early = callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
#     tuner.search(X_train, y_train, epochs=50, validation_split=0.1, callbacks=[stop_early])
#     best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

#     print(f"""
#     The hyperparameter search is complete. The optimal number of units in the first layer is {best_hps.get('units_1')},
#     the optimal number of units in the second layer is {best_hps.get('units_2')},
#     the optimal number of units in the third layer is {best_hps.get('units_3')},
#     the optimal dropout rate in the first layer is {best_hps.get('dropout_1')},
#     the optimal dropout rate in the second layer is {best_hps.get('dropout_2')},
#     the optimal dropout rate in the third layer is {best_hps.get('dropout_3')},
#     and the optimal learning rate for the optimizer is {best_hps.get('learning_rate')}.
#     """)

#     model = tuner.hypermodel.build(best_hps)
#     history = model.fit(X_train, y_train, epochs=50, validation_split=0.1, callbacks=[stop_early])
#     model.save(model_file)
#     return model, history

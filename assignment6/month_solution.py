# Samson Nguyen
# 1001496565
# CSE 4392 Assignment 6
import numpy as np
from tensorflow import keras
from keras.utils.np_utils import to_categorical


def data_normalization(raw_data, train_start, train_end):
    normalized_data = raw_data.copy()
    for col in range(raw_data[0].size):
        train_data = normalized_data[train_start:train_end, col]
        mean = np.mean(train_data)
        std = np.std(train_data)
        # print(col, mean, std)
        for row in range(train_start, train_end):
            normalized_data[row, col] -= mean
            normalized_data[row, col] /= std
    # print(normalized_data[train_start:train_end])
    return normalized_data


def make_inputs_and_targets(data, months, size, sampling):
    num_observations = 14 * 24 * 6 // sampling
    inputs = np.zeros((size, num_observations, 14))
    targets = np.zeros((size, 12))
    counter = 0
    months_categorical = to_categorical(months, dtype="uint8")
    # rand_input = set()
    # while len(rand_input) < size:
    #     rand_input.add(np.random.randint(0, len(data - 14 * 24 * 6)))
    while True:
        i = np.random.randint(0, len(data) - (14 * 24 * 6))
        x = []
        for n in range(i, i + 14 * 24 * 6, sampling):
            x.append(data[n])
        inputs[counter] = x.copy()
        targets[counter] = months_categorical[i + (num_observations // 2)]
        counter += 1
        if counter == size:
            break
    return inputs, targets


def build_and_train_dense(train_inputs, train_targets, val_inputs, val_targets, filename):
    input_shape = train_inputs[0].shape
    model = keras.Sequential([keras.Input(shape=input_shape),
                              keras.layers.Flatten(),
                              keras.layers.Dense(64, activation="tanh"),
                              keras.layers.Dense(32, activation="tanh"),
                              # keras.layers.Dense(16, activation="tanh"),
                              # keras.layers.Dropout(0.2),
                              keras.layers.Dense(12)
                              ])
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics='accuracy')
    # model.compile(loss='mse', optimizer='adam', metrics='accuracy')
    callbacks = [keras.callbacks.ModelCheckpoint(filepath=filename, monitor='val_accuracy', save_best_only=True)]
    training_history = model.fit(train_inputs,
                                 train_targets,
                                 epochs=10,
                                 validation_data=(val_inputs, val_targets),
                                 callbacks=callbacks)
    return training_history


def build_and_train_rnn(train_inputs, train_targets, val_inputs, val_targets, filename):
    input_shape = train_inputs[0].shape
    model = keras.Sequential([keras.Input(shape=input_shape),
                              keras.layers.Bidirectional(keras.layers.LSTM(32)),
                              # from here on you decide what to do, there are multiple correct options
                              # keras.layers.Bidirectional(keras.layers.LSTM(32)),
                              keras.layers.Dense(64, activation="relu"),
                              keras.layers.Dense(32, activation="relu"),
                              keras.layers.Dense(16, activation="relu"),
                              # keras.layers.Dense(12)
                              keras.layers.Dense(12)
                              ])
    callbacks = [keras.callbacks.ModelCheckpoint(filepath=filename, monitor='val_accuracy', save_best_only=True)]
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics='accuracy')
    # model.compile(loss='mse', optimizer='adam', metrics='accuracy')
    training_history = model.fit(train_inputs,
                                 train_targets,
                                 epochs=10,
                                 validation_data=(val_inputs, val_targets),
                                 callbacks=callbacks)
    return training_history


def test_model(filename, test_inputs, test_targets):
    model = keras.models.load_model(filename)
    # reshape input from slides
    temp = test_inputs
    # temp = np.expand_dims(temp, axis=3)
    # small_test_inputs = np.repeat(temp, 3, axis=3)
    # evaluation
    # model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    loss, acc = model.evaluate(temp, test_targets)
    return acc


def confusion_matrix(filename, test_inputs, test_targets):
    matrix = np.zeros((12, 12))
    model = keras.models.load_model(filename)
    pd = model.predict(test_inputs)
    predictions = [np.argmax(pd[i]) for i in range(len(test_inputs))]
    for i in range(len(predictions)):
        if predictions[i] != np.argmax(test_targets[i]):
            matrix[np.argmax(test_targets[i]), predictions[i]] += 1
    return matrix

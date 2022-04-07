# Samson Nguyen
# 1001496565
# CSE 4392 Assignment 6
import numpy as np
from tensorflow import keras


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
    targets = np.zeros(size)
    counter = 0
    # rand_input = set()
    # while len(rand_input) < size:
    #     rand_input.add(np.random.randint(0, len(data - 14 * 24 * 6)))
    while True:
        i = np.random.randint(0, len(data) - (14 * 24 * 6))
        x = []
        for n in range(i, i + 14 * 24 * 6, sampling):
            x.append(data[n])
        inputs[counter] = x
        targets[counter] = months[(i + num_observations) // 2]
        counter += 1
        if counter == size:
            break
    return inputs, targets


def build_and_train_dense(train_inputs, train_targets, val_inputs, val_targets, filename):
    input_shape = train_inputs[0].shape
    model = keras.Sequential([keras.Input(shape=input_shape),
                              keras.layers.Flatten(),
                              keras.layers.Dense(64, activation="tanh"),
                              # keras.layers.Dropout(0.2, input_shape=(64,)),
                              keras.layers.Dropout(0.2),
                              keras.layers.Dense(12, activation="relu"),
                              # keras.layers.Dense(16, activation="relu"),
                              # keras.layers.LSTM(16),
                              # keras.layers.Flatten(),
                              # keras.layers.Dropout(0.2),
                              keras.layers.Dense(12)
                              ])
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics='accuracy')
    callbacks = [keras.callbacks.ModelCheckpoint(filepath=filename, monitor='val_accuracy', save_best_only=True)]
    training_history = model.fit(train_inputs,
                                 train_targets,
                                 epochs=10,
                                 validation_data=(val_inputs, val_targets),
                                 validation_steps=10,
                                 callbacks=callbacks)
    # model.save(filename)
    return training_history


def build_and_train_rnn(train_inputs, train_targets, val_inputs, val_targets, filename):
    input_shape = train_inputs[0].shape
    model = keras.Sequential([keras.Input(shape=input_shape),
                              keras.layers.Bidirectional(keras.layers.LSTM(32)),
                              # from here on you decide what to do, there are multiple correct options
                              ])
    callbacks = [keras.callbacks.ModelCheckpoint(filepath=filename, monitor='val_accuracy', save_best_only=True)]
    model.compile(loss='mse', optimizer='adam', metrics='accuracy')
    training_history = model.fit(train_inputs,
                                 train_targets,
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
    model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
    loss, acc = model.evaluate(temp, test_targets)
    return acc


def confusion_matrix(filename, test_inputs, test_targets):
    matrix = np.zeros((12, 12))
    model = keras.models.load_model(filename)
    predictions = model.predict(test_inputs)
    for i in range(len(predictions)):
        if np.argmax(predictions[i]) != np.argmax(test_targets[i]):
            matrix[test_targets[i], predictions[i]] += 1
    return

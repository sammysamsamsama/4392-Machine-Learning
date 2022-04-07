# Samson Nguyen
# 1001496565
# CSE 4392 Assignment 4
import random
import sys

import keras.utils.np_utils
from keras.datasets import mnist
from keras.layers import Dense
from keras.models import Sequential

if len(sys.argv) > 1:
    layers = int(sys.argv[1])
    units_per_layer = int(sys.argv[2])
    rounds = int(sys.argv[3])
    hidden_activation = sys.argv[4]
else:
    layers = 4
    units_per_layer = 40
    rounds = 20
    hidden_activation = 'tanh'

# read MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape((x_train.shape[0], 28 * 28))
x_test = x_test.reshape((x_test.shape[0], 28 * 28))
input_shape = (28 * 28,)
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

y_train = keras.utils.np_utils.to_categorical(y_train)
y_test = keras.utils.np_utils.to_categorical(y_test)

# construct neural network
nn = Sequential()
if layers == 2:
    nn.add(Dense(10, input_shape=input_shape, activation='softmax'))
else:
    nn.add(Dense(units_per_layer, input_shape=input_shape, activation=hidden_activation))
    for i in range(layers - 2):
        nn.add(Dense(units_per_layer, activation=hidden_activation))
    nn.add(Dense(10, activation='softmax'))

nn.compile(loss='categorical_crossentropy', optimizer='adam')

# Training
nn.fit(x_train, y_train, epochs=rounds)

# Classification
object_id = 1
accuracy_total = 0
predictions = nn.predict(x_test)
for i in range(len(x_test)):
    tie = False
    true_class = list(y_test[i]).index(1)
    res_max = max(predictions[i])
    # if tie, randomly pick
    if list(predictions[i]).count(res_max) > 1:
        tie = True
        sols = []
        for i in range(len(predictions[i])):
            if predictions[i] == res_max:
                sols.append(i)
        predicted_class = random.choice(sols)
    else:
        predicted_class = list(predictions[i]).index(res_max)

    accuracy = 0
    if not tie and predicted_class == true_class:
        accuracy = 1
    elif not tie and predicted_class != true_class:
        accuracy = 0
    elif tie and predictions[i][predicted_class] == res_max:
        accuracy = 1 / predictions[i].count(res_max)
    elif tie and not predictions[i][predicted_class] == res_max:
        accuracy = 0
    print('ID=%5d, predicted=%10s, true=%10s, accuracy=%4.2f\n' % (object_id, predicted_class, true_class, accuracy))
    object_id += 1
    accuracy_total += accuracy
classification_accuracy = accuracy_total / object_id
print('classification accuracy=%6.4f\n' % (classification_accuracy))

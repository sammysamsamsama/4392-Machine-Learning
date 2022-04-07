# Samson Nguyen
# 1001496565
# CSE 4392 Assignment 4
import random
import sys

import keras.utils.np_utils
from keras.datasets import mnist
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
from keras.models import Sequential

if len(sys.argv) > 1:
    blocks = int(sys.argv[1])
    filter_size = int(sys.argv[2])
    filter_number = int(sys.argv[3])
    region_size = int(sys.argv[4])
    rounds = int(sys.argv[5])
    cnn_activation = sys.argv[6]
else:
    blocks = 2
    filter_size = 3
    filter_number = 5
    region_size = 2
    rounds = 20
    cnn_activation = 'relu'

# read MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
input_shape = (28, 28, 1)
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

y_train = keras.utils.np_utils.to_categorical(y_train)
y_test = keras.utils.np_utils.to_categorical(y_test)

# construct neural network
nn = Sequential()
if blocks == 0:
    nn.add(Dense(10, input_shape=input_shape, activation='softmax'))
else:
    nn.add(Conv2D(filter_number, kernel_size=(filter_size, filter_size), activation=cnn_activation,
                  input_shape=input_shape))
    nn.add(MaxPooling2D(pool_size=(region_size, region_size)))
    for i in range(blocks - 1):
        nn.add(Conv2D(filter_number, kernel_size=(filter_size, filter_size), activation=cnn_activation))
        nn.add(MaxPooling2D(pool_size=(region_size, region_size)))
    nn.add(Flatten())
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

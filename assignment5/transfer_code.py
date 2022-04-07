# Samson Nguyen
# 1001496565
# CSE 4392 Assignment 5
import keras
import keras.utils.np_utils
import numpy as np
from keras import layers
from keras.models import load_model


# The train_model function trains the model using the given training inputs and training labels
# (which come from the CIFAR10 dataset), using the specified batch size and number of epochs.
# For training, use Sparse Categorical Crossentropy as the loss function and Adam as the optimizer.
# You should let Keras use default values for any options that are not explicitly discussed in the task description.
# You need to decide what exactly is already done in cifar2mnist.py, and what this function needs to do so that the
# code works correctly
def train_model(model, cifar_tr_inputs, cifar_tr_labels, batch_size, epochs):
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam')
    model.fit(cifar_tr_inputs, cifar_tr_labels, batch_size=batch_size, epochs=epochs)


# The load_and_refine function does the transfer learning as discussed in the slides and in class.
# It performs these steps:
# - It loads a pre-trained model from filename.
# - It creates a new model that contains all hidden layers (and already learned weights) of the pre-trained model and a
#   new output layer with randomly initialized weights.
# - It freezes the weights of the hidden layers of the new model.
# - It trains the new model on the specified training_inputs and training_labels (which come from the MNIST dataset),
#   using the specified batch size and number of epochs. For training, use Sparse Categorical Crossentropy as the loss
#   function and Adam as the optimizer. You should let Keras use default values for any options that are not explicitly
#   discussed in the task description.
# - IMPORTANT: Your function is responsible for doing whatever pre-processing needs to be done (and not already done in
#   cifar2mnist.py) on the training inputs to make them work with this model.
# - It returns the new model.
def load_and_refine(filename, training_inputs, training_labels, batch_size, epochs):
    # reshape input from slides
    temp = training_inputs
    temp = np.expand_dims(temp, axis=3)
    small_train_inputs = np.repeat(temp, 3, axis=3)
    input_shape = small_train_inputs[0].shape
    # load pre-trained model from filename
    model = load_model(filename)
    # from slides https://athitsos.utasites.cloud/courses/cse4392_spring2022/lectures/14_transfer_learning.pdf
    num_layers = len(model.layers)
    small_num_classes = np.max(training_labels) + 1
    # create a new model with pre-trained hidden layers and new output layer
    refined_model = keras.Sequential([keras.Input(shape=input_shape)] +
                                     model.layers[0:num_layers - 1] +
                                     [layers.Dense(small_num_classes, activation='softmax')])
    # freeze hidden layer weights
    for i in range(num_layers - 1):
        refined_model.layers[i].trainable = False
    # train the new model
    refined_model.compile(loss='sparse_categorical_crossentropy', optimizer='adam')
    refined_model.fit(small_train_inputs, training_labels, batch_size=batch_size, epochs=epochs)
    return refined_model


# The evaluate_my_model function computes the classification accuracy of the model on the given test inputs and test
# labels (which come from the MNIST dataset). Your function can call the Keras model.evaluate to do the main work, but
# (similar to load_and_refine) your function is responsible for any pre-processing that needs to be done on the test
# inputs before you call model.evaluate.
def evaluate_my_model(model, test_inputs, test_labels):
    # reshape input from slides
    temp = test_inputs
    temp = np.expand_dims(temp, axis=3)
    small_test_inputs = np.repeat(temp, 3, axis=3)
    # evaluation
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    loss, acc = model.evaluate(small_test_inputs, test_labels)
    return acc

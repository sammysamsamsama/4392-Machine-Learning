# Samson Nguyen
# 1001496565
# CSE 4392 Assignment 6
import numpy as np
from tensorflow import keras

# The data_normalization function normalizes the data so that the mean for each feature is 0 and the standard deviation
# for each feature is 1. The function takes as inputs:
#   data: the time series of weather observations that the code reads from the jena_climate_2009_2016.csv file.
#   train_start, train_end: these specify, respectively, the start and end of the segment of raw_data that should be
#       used for training.
# The data_normalization function returns a time series normalized_data, that is obtained by normalizing raw_data so
# that, for each feature, the mean value over the training segment is 0, and the standard deviation over the training
# segment is 1.
def data_normalization(raw_data, train_start, train_end):
    normalized_data = raw_data.copy()
    for col in range(raw_data[0].size):
        train_data = normalized_data[train_start:train_end, col]
        mean = np.mean(train_data)
        std = np.std(train_data)
        print(col, mean, std)
        for row in range(train_start, train_end):
            normalized_data[row,col] -= mean
            normalized_data[row,col] /= std
    print(normalized_data[train_start:train_end])
    return normalized_data

# The make_inputs_and_targets function creates a a set of input objects and target values, that can be used as training,
# validation or test set. The function takes as inputs:
#   data: a time series which in our code is a segment (training, validation, or test segment) of the normalized_data
#       time series.
#   months: this is a time series of target values for data, so that months[i] is the correct month for the moment in
#       time in which data[i] was recorded. Numbers are assigned to months following the usual convention (1 for
#       January, 2 for February, etc.), except that 0 (and not 12) is assigned to December. This makes it easier later
#       to train a model, as we will have class labels that start from 0.
#   size: this specifies the size of the resulting set of inputs and targets. For example, if size == 10000, then the
#       function extracts and returns 10,000 input vectors and 10,000 target values.
#   sampling: this specifies how to sample the values in data. For example, if sampling == 6, then we sample one out of
#       every six time steps from the data time series. This reduces the length of each input vector by a factor equal to
#       sampling.
# The make_inputs_and_targets function returns two values:
#   inputs: This is a three-dimensionaly numpy array. The first dimension is equal to the argument size. inputs[i] is a
#       2D matrix containing two weeks of consecutive weather observations extracted from data, using the appropriate
#       sampling rate. To determine the number of observations (number of time steps) that inputs[i] should have to
#       cover two weeks of data, consider that the Jena Climate dataset records one observation every ten minutes, and
#       consider the sampling rate (with a sampling rate of 6, inputs[i] should only contain one observation per hour,
#       which gives a total of 336 observations). As a reminder, each observation is a 14-dimensional vector.
#   targets: These are the target values for the inputs. targets[i] should be the month corresponding to the moment at
#       which the mid-point of inputs[i] was recorded. For example, if the sampling rate is 6, inputs[i] contains 336
#       observations, and targets[i] should be the month corresponding to the moment when inputs[i][168] was recorded.
def make_inputs_and_targets(data, months, size, sampling):
    return


def build_and_train_dense(train_inputs, train_targets, val_inputs, val_targets, filename):
    return


def build_and_train_rnn(train_inputs, train_targets, val_inputs, val_targets, filename):
    return


def test_model(filename, test_inputs, test_targets):
    return


def confusion_matrix(filename, test_inputs, test_targets):
    return

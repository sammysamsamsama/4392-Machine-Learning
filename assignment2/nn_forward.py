import os
import sys
from math import exp

import numpy as np


# (layers, units, biases, weights) = nn_load(pathname)
# pathname: specifies the name of a local file
# layers: an integer, specifies the number of layers in the network.
#    units: a list of integers specifying the number of units in each layer.
#    units[L] is the number of units in layer L (where layer 1 is the input layer).
#    units[0] = None, since in our notation there is no layer 0.
#    units[1] is the number of units in the input layer
#    units[layers] is the number of units in the output layer
# biases: a list of numpy column matrices.
#    biases[L] contains the biases in layer L (where layer 1 is the input layer).
#    biases[0] = None, since in our notation there is no layer 0.
#    biases[1] = 1, since the input layer contains no perceptrons and thus no bias values
#    biases[2] contains the biases in the first hidden layer
#    biases[layers] contains the biases in the output layer.
# weights: a list of numpy matrices specifying the weights in each layer.
#    weights[L] contains the weights in layer L (where layer 1 is the input layer).
#        Its number of rows is the number of units in layer L (saved as units[L])
#        Its number of columns is the number of units in layer L-1 (saved as units[L-1])
#        Every row specifies the weights in a unit of layer L
#    weights[0] = None, since in our notation there is no layer 0.
#    weights[1] = 1, since the input layer contains no perceptrons and thus no weight values
#    weights[2] contains the weights in the first hidden layer
#    weights[layers-2] contains the biases in the output layer.
# With the exception of the input layer, each layer L is assumed to be fully
# connected, meaning that the input of each unit in layer L is connected
# to the outputs of all units in layer L-1.

def nn_load(pathname):
    if not (os.path.isfile(pathname)):
        print("read_data: %s not found", pathname)
        return None

    in_file = open(pathname)

    # get number of layers
    line = in_file.readline()
    items = line.split()
    if (len(items) != 2) or (items[0] != "layers:"):
        print("nn_load: invalid layers line %s" % (line))
        in_file.close()
        return None

    layers = int(items[1])

    # get number of units in each layer
    units = (layers + 1) * [None]
    line = in_file.readline()
    items = line.split()
    if (len(items) != layers + 1) or (items[0] != "units:"):
        print("nn_load: invalid units line %s" % (line))
        in_file.close()
        return None

    for i in range(0, layers):
        units[i + 1] = int(items[i + 1])

    # read the biases and weights for each layer
    biases = [None, None]
    weights = [None, None]

    for i in range(2, layers + 1):
        current_units = units[i]
        previous_units = units[i - 1]

        line = in_file.readline().strip()
        if (line != "start layer"):
            print("nn_load: expected 'start layer', invalid line %s" % (line))
            in_file.close()
            return None

        # read current bias
        line = in_file.readline().strip()
        if (line != "start bias"):
            print("nn_load: expected 'start bias', invalid line %s" % (line))
            in_file.close()
            return None
        b = read_matrix(in_file, 1, current_units)
        b = b.transpose();
        biases.append(b)
        line = in_file.readline().strip()
        if (line != "end bias"):
            print("nn_load: expected 'end bias', invalid line %s" % (line))
            in_file.close()
            return None

        # read current weights matrix
        line = in_file.readline().strip()
        if (line != "start w"):
            print("nn_load: expected 'start w', invalid line %s" % (line))
            in_file.close()
            return None
        w = read_matrix(in_file, current_units, previous_units)
        weights.append(w)
        # read current weights matrix
        line = in_file.readline().strip()
        if (line != "end w"):
            print("nn_load: expected 'start w', invalid line %s" % (line))
            in_file.close()
            return None

    in_file.close()
    return (layers, units, biases, weights)


# arguments:
#    in_file: source file of the data
#    rows, cols: size of the matrix that should be read
# returns:
#    result: a matrix of size rows x cols
def read_matrix(in_file, rows, cols):
    result = np.zeros((rows, cols))

    for row in range(0, rows):
        file_line = in_file.readline()
        items = file_line.split()
        cols2 = len(items)
        if (cols2 != cols):
            print("read_matrix: Line %d, %d columns expected, %d columns found" % (row, cols, cols2))
            return None
        for col in range(0, cols):
            result[row][col] = float(items[col])

    return result


def print_nn_info(layers, units, biases, weights):
    print("There are %d layers" % (layers))
    print("Units in each layer:", units[1:])
    for i in range(2, len(biases)):
        print("layer %d biases:" % (i), biases[i][:, 0])
    print()
    for i in range(2, len(weights)):
        print("layer %d weights:" % (i))
        print(weights[i])


def test_nn_load():
    default_directory = "."
    default_nn_file = "nn_xor.txt"

    if (len(sys.argv) >= 2):
        nn_file = sys.argv[1]
    else:
        nn_file = default_directory + "/" + default_nn_file

    (layers, units, biases, weights) = nn_load(nn_file)
    print_nn_info(layers, units, biases, weights)


class Perceptron:
    def __init__(self, bias, weights, activation):
        self.bias = bias
        self.weights = weights
        self.activation = activation

    def compute(self, inputs, return_a=False):
        a = self.bias + sum(w * i for w, i in zip(self.weights, inputs))
        z = 0
        if self.activation == 'step':
            z = 0 if a < 0 else 1
        elif self.activation == 'sigmoid':
            z = 1 / (1 + exp(-a))
        return a, z if return_a else z

    def __str__(self):
        return "Perceptron(" + str(self.bias) + " " + str(self.weights) + " " + str(self.activation) + ")"


# test_nn_load()
if (len(sys.argv) >= 2):
    nn_file = sys.argv[1]
else:
    nn_file = "./nn_xor.txt"
layers, units, biases, weights = nn_load(nn_file)
activation = sys.argv[3]

p_layers = [[]] + [u * [None] for u in units[1:]]
a_layers = [[]] + [u * [None] for u in units[2:]]
z_layers = [u * [None] for u in units[1:]]

for i in range(2, layers + 1):
    # print("layer", i)
    # print("units", units[i])
    for j in range(units[i]):
        # print("unit", j)
        p_layers[i][j] = Perceptron(biases[i][j][0], weights[i][j], activation)
        # print(p_layers[i][j])
p_layers = p_layers[1:]

input_file = open(sys.argv[2])
inputs = list(float(val.strip()) for val in input_file.readlines())
z_layers[0] = inputs
for layer in range(1, layers):
    a_s = []
    z_s = []
    for p in p_layers[layer]:
        a, z = p.compute(z_layers[layer - 1], return_a=True)
        # print('\t', a, z)
        a_s.append(a)
        z_s.append(z)
    a_layers[layer] = a_s
    z_layers[layer] = z_s

# print(p_layers)
# print(a_layers)
# print(z_layers)
for i in range(1, layers):
    string = "layer %d, a values: [" % i
    for a_val in a_layers[i]:
        string += "%9.4f" % a_val
    string += " ]"
    print(string)
print()
for i in range(layers):
    string = "layer %d, z values: [" % i
    for z_val in z_layers[i]:
        string += "%9.4f" % z_val
    string += " ]"
    print(string)

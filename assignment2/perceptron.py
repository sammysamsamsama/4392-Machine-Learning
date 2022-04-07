import sys
from math import exp

# get command line args
weights_file = open(sys.argv[1])
input_file = open(sys.argv[2])
activation = sys.argv[3]


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


bias = float(weights_file.readline())
weights = list(float(w.strip()) for w in weights_file.readlines())
inputs = list(float(i.strip()) for i in input_file.readlines())
p = Perceptron(bias, weights, activation)
a, z = p.compute(inputs, return_a=True)

print("a = %.4f\nz = %.4f" % (a, z))

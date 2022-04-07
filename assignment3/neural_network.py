# Samson Nguyen
# 1001496565
# CSE 4392 Assignment 3
import random
import sys
from math import exp


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
            if a > -50:
                z = 1 / (1 + exp(-a))
            else:
                z = 1
        if return_a:
            return a, z
        else:
            return z
        # return a, z if return_a is True else z

    def __str__(self):
        return "Perceptron(" + str(self.bias) + " " + str(self.weights) + " " + str(self.activation) + ")"


class Neural_Network:
    def __init__(self, layers, input_dimensionality, units_per_layer, classes):
        self.L = layers
        self.D = input_dimensionality
        self.K = classes
        self.layers = [[]] * (layers + 1)
        # layer 1: input_dimensionality
        self.layers[1] = [None] * input_dimensionality
        if self.L == 2:
            # layer L: classes, as many weights as input_dimensionality
            self.layers[self.L] = [Perceptron(bias=random.uniform(-0.05, 0.05),
                                              weights=[random.uniform(-0.05, 0.05) for n in
                                                       range(self.D)],
                                              activation='sigmoid') for c in range(self.K)]
        elif self.L > 2:
            # layers 2...L: units_per_layer, as many weights as previous layer units
            for l in range(2, self.L):
                self.layers[l] = [Perceptron(bias=random.uniform(-0.05, 0.05),
                                             weights=[random.uniform(-0.05, 0.05) for n in
                                                      range(len(self.layers[l - 1]))],
                                             activation='sigmoid') for c in range(units_per_layer)]
            # layer L: classes, as many weights as previous layer units
            self.layers[layers] = [Perceptron(bias=random.uniform(-0.05, 0.05),
                                              weights=[random.uniform(-0.05, 0.05) for n in range(units_per_layer)],
                                              activation='sigmoid') for c in range(self.K)]

    def classify(self, inputs):
        # Step 1: Initialize Input Layer
        # z contains all unit outputs
        z = [[]] * (self.L + 1)
        z[1] = [None] * (self.D + 1)
        for I in range(1, self.D + 1):
            z[1][I] = inputs[I - 1]

        # Step 2: Compute Outputs
        for l in range(2, self.L + 1):
            z[l] = [None] * (len(self.layers[l]) + 1)
            for i in range(1, len(self.layers[l]) + 1):
                z[l][i] = self.layers[l][i - 1].compute(z[l - 1][1:])

        return z[self.L][1:]

    def backpropogate(self, inputs, targets, eta):
        # Step 1: Initialize Input Layer
        # z contains all unit outputs
        z = [[]] * (self.L + 1)
        z[1] = [None] * (self.D + 1)
        for I in range(1, self.D + 1):
            z[1][I] = inputs[I - 1]

        # Step 2: Compute Outputs
        # a contains weighted sums of unit inputs
        a = [[]] * (self.L + 1)
        for l in range(2, self.L + 1):
            a[l] = [None] * (len(self.layers[l]) + 1)
            z[l] = [None] * (len(self.layers[l]) + 1)
            for i in range(1, len(self.layers[l]) + 1):
                a[l][i], z[l][i] = self.layers[l][i - 1].compute(z[l - 1][1:], return_a=True)

        # Step 3: Compute New d Values
        d = [[]] * (self.L + 1)
        d[self.L] = [None] * (self.K + 1)
        for i in range(1, self.K + 1):
            d[self.L][i] = (z[self.L][i] - targets[i - 1]) * z[self.L][i] * (1 - z[self.L][i])
        for l in range(self.L - 1, 1, -1):
            d[l] = [None] * (1 + len(self.layers[l]))
            for i in range(1, 1 + len(self.layers[l])):
                temp_sum = 0
                for k in range(1, 1 + len(self.layers[l + 1])):
                    temp_sum += d[l + 1][k] * self.layers[l + 1][k - 1].weights[i - 1]
                d[l][i] = temp_sum * z[l][i] * (1 - z[l][i])

        # Step 4: Update Weights
        for l in range(2, self.L + 1):
            for i in range(1, 1 + len(self.layers[l])):
                self.layers[l][i - 1].bias = self.layers[l][i - 1].bias - eta * d[l][i]
                for j in range(1, 1 + len(self.layers[l - 1])):
                    self.layers[l][i - 1].weights[j - 1] = self.layers[l][i - 1].weights[j - 1] - eta * d[l][i] * \
                                                           z[l - 1][j]
        return z[self.L][1:]


if len(sys.argv) > 1:
    training_file = sys.argv[1]
    test_file = sys.argv[2]
    layers = int(sys.argv[3])
    units_per_layer = int(sys.argv[4])
    rounds = int(sys.argv[5])
else:
    training_file = 'pendigits_string_training.txt'
    test_file = 'pendigits_string_test.txt'
    layers = 3
    units_per_layer = 20
    rounds = 20
dimensionality = 0
classes = set()

# read training_file
trf = open(training_file)
tsf = open(test_file)
for line in trf:
    ln = line.split()
    dimensionality = len(ln) - 1
    classes.add(ln[-1])

trf.seek(0, 0)

targets = [[]] * len(classes)
for c in range(len(classes)):
    targets[c] = [0] * len(classes)
    targets[c][c] = 1
targets = {tuple(t) for t in targets}

class_to_t = dict(zip(classes, targets))
t_to_class = dict(zip(targets, classes))

# find maximum absolute data value
max_val = None
for line in trf:
    for val in line.strip().split()[:-1]:
        if max_val is None:
            max_val = float(val)
        else:
            max_val = max(max_val, float(val))
trf.seek(0, 0)
for line in tsf:
    for val in line.strip().split()[:-1]:
        max_val = max(max_val, float(val))
tsf.seek(0, 0)

nn = Neural_Network(layers, dimensionality, units_per_layer, len(classes))
eta = 1
for n in range(rounds):
    # print('round', n)
    trf.seek(0, 0)
    for line in trf:
        l = line.strip().split()
        x = [float(xi) / max_val for xi in l[:-1]]
        t = class_to_t.get(l[-1])
        nn.backpropogate(x, t, eta)
    eta *= 0.98
trf.close()

object_id = 1
accuracy_total = 0
for line in tsf:
    tie = False
    l = line.strip().split()
    true_class = l[-1]
    x = [float(xi) / max_val for xi in l[:-1]]
    res = nn.classify(x)
    res_max = max(res)
    # if tie, randomly pick
    if res.count(res_max) > 1:
        tie = True
        sols = []
        for i in range(len(res)):
            if res[i] == res_max:
                sols.append(t_to_class.get([0] * i + [1] + [0] * (len(res) - 1 - i)))
        predicted_class = random.choice(sols)
    else:
        res_max_idx = res.index(res_max)
        prediction_out = tuple([0] * res_max_idx + [1] + [0] * (len(res) - 1 - res_max_idx))
        predicted_class = t_to_class.get(prediction_out)

    accuracy = 0
    if not tie and predicted_class == l[-1]:
        accuracy = 1
    elif not tie and predicted_class != l[-1]:
        accuracy = 0
    elif tie and res[class_to_t.get(l[-1]).index(1)] == res_max:
        accuracy = 1 / res.count(res_max)
    elif tie and not res[class_to_t.get(l[-1]).index(1)] == res_max:
        accuracy = 0
    print(
        'ID=%5d, predicted=%10s, true=%10s, accuracy=%4.2f\n' % (object_id, predicted_class, true_class, accuracy))
    object_id += 1
    accuracy_total += accuracy
tsf.close()
classification_accuracy = accuracy_total / object_id
print('classification accuracy=%6.4f\n' % (classification_accuracy))

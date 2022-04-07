# Samson Nguyen
# 1001496565
# CSE 4392 Assignment 4
import random
import sys

from keras.layers import Dense
from keras.models import Sequential

if len(sys.argv) > 1:
    training_file = sys.argv[1]
    test_file = sys.argv[2]
    layers = int(sys.argv[3])
    units_per_layer = int(sys.argv[4])
    rounds = int(sys.argv[5])
    hidden_activation = sys.argv[6]
else:
    training_file = 'pendigits_string_training.txt'
    test_file = 'pendigits_string_test.txt'
    layers = 4
    units_per_layer = 40
    rounds = 20
    hidden_activation = 'tanh'
dimensionality = 0
classes = set()

# read training_file
trf = open(training_file)
tsf = open(test_file)
for object in trf:
    ln = object.split()
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
for object in trf:
    for val in object.strip().split()[:-1]:
        if max_val is None:
            max_val = float(val)
        else:
            max_val = max(max_val, float(val))
trf.seek(0, 0)
for object in tsf:
    for val in object.strip().split()[:-1]:
        max_val = max(max_val, float(val))
tsf.seek(0, 0)

# construct neural network
nn = Sequential()
if layers == 2:
    nn.add(Dense(len(classes), input_dim=dimensionality, activation='softmax'))
else:
    nn.add(Dense(units_per_layer, input_dim=dimensionality, activation=hidden_activation))
    for i in range(layers - 2):
        nn.add(Dense(units_per_layer, activation=hidden_activation))
    nn.add(Dense(len(classes), activation='softmax'))

nn.compile(loss='categorical_crossentropy', optimizer='adam')

# Training
training_data = [line.strip().split() for line in trf]
trf.close()
X = [[float(xi) / max_val for xi in line[:-1]] for line in training_data]
t = [class_to_t.get(line[-1]) for line in training_data]
nn.fit(X, t, epochs=rounds)

# Classification
object_id = 1
accuracy_total = 0
test_data = [line.strip().split() for line in tsf]
X = [[float(xi) / max_val for xi in line[:-1]] for line in test_data]
predictions = nn.predict(X)
for i in range(len(X)):
    tie = False
    true_class = test_data[i][-1]
    res_max = max(predictions[i])
    # if tie, randomly pick
    if list(predictions[i]).count(res_max) > 1:
        tie = True
        sols = []
        for i in range(len(predictions[i])):
            if predictions[i] == res_max:
                sols.append(t_to_class.get([0] * i + [1] + [0] * (len(predictions[i]) - 1 - i)))
        predicted_class = random.choice(sols)
    else:
        res_max_idx = list(predictions[i]).index(res_max)
        prediction_out = tuple([0] * res_max_idx + [1] + [0] * (len(predictions[i]) - 1 - res_max_idx))
        predicted_class = t_to_class.get(prediction_out)

    accuracy = 0
    if not tie and predicted_class == test_data[i][-1]:
        accuracy = 1
    elif not tie and predicted_class != test_data[i][-1]:
        accuracy = 0
    elif tie and predictions[i][class_to_t.get(test_data[i][-1]).index(1)] == res_max:
        accuracy = 1 / predictions[i].count(res_max)
    elif tie and not predictions[i][class_to_t.get(test_data[i][-1]).index(1)] == res_max:
        accuracy = 0
    print('ID=%5d, predicted=%10s, true=%10s, accuracy=%4.2f\n' % (object_id, predicted_class, true_class, accuracy))
    object_id += 1
    accuracy_total += accuracy
classification_accuracy = accuracy_total / object_id
print('classification accuracy=%6.4f\n' % (classification_accuracy))

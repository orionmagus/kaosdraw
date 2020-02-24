import numpy as np
import pandas as pd
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from utils.activation_functions import truncated_normal, get_func


class NeuronLayer(object):
    def __init__(
        self,
        layer_id,
        act_func='sigmoid',
        # layer_in=None, layer_out=None, nn=None,
        shape=(6, 6),
        learning_rate=0.1,
        bias=None
    ):
        self.layer_id = layer_id
        self.shape = shape

        self.learning_rate = learning_rate
        self.bias = bias

        inputs, outputs = self.shape
        bias_node = 1 if self.bias else 0
        inputs = inputs + bias_node
        outputs = outputs
        rad = 1 / np.sqrt(inputs)
        X = truncated_normal(mean=0, sd=1, low=-rad, upp=rad)
        self.weights = X.rvs((outputs, inputs))
        self.inputs = None
        self.outputs = None
        self.errors = None

        self._act_func, self._der_func = get_func(act_func)

    def feed_forward(self, input_vector):
        # input_vector and target_vector can be tuple, list or ndarray

        bias_node = 1 if self.bias else 0
        if self.bias:
            # adding bias node to the end of the inpuy_vector
            input_vector = np.concatenate((input_vector, [self.bias]))

        input_vector = np.array(input_vector, ndmin=2).T
        #
        self.inputs = input_vector
        output_vector1 = np.dot(self.weights, input_vector)
        output_vector = self.activation_function(output_vector1)
        self.outputs = output_vector
        return output_vector

    def backpropagation(self, target_vector):
        if self.outputs is not None:
            target_vector = np.array(target_vector, ndmin=2).T
            output_errors = target_vector - self.outputs
            # update the weights:
            if self.layer_id.startswith('O'):
                self._errors_output(output_errors)
            else:
                self._errors_hidden(output_errors)
            return output_errors
        return target_vector

    def _errors_output(self, output_errors):
        tmp = output_errors * self.derivation_function(self.outputs)
        tmp = self.learning_rate * np.dot(tmp, self.inputs.T)
        self.weights += tmp

    def _errors_hidden(self, output_errors):
        # calculate hidden errors:
        _errors = np.dot(self.weights.T, output_errors)
        # update the weights:
        tmp = _errors * self.derivation_function(self.outputs)
        if self.bias:
            # ???? last element cut off, ???
            x = np.dot(tmp, input_vector.T)[:-1, :]
        else:
            x = np.dot(tmp, input_vector.T)
        self.weights += self.learning_rate * x

    def activation_function(self, x):
        return self._act_func(x)

    def derivation_function(self, x):
        return self._der_func(x)

    # data will flow through the neural network.
    def feed_forwhard(self):
        self.hidden = self.sigmoid(np.dot(self.inputs, self.weights))

    # going backwards through the network to update weights

    def backpropaghation(self):
        self.error = self.outputs - self.hidden
        delta = self.error * self.activation_function(self.hidden, deriv=True)
        self.weights += np.dot(self.inputs.T, delta)

    # train the neural net for 25,000 iterations
    def traihn(self, epochs=25000):
        for epoch in range(epochs):
            # flow forward and produce an output
            self.feed_forward()
            # go back though the network to make corrections based on the output
            self.backpropagation()
            # keep track of the error history over each epoch
            self.error_history.append(np.average(np.abs(self.error)))
            self.epoch_list.append(epoch)

    # function to predict output on new and unseen input data
    def predict(self, new_input):
        prediction = self.activation_function(np.dot(new_input, self.weights))
        return prediction


def normalize_range(val, irange=(1, 52)):
    m, mx = [x * 1.0 for x in irange]
    return (float('{}'.format(val)) - m)/mx


def denormalize_range(val, irange=(1, 52)):
    m, mx = [x * 1.0 for x in irange]
    return int(round(float('{}'.format(val)) * mx + m))


def to_nparray(balls, num=6, irange=(1, 52), brange=(1, 52), isdata=True):
    _vals = [normalize_range(x, irange) for x in balls[:num]]
    s_vals = sorted(_vals)
    if isdata is False:
        s_vals.extend([normalize_range(x, brange) for x in balls[-1:]])
    return np.array(s_vals, dtype=float)


def to_data(vec, num=6, irange=(1, 52), brange=(1, 52)):
    m, mx = irange
    # s_vals = [denormalize_range(x, irange) for x in vec.tolist()]
    s_vals = vec * mx + m
    return s_vals.astype(int)


def to_data_tensor(vec, num=6, irange=(1, 52), brange=(1, 52)):
    m, mx = irange
    vec = np.array(vec)
    # s_vals = [denormalize_range(x, irange) for x in vec.tolist()]
    s_vals = vec * mx + m
    return s_vals.astype(int)


def accuracy__match(output, target, num=6):
    match = sum([1 for n in output if n in target[:num]])
    if match >= 3:
        x = 0.5 if target[-1] in output else 0.0
        return (match+x)/(num*1.0)
    return 0.0


def accuracy_test(output, target, num=6):
    result = []
    match = 0
    for n in output:
        if isinstance(n, (list, tuple, np.ndarray)):
            result.append(accuracy_test(n, target, num))
        else:
            match = 1
    if match >= 1:
        result.append(accuracy__match(output, target, num))
    return np.average(result)


def apprec(x, cols=('ball1', 'ball2', 'ball3', 'ball4', 'ball5', 'ball6', 'bonusBall')):
    x['record'] = to_nparray([x[c] for c in cols])
    return x


def train_split(data, split_by=25, cols=('ball1', 'ball2', 'ball3', 'ball4', 'ball5', 'ball6', 'bonusBall')):
    r = data.shape[0]
    chunk_indices = [(k, min(r, k+split_by)-1, min(r, k+split_by))
                     for k in range(0, r-split_by)]
    data['record'] = 0
    data = data.apply(apprec, axis=1)
    inputs = []
    targets = []
    for s, e, t in chunk_indices:
        inputs.append(data[s:e].record.tolist())
        targets.append(data[t:t+1].record)
    ncols = ['R{}'.format(x) for x in range(1, split_by)]
    trgs = pd.concat(targets)
    return pd.DataFrame(inputs, columns=ncols, index=trgs.index)


def to_tensor(v):
    return torch.as_tensor([r.tolist() for r in v.tolist()], dtype=torch.float)


def to_target(v):
    return torch.as_tensor([r for r in v.tolist()], dtype=torch.float)


class NeuralNet(object):
    def __init__(
        self,
        shape=(24, 120, 84, 1),
        learning_rate=0.01,
        bias=None
    ):
        self.bias = bias
        bias_node = 1 if self.bias else 0
        self.shape = shape
        self.layer_shapes = [(shape[i], shape[i+1])
                             for i in range(len(shape)-1)]
        self.learning_rate = learning_rate
        self.layers = []
        self.create_layers()

    def create_layers(self):
        out = len(self.layer_shapes) - 1
        for i, shp in enumerate(self.layer_shapes):
            ins, outs = shp
            layer_id = '{}-{}-{}-{}'.format('O' if i ==
                                            out else 'H', i, ins, outs)
            self.layers.append(NeuronLayer(layer_id,
                                           shape=shp,
                                           learning_rate=self.learning_rate,
                                           bias=self.bias
                                           ))

    def feed_forward(self, input_vector):
        curr = input_vector
        for i in range(len(self.layers)):
            out = self.layers[i].feed_forward(curr)
            curr = out
        return curr

    def backpropagation(self, target_vector):
        curr = target_vector
        for i in reversed(list(range(len(self.layers)))):
            out = self.layers[i].backpropagation(curr)
            curr = out

    def train(self, training_data, epochs=250):
        for epoch in range(epochs):
            for input_vector, target_vector in training_data:
                self.feed_forward(input_vector)
                self.backpropagation(target_vector)

    def predict(self, inputs):
        curr = inputs
        for i in reversed(list(range(len(self.layers)))):
            out = self.layers[i].predict(curr)
            curr = out
        return curr


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(6, 120)  # 6*6 from image dimension
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 6)

    def forward(self, x):
        x = F.sigmoid(self.fc1(x))
        x = F.sigmoid(self.fc2(x))
        x = self.fc3(x)
        return x


"""
for i in range(len(input_tensors)):
    inputs, target = (input_tensors[i], tens_targets[i],)
    output = net(inputs)
    loss = criterion(output, target)
    loss.backward()
    optimizer.step()
"""

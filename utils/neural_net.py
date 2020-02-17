import numpy as np
import pandas as pd
from utils.activation_functions import get_func, truncated_normal
import json
from lotto.models import NeuralNetGroup, NeuralModel as NeuralNet
# activation_function = get_func('sigmoid')


def fromdt(dt, c):
    return dt.filter(regex=c).to_numpy()[1:]


def net_group(data, cols):
    nets = {k: NeuralNetwork(k) for k in cols}
    for n, net in nets.items():
        td = fromdt(data, n)
        net.train(td[:350])
        net.save()
    return nets


class NeuralNetwork(object):

    def __init__(
        self,
        net_ident,
        position=1,
        inputs=1,
        shape=(
            {'output': 12, 'activation_function': 'relu'},
            {'output': 1, 'activation_function': 'sigmoid'}
        ),
        learning_rate=0.006,
        metrics='accuracy',
        loss='accuracy',
        bias=None, node_weights=None,
        **kw
    ):
        self.net_ident = net_ident
        self.position = position
        self.bias = bias
        self.shape = shape
        self.bias_node = 1 if self.bias else 0
        self.inputs = inputs
        self.no_of_in_nodes = inputs
        h, o = [s['output'] for s in shape]
        self.no_of_hidden_nodes = h
        self.no_of_out_nodes = o
        ih, out = [s['activation_function'] for s in shape]
        self.hidden_activ, self.hidden_deriv = get_func(ih)
        self.out_activ, self.out_deriv = get_func(out)
        self.learning_rate = learning_rate
        self.errors = []
        self.metrics = metrics
        self.loss = loss
        self.metric_fn = lambda x, xx: x/xx
        self.loss_fn = lambda o, t: t-o
        if node_weights:
            for n, weights in json.loads(node_weights).items():

                setattr(self, n, np.array(weights))
        else:
            self.create_weight_matrices()

    def create_weight_matrices(self):
        """ A method to initialize the weight matrices of the neural 
        network with optional bias nodes"""

        # bias_node = 1 if self.bias else 0

        rad = 1 / np.sqrt(self.no_of_in_nodes + self.bias_node)
        X = truncated_normal(mean=0, sd=1, low=-rad, upp=rad)
        self.weights_in_hidden = X.rvs((self.no_of_hidden_nodes,
                                        self.no_of_in_nodes + self.bias_node))

        rad = 1 / np.sqrt(self.no_of_hidden_nodes + self.bias_node)
        X = truncated_normal(mean=0, sd=1, low=-rad, upp=rad)
        self.weights_hidden_out = X.rvs((self.no_of_out_nodes,
                                         self.no_of_hidden_nodes + self.bias_node))

    def save(self):
        ng = NeuralNetGroup.objects.get(pk=1)
        inf = pd.DataFrame(self.weights_in_hidden)
        inf.to_pickle('{}_in.pickle'.format(self.net_ident))
        out = pd.DataFrame(self.weights_hidden_out)
        out.to_pickle('{}_out.pickle'.format(self.net_ident))

        cfg = {
            'group': ng,
            'accuracy': np.mean(self.errors),
            'node_weights': {
                'weights_in_hidden': out.to_json(default_handler=str),
                'weights_hidden_out': out.to_json(default_handler=str),
            },
            'shape': json.dumps(self.shape),
            'training_epochs': self.epochs,
            'net_ident': str(self.net_ident),
            'tech_class': 'utils.neural_net.NeuralNetwork'
        }
        for att in ['inputs', 'position', 'loss', 'bias', 'learning_rate', 'metrics']:
            cfg[att] = getattr(self, att)
        c, r = NeuralNet.objects.get_or_create(**cfg)
        if not c:
            for n, v in cfg.items():
                setattr(r, n, v)
                r.save()

    @classmethod
    def load(cls, cfg):
        return cls(**cfg)

    def train(self, training_data, epochs=250):
        self.epochs = epochs
        for epoch in range(epochs):
            for input_vector, target_vector in training_data:
                self.train_epoch(input_vector, target_vector)

    def train_epoch(self, input_vector, target_vector):
        # input_vector and target_vector can be tuple, list or ndarray
        if self.bias_node > 0:
            # adding bias node to the end of the input_vector
            input_vector = np.concatenate((input_vector, [self.bias]))

        input_vector = np.array(input_vector, ndmin=2).T
        target_vector = np.array(target_vector, ndmin=2).T

        output_vector1 = np.dot(self.weights_in_hidden, input_vector)
        output_vector_hidden = self.hidden_activ(output_vector1)

        if self.bias:
            output_vector_hidden = np.concatenate(
                (output_vector_hidden, [[self.bias]]))

        output_vector2 = np.dot(self.weights_hidden_out, output_vector_hidden)
        output_vector_network = self.out_activ(output_vector2)

        output_errors = target_vector - output_vector_network
        # update the weights:
        tmp = output_errors * self.out_deriv(output_vector_network)
        # tmp = output_errors * output_vector_network * \
        #     (1.0 - output_vector_network)
        tmp = self.learning_rate * np.dot(tmp, output_vector_hidden.T)
        self.weights_hidden_out += tmp

        # calculate hidden errors:
        hidden_errors = np.dot(self.weights_hidden_out.T, output_errors)
        # update the weights:
        tmp = hidden_errors * self.hidden_deriv(output_vector_network)
        # tmp = hidden_errors * output_vector_hidden * \
        #     (1.0 - output_vector_hidden)
        if self.bias:
            # ???? last element cut off, ???
            x = np.dot(tmp, input_vector.T)[:-1, :]
        else:
            x = np.dot(tmp, input_vector.T)
        self.weights_in_hidden += self.learning_rate * x
        self.addmetrics(output_vector_network, target_vector)

    def addmetrics(self, prediction, target):
        self.errors.append(self.metric_fn(prediction, target))

    def predict(self, input_vector):
        # input_vector can be tuple, list or ndarray

        if self.bias:
            # adding bias node to the end of the inpuy_vector
            input_vector = np.concatenate((input_vector, [1]))
        input_vector = np.array(input_vector, ndmin=2).T

        output_vector = np.dot(self.weights_in_hidden, input_vector)
        output_vector = self.hidden_activ(output_vector)

        if self.bias:
            output_vector = np.concatenate((output_vector, [[1]]))

        output_vector = np.dot(self.weights_hidden_out, output_vector)
        output_vector = self.out_activ(output_vector)

        return output_vector

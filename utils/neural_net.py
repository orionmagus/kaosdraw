import numpy as np
import pandas as pd
from utils.activation_functions import get_func, truncated_normal

# activation_function = get_func('sigmoid')
np.savez


class NeuralNetwork(object):

    def __init__(self,
                 net_id,
                 shape=(1, 12, 1)
                 learning_rate=0.006,
                 activ_fn='sigmoid',
                 bias=None
                 ):
        self.bias = bias

        self.bias_node = 1 if self.bias else 0

        self.no_of_in_nodes,
        self.no_of_hidden_nodes,
        self.no_of_out_nodes = shape
        self.activation_function, self.derivation_function = get_func(activ_fn)
        self.learning_rate = learning_rate

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
        out = pd.DataFrame(self.weights_hidden_out)
        out.to_pickle('{}_out.pickle'.format())

    def train(self, training_data, epochs=250):
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
        output_vector_hidden = self.activation_function(output_vector1)

        if self.bias:
            output_vector_hidden = np.concatenate(
                (output_vector_hidden, [[self.bias]]))

        output_vector2 = np.dot(self.weights_hidden_out, output_vector_hidden)
        output_vector_network = self.activation_function(output_vector2)

        output_errors = target_vector - output_vector_network
        # update the weights:
        tmp = output_errors * output_vector_network * \
            (1.0 - output_vector_network)
        tmp = self.learning_rate * np.dot(tmp, output_vector_hidden.T)
        self.weights_hidden_out += tmp

        # calculate hidden errors:
        hidden_errors = np.dot(self.weights_hidden_out.T, output_errors)
        # update the weights:
        tmp = hidden_errors * output_vector_hidden * \
            (1.0 - output_vector_hidden)
        if self.bias:
            # ???? last element cut off, ???
            x = np.dot(tmp, input_vector.T)[:-1, :]
        else:
            x = np.dot(tmp, input_vector.T)
        self.weights_in_hidden += self.learning_rate * x

    def predict(self, input_vector):
        # input_vector can be tuple, list or ndarray

        if self.bias:
            # adding bias node to the end of the inpuy_vector
            input_vector = np.concatenate((input_vector, [1]))
        input_vector = np.array(input_vector, ndmin=2).T

        output_vector = np.dot(self.weights_in_hidden, input_vector)
        output_vector = self.activation_function(output_vector)

        if self.bias:
            output_vector = np.concatenate((output_vector, [[1]]))

        output_vector = np.dot(self.weights_hidden_out, output_vector)
        output_vector = self.activation_function(output_vector)

        return output_vector

import numpy as np
from scipy.stats import truncnorm, crystalball
from sklearn.model_selection import train_test_split
import sklearn.neural_network
import sklearn.metrics
@np.vectorize
def sigmoid(x):
    return 1 / (1 + np.e ** -x)


@np.vectorize
def sigmoid_deriv(x):
    return x * (1 - x)


def cball(mean=0, sd=1, low=0, upp=10):
    return crystalball(
        (low - mean) / sd, (upp - mean) / sd, loc=mean, scale=sd)


def truncated_normal(mean=0, sd=1, low=0, upp=10):
    return truncnorm(
        (low - mean) / sd, (upp - mean) / sd, loc=mean, scale=sd)


# alternative activation function
def ReLU(x):
    return np.maximum(0.0, x)

# derivation of relu


def ReLU_derivation(x):
    if x <= 0:
        return 0
    else:
        return 1


def get_func(n='sigmoid'):
    return {'sigmoid': (sigmoid, sigmoid_deriv), 'relu': (ReLU, ReLU_derivation)}.get(n, (sigmoid, sigmoid_deriv))

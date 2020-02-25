import numpy as np
from utils.activation_functions import truncated_normal, get_func
from utils.functions import hasget, dfn
from threading import Thread, Event
from queue import Queue
import zmq
from utils.zmq_helpers import _ident, pub, sub

"""
ctx=None, 
in_url='ipc:///tmp/in', 
in_topic=b'in', 
out_url='', 
out_topic=b'out',


"""
npool = 53 - np.arange(1, 53)


def non_zero(x):
    return [c for c in x if c > 0]


nmax = 2 ** 53 - 1


def from_int(x):
    return non_zero(np.where(
        np.array([c == '1' for c in list('{:0>52s}'.format(bin(x)[2:]))]), npool, 0))


def to_int(x):
    return int(''.join((1 - np.array(x)).astype(str)), 2)


def to_float_vec(x):
    return np.array(list(map(to_int, x))) / nmax


def num_result(x, size=6):
    return [x[i] for i in list(set(np.random.randint(0, len(x)-1, size+4).tolist()))[:size]]


def out_to_int(out):
    return int(round(np.sum(out) * nmax))


def output_results(out, activation, out_size=6):
    res = from_int(out_to_int(activation(out)))
    return [num_result(res) for x in range(out_size)]


def pool_accuracy(ypred, ytrue):
    na = 1 - np.array(ypred)
    nb = 1 - np.array(ytrue)
    return np.sum(na & nb)/6.0


class Neuron(Thread):
    def __init__(self,
                 in_sock, out_sock, ctl_sock,
                 chromosomes=None,
                 shape=(6, 5),
                 activation_fn='sigmoid'
                 ):
        self._id = _ident()
        a, d = get_func(activation_fn)
        if not chromosomes:
            rad = 1 / np.sqrt(shape[0])
            X = truncated_normal(mean=0, sd=1, low=-rad, upp=rad)
            chromosomes = X.rvs(shape)
        self.chromosomes = chromosomes
        self.activation = a
        self.derivation = d

        self.poller = zmq.Poller()
        self.ctl_sock = ctl_sock

        self.poller.register(self.ctl_sock, zmq.POLLIN)

        self.in_sock = in_sock
        self.poller.register(self.in_sock, zmq.POLLIN)

        self.out_sock = out_sock
        self._stop = Event()
        self.out = None

    # function using _stop function
    def stop(self, a):
        self._stop.set()

    def stopped(self):
        return self._stop.isSet()

    def run(self):
        while True:
            if self.stopped():
                return
            events = self.poller.poll()
            if self.ctl_sock in dict(events):
                call = self.ctl_sock.recv_string()
                arg = self.ctl_sock.recv_pyobj()
                method = hasget(self, call, dfn)
                method(arg)
            if self.in_sock in dict(events):
                topic = self.ctl_sock.recv_string()
                vec = self.in_sock.recv_pyobj()
                input_vector = np.array(vec, ndmin=2).T

                self.out = self.activation(
                    np.dot(self.chromosomes.T, input_vector))

    def meiosis(self):
        a, b = self.chromosomes.shape

    def mutate(self):
        pass

    def mate(self, other):
        pass

    def fitness(self, target):
        return pool_accuracy(self.out, target)

    def send(self, msg):
        pass


class Layer:
    def __init__(self, ctx=None, ident=None, out='out', inputs=5, outputs=20, num_mates=8, activ='sigmoid'):
        self.ctx = ctx or zmq.Context.instance()
        self._id = ident or _ident()
        self.in_channel = 'inproc://in{}'.format(self._id)
        self.out_channel = 'inproc://{}'.format(out)
        self.ctl_channel = 'inproc://ctl{}'.format(self._id)
        self.in_sock = pub(self.ctx, self.in_channel)
        self.out_sock = pub(self.ctx, self.out_channel)
        self.ctl_sock = pub(self.ctx, self.ctl_channel)
        self.nodes = [Neuron(
            sub(self.ctx, self.in_channel),
            self.out_sock,
            sub(self.ctx, self.ctl_channel),
            shape=(inputs, 1),
            activation_fn=activ
        ) for x in range(outputs)]

    def fwd(self, input_vector):
        ins = list(map(float, input_vector))
        self.in_sock.send_string(b'input', zmq.SNDMORE)
        self.in_sock.send_pyobj(ins)

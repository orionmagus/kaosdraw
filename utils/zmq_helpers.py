import zmq
from uuid import uuid4
import numpy as np
import re
import binascii
import os
from random import randint


def _ident(l=8):
    return uuid4().hex[:l]


def msg_from_str(string):
    return np.array(string.split()).astype(int)


def msg_to_str(msg):
    return ''.join(np.array(msg).astype(str).tolist())


def pipe(ctx):
    """build inproc pipe for talking to threads
    mimic pipe used in czmq zthread_fork.
    Returns a pair of PAIRs connected via inproc
    """
    a = ctx.socket(zmq.PAIR)
    b = ctx.socket(zmq.PAIR)
    a.linger = b.linger = 0
    a.hwm = b.hwm = 1
    iface = "inproc://%s" % binascii.hexlify(os.urandom(8))
    a.bind(iface)
    b.connect(iface)
    return a, b


def pub(ctx, iface):
    a = ctx.socket(zmq.PUB)
    a.linger = 0
    a.hwm = 1
    a.bind(iface)
    return a


def sub(ctx, iface):
    b = ctx.socket(zmq.SUB)
    b.linger = 0
    b.hwm = 1
    b.setsockopt(zmq.SUBSCRIBE, b'*')
    b.connect(iface)
    return b

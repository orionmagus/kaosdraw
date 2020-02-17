import numpy as np
import numpy.lib.mixins
from numbers import Number
from math import *
# from tensorflow.keras.metrics import binary_accuracy


def conv(n, N=52):
    n = int(n) if isinstance(n, str) else n
    n = math.ceil(n) if isinstance(n, float) else n
    n = n % N
    return N - (n-1)


def iconv(n, N=52):
    n = n % N
    return (N - n) + 1


def sconv(n, N=52):
    return str(iconv(n, N))


HANDLED_FUNCTIONS = {}


def implements(np_function):
    "Register an __array_function__ implementation for DiagonalArray objects."
    def decorator(func):
        HANDLED_FUNCTIONS[np_function] = func
        return func
    return decorator


@implements(np.sum)
def sum(arr):
    "Implementation of np.sum for DiagonalArray objects"
    return arr._i * arr._N


@implements(np.mean)
def mean(arr):
    "Implementation of np.mean for DiagonalArray objects"
    return arr._i / arr._N


def binfmt(N=52):
    return str('{:0>'+str(N)+'s}').format


def arr2str(arr):
    return ''.join(np.array(arr).astype(str))


def align_bin(val, N=52):

    return binfmt(N)(str(val))


def bin2intarr(v):
    return np.array(list(v[2:] if str(v).startswith('0b') else v)).astype(int)


def nfill(arr, N=52, val=0):
    s = [val]*(N-len(arr))
    return np.concatenate([s, arr])


def binarr_to_values(val, N=52):
    val = nfill(val, N, 0)  # bin2intarr(align_bin(arr2str(val), N))
    val = val[:N]
    if np.sum(val) > 6:
        print(val)
    return np.array([d for d in np.where(val > 0, pool(N), 0) if d > 0])


def pool(N=52):
    return N - np.arange(N)


def nmax(N=52):
    return int(str('{:0<'+str(N)+'s}').format('111111'), 2)


def sigmoid(x):
    return 1 / (1 + np.e ** -x)


def sigmoid_deriv(x):
    return x * (1 - x)


def binarr2int(arr):
    return int(''.join(np.array(arr).astype(str)), 2)


def binmax(arr):
    return np.array([1 if n > 0 else 0 for n in arr])


class NumPool(numpy.lib.mixins.NDArrayOperatorsMixin):
    def __init__(self, value=tuple(), shape=(6, 52), **kw):
        if not isinstance(value, list) or not isinstance(value, tuple):
            if isinstance(value, Number):
                if value <= 52:
                    value = [value]
        self._max, self.N = shape
        self._N = pool(self.N)
        self._cols = (
            'ball1', 'ball2', 'ball3', 'ball4', 'ball5', 'ball6',
            #  'bonusBall', 'powerBall'
        )
        self._i = []
        val = [self.draw(v) for v in value]

        for name in self._cols:
            if name in list(kw.keys()):
                self.draw(kw.get(name))

        # setattr(self, name, value)
    def get_results(self, to_numpy=True):
        if not to_numpy:
            return self._i

        return {self._cols[i]: ball.array() for i, ball in enumerate(self._i) if i < len(self._cols)}

    # def __dict__(self):
    #     return dict(zip(self._cols, self._i))

    def __array__(self):
        return np.where(np.isin(self._N, self._i), 0, 1)

    def __repr__(self):
        return f"{self.__class__.__name__}({self.N}-{self._i})"

    def draw(self, value):
        if (not value or
            len(self._i) == self._max or
                value in self._i):
            return None

        b = ball(value, N=self.N, p=np.array(self))
        self._i.append(b)
        return b

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        if method == '__call__':
            N = None
            scalars = []
            for input in inputs:
                # In this case we accept only scalar numbers or ResultsArray.
                if isinstance(input, Number):
                    scalars.append(input)
                elif isinstance(input, self.__class__):
                    scalars.append(np.array(input))
                    N = len(self._N)
                else:
                    return NotImplemented
            return self.__class__(binarr_to_values(ufunc(*scalars, **kwargs)))
        else:
            return NotImplemented


DEFAULT_POOL = pool(52)


def ball(value, N=52, p=None):
    b = BallInt(value)
    b.ext(N=N, p=p)
    return b


def array_to_ball(r):
    cr = np.ceil(r, ).astype(int)
    return ball(
        (int(''.join(np.array([1 if n > 0 else 0 for n in cr.tolist()]).astype(
            str)), 2) - 1) % len(r),
        N=len(r),
        p=1-cr
    )


class BallInt(int):
    def __init__(self, value=1):
        # super().__init__(value)
        self._pool = DEFAULT_POOL
        self.N = 52
        self.value = value

    def ext(self, N=52, p=DEFAULT_POOL):
        self._pool = p
        self.N = N

    def get_value(self):
        return self._value

    def set_value(self, value):
        if isinstance(value, Number):
            if not isinstance(value, int):
                value = ceil(float(str(value)))

        if isinstance(value, str):
            if len(value) > 3:
                value = binarr_to_values(
                    np.array(list(value)).astype(int))
        if (
            isinstance(value, list) or
            isinstance(value, tuple) or
            isinstance(value, np.ndarray) or
            hasattr(value, '__array__')
        ):
            if len(value) < self.N or np.max(np.abs(value)) > 20:
                value = ceil(np.mean(value))
            else:
                value = np.array(value) * 2 - 1
        value = value % self.N if value != self.N else self.N
        self._value = 1 if value < 1 else value

    value = property(fget=get_value, fset=set_value)

    def __repr__(self):
        return f"{self.__class__.__name__}({self.value})"

    def __str__(self):
        return ''.join(self.array().astype(str))

    def array(self):
        return self.__array__()

    def __array__(self):
        return np.where(np.isin(DEFAULT_POOL, [self.value]), 2, self._pool)

    def __index__(self):
        return super().__index__()

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        if method == '__call__':
            N = None
            s = False
            scalars = []
            for inpt in inputs:
                # In this case we accept only scalar numbers or ResultsArray.
                if isinstance(inpt, Number):
                    scalars.append(inpt)
                if isinstance(inpt, str):
                    t = self.__class__(value=inpt)
                    scalars.append(np.array(t))
                if isinstance(inpt, np.ndarray):
                    scalars.append(inpt)
                elif isinstance(inpt, self.__class__):
                    scalars.append(np.array(inpt))
                    N = self.N
                else:
                    return NotImplemented
            return self.__class__(ufunc(*scalars, **kwargs))
        else:
            print(f"NotImplemented {repr(self)} - {method}")
            return NotImplemented
    # def __or__(self, other):
    #     other=other if isinstance(
    #         other, self.__class__) else self.__class__(other)
    #     return self.__class__(int(self) | int(other))

    # def __and__(self, other):
    #     other=other if isinstance(
    #         other, self.__class__) else self.__class__(other)
    #     return self.__class__(int(self) & int(other))

    # def __not__(self):
    #     return self.__class__(~int(self))

    # def __xor__(self, other):
    #     other=other if isinstance(
    #         other, self.__class__) else self.__class__(other)
    #     return self.__class__(int(self) ^ int(other))
    # # def __div__(self, other):
    # #     other = other if isinstance(
    # #         other, self.__class__) else self.__class__(other)
    # #     return np.sum(np.array(list(bin(int(self) & int(other))[2:])).astype(int)) / 6.0

    # def __sub__(self, other):
    #     other=other if isinstance(
    #         other, self.__class__) else self.__class__(other)
    #     t, p=(self._i, other._i)
    #     return self.__class__([dgt for dgt in np.where(np.isin(t, p), 0, p).tolist() if dgt > 0])

    # def __add__(self, other):
    #     other=other if isinstance(
    #         other, self.__class__) else self.__class__(other)
    #     return self.__class__(binmax(np.array(self) + np.array(other)))

    # def __mul__(self, other):
    #     other=other if isinstance(
    #         other, self.__class__) else self.__class__(other)
    #     return self.__class__(binmax(np.array(self) * np.array(other)))

    # # def __idiv__(self, other):
    # #     return self.__div__(other)

    # def __truediv__(self, other):
    #     other=other if isinstance(
    #         other, self.__class__) else self.__class__(other)
    #     return np.sum(np.array(list(bin(int(self) & int(other))[2:])).astype(int)) / 6.0

    # def __itruediv__(self, other):
    #     return self.__truediv__(other)

    # def __isub__(self, other):
    #     return self.__sub__(other)

    # def __iadd__(self, other):
    #     return self.__add__(other)

    # def __imul__(self, other):
    #     return self.__mul__(other)


class ResultInt(object):
    def __init__(self, value=(1,)):
        self._N = pool(52)
        if not isinstance(value, list) or not isinstance(value, tuple):
            if isinstance(value, Number):
                if value <= 52:
                    value = [value]
                else:
                    value = binarr_to_values(
                        np.array(list(bin(round(value))[2:])).astype(int))
            if isinstance(value, np.ndarray):
                if np.max(value) == 1:
                    value = binarr_to_values(value)
                else:
                    value = value.tolist()
            if isinstance(value, str):
                if len(value) > 3:
                    value = binarr_to_values(
                        np.array(list(value)).astype(int))
        else:
            if len(value) > 12:
                value = binarr_to_values(
                    np.array(list(value)).astype(int))

        self._i = list(map(lambda x: x % 52 if x != 52 else 52, value))

    def __repr__(self):
        return f"{self.__class__.__name__}(N={len(self._N)}, value={self._i})"

    def __str__(self):
        return bin(int(self))[2:]

    def __array__(self):
        return np.where(np.isin(self._N, self._i), 1, 0)

    def __int__(self):
        return int(''.join(np.array(self).astype(str)), 2)

    def __index__(self):
        return int(self)

    def __or__(self, other):
        other = other if isinstance(
            other, self.__class__) else self.__class__(other)
        return self.__class__(int(self) | int(other))

    def __and__(self, other):
        other = other if isinstance(
            other, self.__class__) else self.__class__(other)
        return self.__class__(int(self) & int(other))

    def __not__(self):
        return self.__class__(~int(self))

    def __xor__(self, other):
        other = other if isinstance(
            other, self.__class__) else self.__class__(other)
        return self.__class__(int(self) ^ int(other))
    # def __div__(self, other):
    #     other = other if isinstance(
    #         other, self.__class__) else self.__class__(other)
    #     return np.sum(np.array(list(bin(int(self) & int(other))[2:])).astype(int)) / 6.0

    def __sub__(self, other):
        other = other if isinstance(
            other, self.__class__) else self.__class__(other)
        t, p = (self._i, other._i)
        return self.__class__([dgt for dgt in np.where(np.isin(t, p), 0, p).tolist() if dgt > 0])

    def __add__(self, other):
        other = other if isinstance(
            other, self.__class__) else self.__class__(other)
        return self.__class__(binmax(np.array(self) + np.array(other)))

    def __mul__(self, other):
        other = other if isinstance(
            other, self.__class__) else self.__class__(other)
        return self.__class__(binmax(np.array(self) * np.array(other)))

    # def __idiv__(self, other):
    #     return self.__div__(other)

    def __truediv__(self, other):
        other = other if isinstance(
            other, self.__class__) else self.__class__(other)
        return np.sum(np.array(list(bin(int(self) & int(other))[2:])).astype(int)) / 6.0

    def __itruediv__(self, other):
        return self.__truediv__(other)

    def __isub__(self, other):
        return self.__sub__(other)

    def __iadd__(self, other):
        return self.__add__(other)

    def __imul__(self, other):
        return self.__mul__(other)


class ResultsArray(numpy.lib.mixins.NDArrayOperatorsMixin):

    def __init__(self, N=52, value=(1, 2,)):
        self._N = pool(N)
        if len(value) == N:
            if all(np.isin(np.array(value), [0, 1])):
                value = binarr_to_values(value, N)
        self._i = value  # [conv(n, N) for n in value]
        self._m = nmax(N)

    @property
    def max_val(self):
        return self._m

    def __repr__(self):
        return f"{self.__class__.__name__}(N={len(self._N)}, value={self._i})"

    def __str__(self):
        return str(self._i)

    def __int__(self):
        return int(''.join(np.array(self).astype(str)), 2)

    def __float__(self):
        return (int(self) * 1.0) / self.max_val

    @classmethod
    def from_array(cls, val, N=52):
        return cls(N=N, value=binarr_to_values(val, N))

    @classmethod
    def from_float(cls, f, N=52):
        mx = int(''.join(np.where(np.arange(N) < 6, 1, 0).astype(str)), 2) * 1.0
        return cls.from_int(int(round(f * mx)), N)

    @classmethod
    def from_int(cls, val, N=52):
        return cls(N=N, value=[N - (i + 1) for i, n in enumerate(list(bin(val)[2:])) if n == '1'])

    def __array__(self):
        return np.where(np.isin(self._N, self._i), 1, 0)

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        if method == '__call__':
            N = None
            scalars = []
            for input in inputs:
                # In this case we accept only scalar numbers or ResultsArray.
                if isinstance(input, Number):
                    scalars.append(input)
                elif isinstance(input, self.__class__):
                    scalars.append(np.array(input))
                    N = len(self._N)
                else:
                    return NotImplemented
            return self.__class__(N, binarr_to_values(ufunc(*scalars, **kwargs)))
        else:
            return NotImplemented

#     def __array_function__(self, func, types, args, kwargs):
#        if func not in HANDLED_FUNCTIONS:
#            return NotImplemented
#        # Note: this allows subclasses that don't override
#        # __array_function__ to handle ResultsArray objects.
#        if not all(issubclass(t, self.__class__) for t in types):
#            return NotImplemented
#        return HANDLED_FUNCTIONS[func](*args, **kwargs)


def rarray(v, N=52):
    return ResultsArray(N=N, value=v)

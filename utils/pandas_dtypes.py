from pandas.api.extensions import (
    register_extension_dtype,
    ExtensionDtype,
    ExtensionScalarOpsMixin,
    ExtensionArray
)
from pandas._libs import lib
from utils.numbers import *
from utils.functions import (hasget, dfn)
import numpy as np
import re


class LottoArray(ExtensionArray, ExtensionScalarOpsMixin):
    def __init__(self, *args, **kwargs):
        value = kwargs.get('value', args[0] if len(args) > 0 else [])
        kwargs['value'] = value
        self._init(**kwargs)

    def _init(self, value=tuple(), shape=(6, 52), **kw):
        self._max, self.N = shape
        self._N = pool(self.N)
        if not isinstance(value, list) or not isinstance(value, tuple):
            if isinstance(value, Number):
                if isinstance(value, float):
                    value = int(round(value * nmax(self.N)))
                if isinstance(value, int):
                    if value <= 52:
                        value = [value]
                    else:
                        value = non_zero(
                            np.where(
                                np.array([c == '1' for c in list(
                                    '{:0>52s}'.format(bin(x)[2:]))]),
                                self._N,
                                0
                            )).tolist()

        if isinstance(value, np.ndarray):
            if np.max(value.astype(int)) == 1:
                value = non_zero(
                    np.where(
                        np.array([c == '1' for c in list(
                            '{:0>52s}'.format(''.join(value.astype(str))))]),
                        self._N,
                        0
                    ))

        self._cols = (
            'ball1', 'ball2', 'ball3', 'ball4', 'ball5', 'ball6',
            #  'bonusBall', 'powerBall'
        )
        self._i = []
        vals = [self.draw(v) for v in value]

        for name in self._cols:
            if name in list(kw.keys()):
                self.draw(kw.get(name))

    @property
    def max(self):
        return nmax(self.N) * 1.0
    # def _asdict(self):
    #     return dict(zip(self._cols, self._i))

    def __int__(self):
        return int(''.join(np.array(self).astype(str)), 2)

    def __index__(self):
        return float(self)

    def __float__(self):
        return int(self) / self.max

    def get_results(self, to_numpy=True):
        if not to_numpy:
            return self._i

        return {self._cols[i]: ball.array() for i, ball in enumerate(self._i) if i < len(self._cols)}

    def _asdict(self):
        return dict(zip(self._cols, self._i))

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
            N = self.ndim
            scalars = []
            for input in inputs:
                # In this case we accept only scalar numbers or ResultsArray.
                if isinstance(input, Number):
                    scalars.append(input)
                elif isinstance(input, self.__class__):
                    scalars.append(np.array(input))
                    N = self.ndim
                else:
                    return NotImplemented
            return self.__class__(binarr_to_values(ufunc(*scalars, **kwargs)))
        else:
            return NotImplemented

    @classmethod
    def _from_sequence(cls, scalars, dtype=None, copy=False):
        vals = np.array(scalars, dtype=dtype, copy=copy)
        _inst = cls(value=vals)
        return _inst

    @classmethod
    def _from_sequence_of_strings(cls, strings, dtype=None, copy=False):
        return cls._from_sequence(scalars, dtype=dtype, copy=copy)

    @classmethod
    def _from_factorized(cls, values, original):
        """
        Reconstruct an ExtensionArray after factorization.

        Parameters
        ----------
        values : ndarray
            An integer ndarray with the factorized values.
        original : ExtensionArray
            The original ExtensionArray that factorize was called on.

        See Also
        --------
        factorize
        ExtensionArray.factorize
        """
        return cls(values)

    # ------------------------------------------------------------------------
    # Must be a Sequence
    # ------------------------------------------------------------------------

    def __getitem__(self, item):
        # type (Any) -> Any
        """
        Select a subset of self.

        Parameters
        ----------
        item : int, slice, or ndarray
            * int: The position in 'self' to get.

            * slice: A slice object, where 'start', 'stop', and 'step' are
              integers or None

            * ndarray: A 1-d boolean NumPy ndarray the same length as 'self'

        Returns
        -------
        item : scalar or ExtensionArray

        Notes
        -----
        For scalar ``item``, return a scalar value suitable for the array's
        type. This should be an instance of ``self.dtype.type``.

        For slice ``key``, return an instance of ``ExtensionArray``, even
        if the slice is length 0 or 1.

        For a boolean mask, return an instance of ``ExtensionArray``, filtered
        to the values where ``item`` is True.
        """
        if isinstance(item, int):
            return self._i[item]
        elif isinstance(item, np.ndarray):
            return self.__class__(value=[v for ad, v in zip(item.tolist(), self._i) if ad])
        return self

    def __setitem__(self, key, value):
        """
        Set one or more values inplace.

        This method is not required to satisfy the pandas extension array
        interface.

        Parameters
        ----------
        key : int, ndarray, or slice
            When called from, e.g. ``Series.__setitem__``, ``key`` will be
            one of

            * scalar int
            * ndarray of integers.
            * boolean ndarray
            * slice object

        value : ExtensionDtype.type, Sequence[ExtensionDtype.type], or object
            value or values to be set of ``key``.

        Returns
        -------
        None
        """
        # Some notes to the ExtensionArray implementor who may have ended up
        # here. While this method is not required for the interface, if you
        # *do* choose to implement __setitem__, then some semantics should be
        # observed:
        #
        # * Setting multiple values : ExtensionArrays should support setting
        #   multiple values at once, 'key' will be a sequence of integers and
        #  'value' will be a same-length sequence.
        #
        # * Broadcasting : For a sequence 'key' and a scalar 'value',
        #   each position in 'key' should be set to 'value'.
        #
        # * Coercion : Most users will expect basic coercion to work. For
        #   example, a string like '2018-01-01' is coerced to a datetime
        #   when setting on a datetime64ns array. In general, if the
        #   __init__ method coerces that value, then so should __setitem__
        # Note, also, that Series/DataFrame.where internally use __setitem__
        # on a copy of the data.
        if isinstance(key, int):
            self._i[item] = value

        elif isinstance(key, np.ndarray):
            self._i = [v for ad, v in zip(item.tolist(), value) if ad]

    def __len__(self):
        return len(self._i)

    def __iter__(self):
        """
        Iterate over elements of the array.
        """
        # This needs to be implemented so that pandas recognizes extension
        # arrays as list-like. The default implementation makes successive
        # calls to ``__getitem__``, which may be slower than necessary.
        for i in range(len(self._i)):
            yield self[i]

    def to_numpy(self, dtype=None, copy=False, na_value=lib.no_default):
        result = np.asarray(self, dtype=dtype)
        if copy or na_value is not lib.no_default:
            result = result.copy()
        if na_value is not lib.no_default:
            result[self.isna()] = na_value
        return result

    # ------------------------------------------------------------------------
    # Required attributes
    # ------------------------------------------------------------------------

    @property
    def dtype(self):
        """
        An instance of 'ExtensionDtype'.
        """
        return LottoResult()

    @property
    def shape(self):
        """
        Return a tuple of the array dimensions.
        """
        return (len(self._N),)

    @property
    def ndim(self):
        """
        Extension Arrays are only allowed to be 1-dimensional.
        """
        return len(self._N)

    @property
    def nbytes(self):
        return 408

    # ------------------------------------------------------------------------
    # Additional Methods
    # ------------------------------------------------------------------------

    def astype(self, dtype, copy=True):

        return np.array(self._i, dtype=dtype, copy=copy)

    def isna(self):

        return np.isnan(self._i)

    def _values_for_factorize(self):
        """
        Return an array and missing value suitable for factorization.

        Returns
        -------
        values : ndarray

            An array suitable for factorization. This should maintain order
            and be a supported dtype (Float64, Int64, UInt64, String, Object).
            By default, the extension array is cast to object dtype.
        na_value : object
            The value in `values` to consider missing. This will be treated
            as NA in the factorization routines, so it will be coded as
            `na_sentinal` and not included in `uniques`. By default,
            ``np.nan`` is used.

        Notes
        -----
        The values returned by this method are also used in
        :func:`pandas.util.hash_pandas_object`.
        """
        return self.astype(object), np.nan

    def factorize(self, na_sentinel: int = -1):
        """
        Encode the extension array as an enumerated type.

        Parameters
        ----------
        na_sentinel : int, default -1
            Value to use in the `codes` array to indicate missing values.

        Returns
        -------
        codes : ndarray
            An integer NumPy array that's an indexer into the original
            ExtensionArray.
        uniques : ExtensionArray
            An ExtensionArray containing the unique values of `self`.

            .. note::

               uniques will *not* contain an entry for the NA value of
               the ExtensionArray if there are any missing values present
               in `self`.

        See Also
        --------
        factorize : Top-level factorize method that dispatches here.

        Notes
        -----
        :meth:`pandas.factorize` offers a `sort` keyword as well.
        """
        # Implementer note: There are two ways to override the behavior of
        # pandas.factorize
        # 1. _values_for_factorize and _from_factorize.
        #    Specify the values passed to pandas' internal factorization
        #    routines, and how to convert from those values back to the
        #    original ExtensionArray.
        # 2. ExtensionArray.factorize.
        #    Complete control over factorization.
        arr, na_value = self._values_for_factorize()

        codes, uniques = _factorize_array(
            arr, na_sentinel=na_sentinel, na_value=na_value
        )

        uniques = self._from_factorized(uniques, self)
        return codes, uniques

    # ------------------------------------------------------------------------
    # Indexing methods
    # ------------------------------------------------------------------------

    def take(self, indices, allow_fill=False, fill_value=None):
        from pandas.core.algorithms import take

        # If the ExtensionArray is backed by an ndarray, then
        # just pass that here instead of coercing to object.
        data = self.astype(object)

        if allow_fill and fill_value is None:
            fill_value = self.dtype.na_value

        # fill value should always be translated from the scalar
        # type for the array, to the physical storage type for
        # the data, before passing to take.

        result = take(data, indices, fill_value=fill_value,
                      allow_fill=allow_fill)
        return self._from_sequence(result, dtype=self.dtype)

    def copy(self):
        """
        Return a copy of the array.

        Returns
        -------
        ExtensionArray
        """
        return self.__class__(np.array(self._i).copy())

    def view(self, dtype=None):
        """
        Return a view on the array.

        Parameters
        ----------
        dtype : str, np.dtype, or ExtensionDtype, optional
            Default None.

        Returns
        -------
        ExtensionArray
            A view of the :class:`ExtensionArray`.
        """
        # NB:
        # - This must return a *new* object referencing the same data, not self.
        # - The only case that *must* be implemented is with dtype=None,
        #   giving a view with the same dtype as self.
        if dtype is not None:
            raise NotImplementedError(dtype)
        return self._i[:]

    # ------------------------------------------------------------------------
    # Printing
    # ------------------------------------------------------------------------

    def __repr__(self):
        from pandas.io.formats.printing import format_object_summary

        # the short repr has no trailing newline, while the truncated
        # repr does. So we include a newline in our template, and strip
        # any trailing newlines from format_object_summary
        data = format_object_summary(
            self, self._formatter(), indent_for_name=False
        ).rstrip(", \n")
        class_name = f"<{type(self).__name__}>\n"
        return f"{class_name}{data}\nLength: {len(self)}, dtype: {self.dtype}"

    def _formatter(self, boxed=False):
        if boxed:
            return str
        return repr

    # ------------------------------------------------------------------------
    # Reshaping
    # ------------------------------------------------------------------------

    @classmethod
    def _concat_same_type(
        cls, to_concat
    ):
        vals = []
        for n in to_concat:
            if isinstance(n, cls):
                vals.extend(n._i)
        return cls(vals)

    @property
    def _ndarray_values(self):

        return np.array(self._i)

    def _reduce(self, name, skipna=True, **kwargs):
        fn = hasget(np, name, dfn)
        return fn(self._i)


@register_extension_dtype
class LottoResult(ExtensionDtype):
    _metadata = ('value', 'shape', 'ball1', 'ball2',
                 'ball3', 'ball4', 'ball5', 'ball6')

    @property
    def type(self):
        return BallInt  # cls._kind

    @property
    def kind(self):
        return 'i'  # cls._kind

    @property
    def name(cls):
        return 'lotto'  # cls._kind

    @classmethod
    def construct_from_string(cls):
        return 'i'  # cls._kind

    @classmethod
    def _is_numeric(cls):
        return True

    @classmethod
    def _is_boolean(cls):
        return False

    def __init__(self, value=tuple(), shape=(6, 52), **kw):
        self._data = LottoArray(value=value, shape=shape, **kw)

    @classmethod
    def construct_from_string(cls, string):
        cols = ('ball1', 'ball2', 'ball3', 'ball4', 'ball5', 'ball6',)
        # rg = re.compile(r'[\s,]+')
        # values = dict(zip(cols, [int(s) for rg.split(string)]))

        pattern = re.compile(
            r'[\s,]+'.join(['(?P<{}>\d+)'.format(c) for c in cols]))
        match = pattern.match(string)
        if match:
            return cls(**match.groupdict())
        else:
            raise TypeError(
                f"Cannot construct a '{cls.__name__}' from '{string}'")

    @classmethod
    def construct_array_type(cls):
        """
        Return the array type associated with this dtype.
        Returns
        -------
        type
        """
        return LottoArray

import pandas as pd
from utils.pandas_dtypes import LottoResult


@pd.api.extensions.register_dataframe_accessor("lotto")
class LottoAccessor:
    def __init__(self, pandas_obj):
        self._obj = pandas_obj

        self._stype = None

    @property
    def dres(self):
        if self._stype:
            return self._stype
        cols = (
            'ball1', 'ball2', 'ball3', 'ball4', 'ball5', 'ball6',
            #  'bonusBall', 'powerBall'
        )
        self._stype = {c: hasget(self._obj, c) for c in cols}
        return self._stype

    @property
    def result(self):
        return LottoResult(**self.dres)

    def plot(self):
        # plot this array's data on a map, e.g., using Cartopy
        pass

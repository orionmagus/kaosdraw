import numpy as np


class Ball(int):
    
    def __init__(self, val='1', norm=(1, 49)):        
        super().__init__(val)
        self._min, self._max  = norm
        self._max +=1
        self._range = [x for x in range(self._min, self._max)]
        self._value = val

    def _set(self, val):
        inner = int(str(val))
        if val > 1:

    def _value(self):
        return self._inner_value
    def __int__(self):
        return int(self._inner_value)
    
    def __str__(self):
        return int(self._inner_value)
    

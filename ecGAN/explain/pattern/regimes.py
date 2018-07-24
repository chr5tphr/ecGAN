from mxnet import nd

from .base import PatternRegime

class LinearPatternRegime(PatternRegime):
    def __init__(self):
        super().__init__('linear', None)

    @staticmethod
    def condition(y):
        return nd.ones_like(y, ctx=y.context)

class PositivePatternRegime(PatternRegime):
    def __init__(self):
        super().__init__('pos', None)

    @staticmethod
    def condition(y):
        return y>=0.

class NegativePatternRegime(PatternRegime):
    def __init__(self):
        super().__init__('neg', None)

    @staticmethod
    def condition(y):
        return y<0.

class ClipPatternRegime(PatternRegime):
    def __init__(self, ymin, ymax):
        super().__init__('clip<%s,%s>'%(ymin, ymax), None)
        self._ymin = ymin
        self._ymax = ymax

    def condition(self, y):
        z = nd.ones_like(y)
        if self._ymin is not None:
            z = z * (y > self._ymin)
        if self._ymax is not None:
            z = z * (y < self._ymax)
        return z

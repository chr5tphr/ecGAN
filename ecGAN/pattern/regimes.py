from mxnet import nd

from .base import PatternRegime

class LinearPatternRegime(PatternRegime):
    def __init__(self):
        super().__init__('linear', None)

    @staticmethod
    def condition(y):
        return nd.ones_like(y, ctx=y.ctx)

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


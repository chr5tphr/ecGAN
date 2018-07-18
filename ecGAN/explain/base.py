from mxnet import nd

from ..base import Block

class Explainable(Block):
    def relevance(self, *args, **kwargs):
        method = kwargs.get('method', 'layerwise')
        func = getattr(self, 'relevance_'+method)
        R = func(*args, **kwargs)
        return R

import mxnet as mx
from mxnet import nd, gluon, autograd
from mxnet.gluon import nn
import numpy as np

from ..base import Block

class Interpretable(Block):
    def __init__(self, *args, **kwargs):
        self._isinput = kwargs.pop('isinput', False)
        super().__init__(*args, **kwargs)
        self._in = None
        self._out = None

#    def forward(self, *args, **kwargs):
#        raise NotImplementedError
#
    def relevance(self, *args, **kwargs):
        method = kwargs.pop('method', 'dtd')
        func = getattr(self, 'relevance_'+method)
        return func(*args, **kwargs)

    def relevance_sensitivity(self):
        raise NotImplementedError

    def relevance_dtd(self, a, R):
        raise NotImplementedError


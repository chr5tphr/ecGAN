import mxnet as mx
from mxnet import nd, gluon, autograd
from mxnet.gluon import nn
import numpy as np

from .func import im2col_indices
from . import base
from .explain.pattern.base import PatternNet, ActPatternNet
from .explain.pattern import layer as pattern
from .explain.layerwise.base import LayerwiseExplainable, ActLayerwiseExplainable
from .explain.layerwise import layer as layerwise
from .explain.gradient.base import GradBasedExplainable


# Linear Layers
class Dense(pattern.Dense, layerwise.Dense, BatchNormMergable):
    pass

class Conv2D(pattern.Conv2D, layerwise.Conv2D, BatchNormMergable):
    pass

class Conv2DTranspose(pattern.Conv2DTranspose, layerwise.Conv2DTranspose, BatchNormMergable):
    pass


# Activation (-esque) Layers
class Identity(pattern.Identity, layerwise.Identity):
    pass

class ReLU(pattern.ReLU, layerwise.ReLU):
    pass

class LeakyReLU(pattern.LeakyReLU, layerwise.LeakyReLU):
    pass

class Tanh(pattern.Tanh, layerwise.Tanh):
    pass

class Clip(pattern.Clip, layerwise.Clip):
    pass

class BatchNorm(pattern.BatchNorm, layerwise.BatchNorm):
    pass


# Flow Layers
class Concat(pattern.Concat, layerwise.Concat):
    pass

class Parallel(pattern.Parallel, layerwise.Parallel):
    pass

class Sequential(pattern.Sequential, layerwise.Sequential, GradBasedExplainable, Intermediate, BatchNormMergable):
    '''
        Merge batchnorm for Sequential

        returns:
            bool: whether succeeding batchnorm can be removed
    '''
    def merge_batchnorm(self, bnorm=None, ctx=None):
        children = list(self._children.values())
        ckeys = list(self._children.keys())
        retval = True
        if bnorm is not None:
            # if parent is Sequential and wants to merge BatchNorm with our last layer
            retval = hasattr(children[-1], 'merge_batchnorm') and children[-1].merge_batchnorm(bnorm, ctx=ctx)
        for cnkey, child, cnext in zip(ckeys[1:], children[:-1], children[1:]):
            if isinstance(cnext, nn.BatchNorm) and isinstance(child, BatchNormMergable) and child.merge_batchnorm(cnext, ctx=ctx):
                del self._children[cnkey]
        return retval

    def forward(self, x, depth=-1):
        rdep = depth if depth > 0 else (len(self._children.values()) + depth)
        for i, block in enumerate(self._children.values()):
            x = block(x)
            if i == rdep:
                break
        return x

    # def forward_logged(self, x, depth=-1):
    #     self._in = [x]
    #     rdep = depth if depth > 0 else (len(self._children.values()) + depth)
    #     for i, block in enumerate(self._children.values()):
    #         x = block.forward_logged(x)
    #         if i == depth:
    #             break
    #     self._out = x
    #     return self._out


import mxnet as mx
from mxnet import nd, gluon, autograd
from mxnet.gluon import nn
import numpy as np

from ... import base
from ...func import linspace, Mlist
from .base import LayerwiseExplainable, LinearLayerwiseExplainable, ActLayerwiseExplainable
from logging import getLogger

# Linear Layers
class Dense(LinearLayerwiseExplainable, base.Dense):
    pass

class Conv2D(LinearLayerwiseExplainable, base.Conv2D):
    pass

class Conv2DTranspose(LinearLayerwiseExplainable, base.Conv2DTranspose):
    pass


# Activation (-esque) Layers
class Identity(ActLayerwiseExplainable, base.Identity):
    pass

class ReLU(ActLayerwiseExplainable, base.ReLU):
    pass

class LeakyReLU(ActLayerwiseExplainable, base.LeakyReLU):
    pass

class Tanh(ActLayerwiseExplainable, base.Tanh):
    pass

class Clip(ActLayerwiseExplainable, base.Clip):
    pass

class BatchNorm(ActLayerwiseExplainable, base.BatchNorm):
    pass


# Flow Layers
class Sequential(LayerwiseExplainable, base.Sequential):
    def relevance_layerwise(self, data=None, out=None, **kwargs):
        if data is not None:
            self.forward_logged(data)
        elif self._out is None:
            raise RuntimeError('Block has not yet executed forward_logged!')

        R = self._out if out is None else out
        for child in list(self._children.values())[::-1]:
            R = child.relevance(out=R, **kwargs)
        return R

class Concat(LayerwiseExplainable, base.Concat):
    def relevance_layerwise(self, data=None, out=None, **kwargs):
        if data is not None:
            self.forward_logged(data)
        elif self._in is None:
            raise RuntimeError('Block has not yet executed forward_logged!')

        X = self._in[0]
        Rout = self._out if out is None else out
        dims = [0] + np.cumsum([x.shape[self._concat_dim] for x in X][:-1]).tolist() + [None]
        R = [Rout.slice_axis(axis=self._concat_dim, begin=begin, end=end) for begin, end in zip(dims[:-1], dims[1:])]

        return R

class Parallel(LayerwiseExplainable, base.Parallel):
    def relevance_layerwise(self, data=None, cond=None, out=None, **kwargs):
        if data is not None:
            self.forward_logged(data)
        elif self._in is None:
            raise RuntimeError('Block has not yet executed forward_logged!')

        Rout = self._out if out is None else out
        R = []
        for child, rout in zip(self._children.values(), Rout):
            R.append(child.relevance(rout, **kwargs))

        return R

class Flatten(LayerwiseExplainable, base.Flatten):
    def relevance_layerwise(self, out, *args, **kwargs):
        return out.reshape(self._in[0].shape)

class Reshape(LayerwiseExplainable, base.Reshape):
    def relevance_layerwise(self, out, *args, **kwargs):
        return out.reshape(self._in[0].shape)

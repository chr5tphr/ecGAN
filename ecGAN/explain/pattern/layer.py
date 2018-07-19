import mxnet as mx
from mxnet import nd, gluon, autograd
from mxnet.gluon import nn
import numpy as np

from ... import base
from ...func import im2col_indices, Mlist
from .base import PatternNet, LinearPatternNet, ActPatternNet

# Linear Layers
class Dense(LinearPatternNet, base.Dense):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.act is not None:
            raise RuntimeError('Inline activations are not supported!')

    def _shape_pattern(self):
        return self.weight.shape

    def _prepare_data_pattern(self):
        if self._in is None:
            raise RuntimeError('Block has not yet executed forward_logged!')
        x = self._in[0].flatten()
        y = self._out.flatten()
        yield x, y

    def _backward_pattern(self, y, pattern, pias=None):
        return nd.FullyConnected(y, pattern.T, pias, no_bias=(pias is None),
                                 num_hidden=pattern.shape[1], flatten=self._flatten)

class Conv2D(LinearPatternNet, base.Conv2D):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.act is not None:
            raise RuntimeError('Inline activations are not supported!')

    def _shape_pattern(self):
        chan = self.weight.shape[0]
        ksize = np.prod(self.weight.shape[1:])
        return (chan, ksize)

    def _prepare_data_pattern(self):
        if self._in is None:
            raise RuntimeError('Block has not yet executed forward_logged!')
        for x,y in zip(self._in[0], self._out):
            # -> patch_size x number_of_patches -> transposed
            x = im2col_indices(nd.expand_dims(x, 0), self._kwargs['kernel'][0], self._kwargs['kernel'][1], self._kwargs['pad'][0], self._kwargs['stride'][0]).T
            # -> outsize x number_of_patches -> transposed
            y = y.flatten().T
            yield x, y

    def _backward_pattern(self, y, pattern, pias=None):
        kwargs = self._kwargs.copy()
        kwargs['no_bias'] = True
        kwargs['num_filter'] = self.weight.shape[1]
        pattern = pattern.reshape(self.weight.shape)
        return nd.Deconvolution(y, pattern, name='fwd', **kwargs)

class Conv2DTranspose(LinearPatternNet, base.Conv2DTranspose):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.act is not None:
            raise RuntimeError('Inline activations are not supported!')

    def _shape_pattern(self):
        chan = self.weight.shape[0]
        ksize = np.prod(self.weight.shape[1:])
        return (chan, ksize)

    def _prepare_data_pattern(self):
        if self._in is None:
            raise RuntimeError('Block has not yet executed forward_logged!')
        for x,y in zip(self._in[0], self._out):
            # -> patch_size x number_of_patches -> transposed
            x = x.flatten().T
            # -> outsize x number_of_patches -> transposed
            y = im2col_indices(nd.expand_dims(y, 0), self._kwargs['kernel'][0], self._kwargs['kernel'][1], self._kwargs['pad'][0], self._kwargs['stride'][0]).T
            yield x, y

    def _backward_pattern(self, y, pattern, pias=None):
        kwargs = self._kwargs.copy()
        kwargs['no_bias'] = True
        kwargs['num_filter'] = self.weight.shape[0]
        kwargs.pop('adj')
        pattern = pattern.reshape(self.weight.shape)
        return nd.Convolution(y, pattern, name='fwd', **kwargs)


# Activation (-esque) Layers
class ReLU(ActPatternNet, base.ReLU):
    def backward_pattern(self, y_sig):
        return self.forward(y_sig)

class LeakyReLU(ActPatternNet, base.LeakyReLU):
    def backward_pattern(self, y_sig):
        return self.forward(y_sig)

class Identity(ActPatternNet, base.Identity):
    def backward_pattern(self, y_sig):
        return y_sig

class Tanh(ActPatternNet, base.Tanh):
    pass

class Clip(ActPatternNet, base.Clip):
    pass

class BatchNorm(ActPatternNet, base.BatchNorm):
    pass


# Flow Layers

class Sequential(PatternNet, base.Sequential):
    def init_pattern(self):
        for block in self._children.values():
            block.init_pattern()

    def forward_pattern(self, x):
        for block in self._children.values():
            x = block.forward_pattern(x)
        # WARNING: is hacky and sucks
        self._out = x
        return x

    def overload_weight_pattern(self):
        for block in self._children.values():
            block.overload_weight_pattern()

    def overload_weight_attribution_pattern(self):
        for block in self._children.values():
            block.overload_weight_attribution_pattern()

    def learn_pattern(self):
        for block in self._children.values():
            block.learn_pattern()

    def fit_pattern(self, x):
        self._err = []
        for block in self._children.values():
            x = block.fit_pattern(x)
            self._err.append(block._err)
        return x

    def fit_assess_pattern(self, x):
        self._err = []
        for block in self._children.values():
            x = block.fit_assess_pattern(x)
            self._err.append(block._err)
        return x

    def stats_assess_pattern(self):
        for block in self._children.values():
            block.stats_assess_pattern()

    def assess_pattern(self):
        quals = []
        for block in self._children.values():
            quals.append(block.assess_pattern())
        return quals

    def compute_pattern(self):
        for block in self._children.values():
            block.compute_pattern()

    def explain_pattern(self, X, attribution=False):
        X = Mlist(X)
        X.attach_grad()

        with autograd.record():
            y = self.forward_pattern(X)

        if attribution:
            self.overload_weight_attribution_pattern()
        else:
            self.overload_weight_pattern()

        y.backward(out_grad=y)
        return X.grad

    def backward_pattern(self, y_sig):
        for block in self._children.values()[::-1]:
            y_sig = block.backward_pattern(y_sig)
        return y_sig

class Parallel(PatternNet, base.Parallel):
    def init_pattern(self):
        for child in self._children.values():
            child.init_pattern()

    def forward_pattern(self, X):
        return self.forward(X)

    def overload_weight_pattern(self):
        for child in self._children.values():
            child.overload_weight_pattern()

    def overload_weight_attribution_pattern(self):
        for child in self._children.values():
            child.overload_weight_attribution_pattern()

    def learn_pattern(self):
        for child in self._children.values():
            child.learn_pattern()

    def fit_pattern(self, x, y):
        Y = []
        self._err = []
        for child in self._children.values():
            Y.append(child.fit_pattern())
            self._err.append(child._err)
        return Y

    def fit_assess_pattern(self, x, y):
        Y = []
        self._err = []
        for child in self._children.values():
            Y.append(child.fit_assess_pattern())
            self._err.append(child._err)
        return Y

    def stats_assess_pattern(self):
        for child in self._children.values():
            child.stats_assess_pattern()

    def assess_pattern(self):
        Q = []
        for child in self._children.values():
            Q.append(child.assess_pattern())
        return Q

    def compute_pattern(self):
        for child in self._children.values():
            child.compute_pattern()

    def backward_pattern(self, y_sig):
        S = []
        for child in self._children.values():
            S.append(child.backward_pattern())
        return S

class Concat(ActPatternNet, base.Concat):
    pass



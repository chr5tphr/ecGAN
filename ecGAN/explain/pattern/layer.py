import mxnet as mx
from mxnet import nd, gluon, autograd
from mxnet.gluon import nn
import numpy as np

from . import func as gfunc
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

    def _to_pattern(self, weight):
        return weight.reshape(self._shape_pattern())

    def _to_weight(self, pattern):
        return pattern.reshape(self.weight.shape)

    def _prepare_data_pattern(self):
        if self._in is None:
            raise RuntimeError('Block has not yet executed forward_logged!')
        x = self._in[0].flatten()
        y = self._out.flatten()
        yield x, y

    def _backward_pattern(self, y, pattern, pias=None):
        return nd.FullyConnected(y, pattern.T, pias, no_bias=(pias is None),
                                 num_hidden=pattern.shape[1], flatten=self._flatten)

    def _mean(self, x, y):
        return x.mean(axis=0), y.mean(axis=0)

class Conv2D(LinearPatternNet, base.Conv2D):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.act is not None:
            raise RuntimeError('Inline activations are not supported!')

    def _shape_pattern(self):
        chan = self.weight.shape[0]
        ksize = np.prod(self.weight.shape[1:])
        return (chan, ksize)

    def _to_pattern(self, weight):
        f, c, h, w = self.weight.shape
        return weight.reshape([f, c*h*w])

    def _to_weight(self, pattern):
        return pattern.reshape(self.weight.shape)

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

    # def _mean(self, x, y):
    #     ksize = list(self._kwargs['kernel_size'])
    #     kfsize = np.prod(ksize)
    #     idfilt = nd.one_hot(nd.arange(kfsize), kfsize).reshape([kfsize, 1] + ksize)
    #     kwargs = self._kwargs.copy()
    #     del kwargs['num_filter']
    #     del kwargs['no_bias']

    #     xfilts = [nd.Convolution(x[:,i:i+1], idfilt, no_bias=True, kernel=ksize, num_filter=kfsize **kwargs) for i in range(x.shape[0])]
    #     xfilts = nd.concatenate(xfilts, axis=1)
    #     return x.mean(axis=0), y.mean(axis=0)

class Conv2DTranspose(LinearPatternNet, base.Conv2DTranspose):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.act is not None:
            raise RuntimeError('Inline activations are not supported!')

    def _shape_pattern(self):
        chan = self.weight.shape[0]
        ksize = np.prod(self.weight.shape[1:])
        return (ksize, chan)

    def _to_pattern(self, weight):
        f, c, h, w = self.weight.shape
        return weight.reshape([f, c*h*w]).T

    def _to_weight(self, pattern):
        return pattern.T.reshape(self.weight.shape)

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
    def forward_pattern(self, x):
        return gfunc.leaky_relu(slope=self._alpha)(x)

class Identity(ActPatternNet, base.Identity):
    pass

class Tanh(ActPatternNet, base.Tanh):
    def forward_pattern(self, x):
        return gfunc.tanh()(x)

class Clip(ActPatternNet, base.Clip):
    pass

class BatchNorm(ActPatternNet, base.BatchNorm):
    pass

class Dropout(ActPatternNet, base.Dropout):
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

    def overload_weight_reset(self):
        for block in self._children.values():
            block.overload_weight_reset()

    def overload_weight_pattern(self):
        for block in self._children.values():
            block.overload_weight_pattern()

    def overload_weight_attribution_pattern(self):
        for block in self._children.values():
            block.overload_weight_attribution_pattern()

    def learn_pattern(self):
        for block in self._children.values():
            block.learn_pattern()

    def fit_pattern(self, x, xnb=None):
        if xnb is None:
            xnb = x
        self._err = []
        for block in self._children.values():
            x, xnb = block.fit_pattern(x, xnb)
            self._err.append(block._err)
        return x, xnb

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

    def compute_pattern(self, ctx=None):
        for block in self._children.values():
            block.compute_pattern(ctx=ctx)

    def explain_pattern(self, data, out=None, attribution=False):
        X = Mlist(data)
        X.attach_grad()

        with autograd.record():
            y = self.forward_pattern(X)

        if attribution:
            self.overload_weight_attribution_pattern()
        else:
            self.overload_weight_pattern()

        if out is None:
            out = y
        y.backward(out_grad=out)
        self.overload_weight_reset()
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

    def overload_weight_reset(self):
        for child in self._children.values():
            child.overload_weight_reset()

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

    def compute_pattern(self, ctx=None):
        for child in self._children.values():
            child.compute_pattern(ctx=ctx)

    def backward_pattern(self, y_sig):
        S = []
        for child in self._children.values():
            S.append(child.backward_pattern())
        return S

class Concat(ActPatternNet, base.Concat):
    pass

class Flatten(ActPatternNet, base.Flatten):
    pass

class Reshape(ActPatternNet, base.Reshape):
    pass

class MaxPool2D(ActPatternNet, base.MaxPool2D):
    pass

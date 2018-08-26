from mxnet.gluon import nn
from mxnet import nd

from logging import getLogger
from .func import Mlist

# Abstract
class Block(nn.Block):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._in = None
        self._out = None

    def forward_logged(self, *args, **kwargs):
        self._in = args
        self._out = self.forward(*args, **kwargs)
        return self._out

    def _forward(self, *args, **kwargs):
        raise NotImplementedError

    def sattr(self, name, value):
        # Screw MXNet's messing with __setattr__ of Blocks!
        # (Parameters are put into _reg_params when __setattr__ (see source of MXNet's Block's __setattr__))
        return object.__setattr__(self, name, value)

class Intermediate(Block):
    def forward(self, *args, depth=-1):
        return self.forward(self, *args)

class Linear(Block):
    def __init__(self, *args, **kwargs):
        self.weight = None
        self.bias = None
        super().__init__(*args, **kwargs)

class BatchNormMergable(Block):
    _weight_axis = 0
    _bias_axis = 0

    def merge_batchnorm(self, bnorm, ctx=None):
        if not isinstance(bnorm, nn.BatchNorm):
            raise RuntimeError('Cannot merge_batchnorm with type %s.'%type(bnorm))

        kwargs = bnorm._kwargs.copy()
        del kwargs['axis']

        gamma = bnorm.gamma.data(ctx=ctx)
        beta = bnorm.beta.data(ctx=ctx)
        moving_mean = bnorm.running_mean.data(ctx=ctx)
        moving_var = bnorm.running_var.data(ctx=ctx)

        wmod = nd.BatchNorm(data=self.weight.data(ctx=ctx), gamma=gamma, beta=beta.zeros_like(),
                            moving_mean=moving_mean.zeros_like(), moving_var=moving_var, axis=self._weight_axis, **kwargs)
        self.weight.set_data(wmod)

        if self.bias is not None:
            bmod = nd.BatchNorm(data=self.bias.data(ctx=ctx), gamma=gamma, beta=beta,
                                moving_mean=moving_mean, moving_var=moving_var, axis=self._bias_axis, **kwargs)
            self.bias.set_data(bmod)
        else:
            raise NotImplementedError('Adding bias to previously bias-less linear layers during BatchNorm-merging is not yet supported.')

        return True

# Linear Layers
class Dense(Linear, nn.Dense):
    def _forward(self, data, weight, bias=None):
        weight = weight.reshape(self.weight.shape)
        return nd.FullyConnected(data, weight, bias,
                                 no_bias=(bias is None),
                                 num_hidden=self._units,
                                 flatten=self._flatten)

class Conv2D(Linear, nn.Conv2D):
    def _forward(self, data, weight, bias=None):
        kwargs = self._kwargs.copy()
        kwargs['no_bias'] = bias is None
        weight = weight.reshape(self.weight.shape)
        return nd.Convolution(data, weight, bias, name='fwd', **kwargs)

class Conv2DTranspose(Linear, nn.Conv2DTranspose):
    def _forward(self, data, weight, bias=None):
        kwargs = self._kwargs.copy()
        kwargs['no_bias'] = True
        weight = weight.T.reshape(self.weight.shape)
        return nd.Deconvolution(data, weight, name='fwd', **kwargs)

# Activation (-esque) Layers
class ReLU(Block):
    def forward(self, x):
        return nd.maximum(0., x)

class LeakyReLU(Block):
    def __init__(self, alpha, **kwargs):
        super().__init__(**kwargs)
        self._alpha = alpha

    def forward(self, x):
        return nd.LeakyReLU(x, act_type='leaky', slope=self._alpha, name='fwd')

    def __repr__(self):
        s = '{name}({alpha})'
        return s.format(name=self.__class__.__name__,
                        alpha=self._alpha)

class Identity(Block):
    def forward(self, *args, **kwargs):
        return args[0]

class Tanh(Block):
    def forward(self, x):
        return nd.tanh(x)

class Clip(Block):
    def __init__(self, low=-1., high=1., **kwargs):
        super().__init__(**kwargs)
        self._low = low
        self._high = high

    def forward(self, x):
        return nd.clip(x, self._low, self._high)

class BatchNorm(Block, nn.BatchNorm):
    pass

class Dropout(Block, nn.Dropout):
    pass

# Flow Layers
class Sequential(Block, nn.Sequential):
    def forward_logged(self, x):
        self._in = x
        for child in self._children.values():
            x = child.forward_logged(x)
        self._out = x
        return x

    def forward_single_out(self, data, cond=None, logged=False):
        out = (self.forward_logged if logged else self)(data)
        if cond is None:
            cond = nd.argmax(out, axis=1)
        cond = nd.one_hot(cond, out.shape[1])
        return cond * out

    def __iadd__(self, other):
        self.add(other)
        return self

class Parallel(Block):
    def forward(self, X):
        Y = []
        for child, x in zip(self._children.values(), X):
            Y.append(child.forward(x))
        return Y

    def forward_logged(self, X):
        self._in = X
        Y = []
        for child, x in zip(self._children.values(), X):
            Y.append(child.forward_logged(x))
        self._out = Y
        return Y

class Concat(Block):
    def __init__(self, **kwargs):
        self._concat_dim = kwargs.pop('concat_dim', 1)
        super().__init__(**kwargs)

    def forward(self, X):
        return nd.concat(*X, dim=self._concat_dim)

class Flatten(Block, nn.Flatten):
    pass

class Reshape(Block):
    def __init__(self, shape, *args, **kwargs):
        super().__init__()
        self._shape = shape

    def forward(self, x):
        return x.reshape(self._shape)

class MaxPool2D(Block, nn.MaxPool2D):
    pass

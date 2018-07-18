from mxnet.gluon import nn
from mxnet import nd

from logging import getLogger
from .func import Mlist

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

    def _weight(self):
        raise NotImplementedError

class Dense(Block):
    def _forward(self, data, weight, bias=None):
        weight = weight.reshape(self.weight.shape)
        return nd.FullyConnected(data, weight, bias,
                                 no_bias=(bias is None),
                                 num_hidden=self._units,
                                 flatten=self._flatten)

    def _weight(self, ctx=None):
        return self.weight.data(ctx=ctx)

class Conv2D(Block, nn.Conv2D):
    def _forward(self, data, weight, bias=None)
        kwargs = self._kwargs.copy()
        kwargs['no_bias'] = bias is None
        weight = weight.reshape(self.weight.shape)
        return nd.Convolution(data, weight, bias, name='fwd', **kwargs)

    def _weight(self, ctx=None):
        return self.weight.data(ctx=ctx)

class Conv2DTranspose(Block, nn.Conv2DTranspose):
    def _forward(self, data, weight, bias=None):
        kwargs = self._kwargs.copy()
        kwargs['no_bias'] = True
        weight = weight.reshape(self.weight.shape)
        return nd.Deconvolution(data, weight, name='fwd', **kwargs)

    def _weight(self, ctx=None):
        return self.weight.data(ctx=ctx)

class Intermediate(Block):
    def forward(self, *args, depth=-1):
        return self.forward(self, *args)

class BatchNormMergable(Block):
    # def merge_batchnorm(self, bnorm, ctx=None):
    #     if not isinstance(bnorm, nn.BatchNorm):
    #         raise RuntimeError('Cannot merge_batchnorm with type %s.'%type(bnorm))
    #     mmod = (bnorm.gamma.data(ctx=ctx) / (bnorm.running_var.data(ctx=ctx) + bnorm._kwargs['eps'])**.5)
    #     wshape = [d if i == bnorm._kwargs['axis'] else 1 for i,d in enumerate(self.weight.shape)]
    #     print(self.weight.shape, bnorm.gamma.shape, wshape, bnorm._kwargs)
    #     # TODO shaping is wrong???
    #     wmod = self.weight.data(ctx=ctx) * mmod.reshape(wshape)
    #     self.weight.set_data(wmod)
    #     if self.bias is not None:
    #         bshape = [d if i == bnorm._kwargs['axis'] else 1 for i,d in enumerate(self.bias.shape)]
    #         bmod = (self.bias.data(ctx=ctx) - bnorm.running_mean.data(ctx=ctx).reshape(bshape)) * mmod.reshape(bshape) + bnorm.beta.data(ctx=ctx).reshape(bshape)
    #         self.bias.set_data(bmod)
    #     return True
    _outaxis = 1

    def merge_batchnorm(self, bnorm, ctx=None):
        if not isinstance(bnorm, nn.BatchNorm):
            raise RuntimeError('Cannot merge_batchnorm with type %s.'%type(bnorm))
        kwargs = bnorm._kwargs.copy()
        kwargs['axis'] = self._outaxis
        wmod = nd.BatchNorm(self.weight.data(ctx=ctx), bnorm.gamma.data(ctx=ctx), bnorm.moving_mean.data(ctx=ctx), bnorm.moving_var.data(ctx=ctx), **kwargs)
        self.weight.set_data(wmod)
        if self.bias is not None:
            bmod = nd.BatchNorm(self.bias.data(ctx=ctx), bnorm.gamma.data(ctx=ctx), bnorm.moving_mean.data(ctx=ctx), bnorm.moving_var.data(ctx=ctx), **kwargs)
            self.bias.set_data(wmod)
        return True

class Sequential(Block, nn.Sequential):
    def forward_logged(self, x):
        self._in = x
        for child in self._children.values():
            x = child.forward_logged(x)
        self._out = x
        return x

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


class YSequential(Block):
    _Subnet = None
    def __init__(self, **kwargs):
        self._concat_dim = kwargs.pop('concat_dim', 1)
        super().__init__(**kwargs)
        with self.name_scope():
            self._data_net = self._Subnet()
            self.register_child(self._data_net)

            self._cond_net = self._Subnet()
            self.register_child(self._cond_net)

            self._main_net = self._Subnet()
            self.register_child(self._main_net)

    def addData(self, *args, **kwargs):
        with self._data_net.name_scope():
            self._data_net.add(*args, **kwargs)

    def addCond(self, *args, **kwargs):
        with self._cond_net.name_scope():
            self._cond_net.add(*args, **kwargs)

    def add(self, *args, **kwargs):
        with self._main_net.name_scope():
            self._main_net.add(*args, **kwargs)

    def forward(self, *args):
        # WARNING This is hacky and nowhere standardized
        if len(args) == 1:
            x,y = args[0]
        else:
            x,y = args

        data = self._data_net(x)
        cond = self._cond_net(y)
        combo = nd.concat(data, cond, dim=self._concat_dim)
        return self._main_net.forward(combo)

    def forward_logged(self, x, y):
        self._in = [x, y]

        data = self._data_net.forward_logged(x)
        cond = self._cond_net.forward_logged(y)
        combo = nd.concat(data, cond, dim=self._concat_dim)
        self._out = self._main_net.forward_logged(combo)
        return self._out

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


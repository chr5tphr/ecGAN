import mxnet as mx
from mxnet import nd, gluon, autograd
from mxnet.gluon import nn
import numpy as np

from .func import im2col_indices
from .base import Block, Intermediate
from .pattern.base import PatternNet
from .pattern.layer import SequentialPatternNet, BatchNormPatternNet, DensePatternNet, Conv2DPatternNet, Conv2DTransposePatternNet
from .explain.base import Interpretable
from .explain.layer import SequentialInterpretable, YSequentialInterpretable


class Dense(DenseInterpretable, DensePatternNet):
    pass

class Conv2D(Conv2DInterpretable, Conv2DPatternNet):
    pass

class Conv2DTranspose(Conv2DTransposePatternNet, Conv2DTransposeInterpretable):
    pass

class BatchNorm(BatchNormInterpretable, BatchNormPatternNet):
    pass


class Identity(Interpretable, PatternNet, Block):
    def forward(self, *args, **kwargs):
        return args[0]

    def relevance(self, *args, **kwargs):
        return args[0]

    def init_pattern(self, *args, **kwargs):
        pass

    def learn_pattern(self, *args, **kwargs):
        pass

    def assess_pattern(self, *args, **kwargs):
        pass

    def forward_pattern(self, *args, **kwargs):
        x = args[0]
        if len(args) < 2:
            x_neut = x
        else:
            x_neut = args[1]
        return self.forward(x)

class Clip(Interpretable, Block):
    def forward(self, x):
        return nd.clip(x, 0., 1.)
    def relevance(self, R):
        return R

class LeakyReLU(Interpretable, PatternNet, nn.LeakyReLU):
    def init_pattern(self, *args, **kwargs):
        pass

    def learn_pattern(self, *args, **kwargs):
        pass

    def assess_pattern(self, *args, **kwargs):
        pass

    def forward_pattern(self, *args, **kwargs):
        x = args[0]
        if len(args) < 2:
            x_neut = x
        else:
            x_neut = args[1]
        z = nd.where(x_neut>0, x, self._alpha * x)
        z_neut = self.forward(x_neut)
        return z, z_neut

class Activation(Interpretable, nn.Activation):
    pass

class MaxOut(Block):
    pass


class SequentialIntermediate(Intermediate, nn.Sequential):
    def forward(self, x, depth=-1):
        rdep = depth if depth > 0 else (len(self._children) + depth)
        for i, block in enumerate(self._children):
            x = block(x)
            if i == depth:
                break
        return x

class Sequential(SequentialInterpretable, SequentialPatternNet, SequentialIntermediate):
    pass

class YSequentialBase(Block):
    def __init__(self, **kwargs):
        self._concat_dim = kwargs.pop('concat_dim', 1)
        super().__init__(**kwargs)
        with self.name_scope():
            self._data_net = Sequential()
            self.register_child(self._data_net)

            self._cond_net = Sequential()
            self.register_child(self._cond_net)

            self._main_net = Sequential()
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

    def forward(self, x, y):
        data = self._data_net(x)
        cond = self._cond_net(y)
        combo = nd.concat(data, cond, dim=self._concat_dim)
        return self._main_net.forward(combo)

class YSequentialIntermediate(Intermediate, YSequentialBase):
    def forward(self, x, y, depth=-1):
        data = self._data_net(x)
        cond = self._cond_net(y)
        combo = nd.concat(data, cond, dim=self._concat_dim)
        return self._main_net.forward(combo, depth=depth)

class YSequential(YSequentialInterpretable, YSequentialIntermediate):
    pass

import mxnet as mx
from mxnet import nd, gluon, autograd
from mxnet.gluon import nn
import numpy as np

from .func import im2col_indices
from .base import Block, Intermediate, YSequentialBase, ReLUBase, TanhBase
from .pattern.base import PatternNet, ActPatternNet
from .pattern.layer import SequentialPatternNet, YSequentialPatternNet, DensePatternNet, Conv2DPatternNet, Conv2DTransposePatternNet, ReLUPatternNet, IdentityPatternNet
from .explain.base import Interpretable
from .explain.layer import SequentialInterpretable, YSequentialInterpretable, DenseInterpretable, Conv2DTransposeInterpretable, Conv2DInterpretable, BatchNormInterpretable


class Dense(DenseInterpretable, DensePatternNet):
    pass

class Conv2D(Conv2DInterpretable, Conv2DPatternNet):
    pass

class Conv2DTranspose(Conv2DTransposePatternNet, Conv2DTransposeInterpretable):
    pass

class BatchNorm(ActPatternNet, BatchNormInterpretable):
    pass

class Identity(IdentityPatternNet):
    pass

class Clip(Interpretable, Block):
    def forward(self, x):
        return nd.clip(x, 0., 1.)
    def relevance(self, R):
        return R

class LeakyReLU(Interpretable, ActPatternNet, nn.LeakyReLU):
    pass

class Activation(Interpretable, nn.Activation):
    pass

class MaxOut(Block):
    pass

class ReLU(ReLUPatternNet):
    pass

class Tanh(ActPatternNet, TanhBase):
    pass

class MaxPool2D(ActPatternNet, nn.MaxPool2D):
    pass

class SequentialIntermediate(Intermediate, nn.Sequential):
    def forward(self, x, depth=-1):
        rdep = depth if depth > 0 else (len(self._children.values()) + depth)
        for i, block in enumerate(self._children.values()):
            x = block(x)
            if i == depth:
                break
        return x

class Sequential(SequentialInterpretable, SequentialPatternNet, SequentialIntermediate):
    pass

class YSequentialIntermediate(Intermediate, YSequentialBase):
    def forward(self, x, y, depth=-1):
        data = self._data_net(x)
        cond = self._cond_net(y)
        combo = nd.concat(data, cond, dim=self._concat_dim)
        return self._main_net.forward(combo, depth=depth)

class YSequential(YSequentialInterpretable, YSequentialPatternNet, YSequentialIntermediate):
    _Subnet = Sequential


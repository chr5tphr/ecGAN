import mxnet as mx
from mxnet import nd, gluon, autograd
from mxnet.gluon import nn
import numpy as np

from .func import im2col_indices
from .base import Block, Intermediate, YSequentialBase, SequentialBase, ReLUBase, TanhBase
from .pattern.base import PatternNet, ActPatternNet
from .pattern.layer import SequentialPatternNet, YSequentialPatternNet, DensePatternNet, Conv2DPatternNet,\
                           Conv2DTransposePatternNet, ReLUPatternNet, IdentityPatternNet, ParallelPatternNet
from .explain.base import Interpretable, ActInterpretable
from .explain.layer import SequentialInterpretable, YSequentialInterpretable, DenseInterpretable,\
                           Conv2DTransposeInterpretable, Conv2DInterpretable, BatchNormInterpretable,\
                           ParallelInterpretable, ConcatInterpretable


class Dense(DenseInterpretable, DensePatternNet):
    pass

class Conv2D(Conv2DInterpretable, Conv2DPatternNet):
    pass

class Conv2DTranspose(Conv2DTransposePatternNet, Conv2DTransposeInterpretable):
    pass

class BatchNorm(ActPatternNet, BatchNormInterpretable):
    pass

class Identity(IdentityPatternNet, ActInterpretable):
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

class ReLU(ReLUPatternNet, ActInterpretable):
    pass

class Tanh(ActPatternNet, TanhBase):
    pass

class MaxPool2D(ActPatternNet, nn.MaxPool2D):
    pass

class Concat(ConcatInterpretable, ActPatternNet):
    pass

class Parallel(ParallelInterpretable, ConcatInterpretable):
    pass

class SequentialIntermediate(Intermediate, SequentialBase):
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


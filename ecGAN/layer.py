import mxnet as mx
from mxnet import nd, gluon, autograd
from mxnet.gluon import nn
import numpy as np

from .func import im2col_indices
from .base import Block, Intermediate, YSequentialBase, SequentialBase, ReLUBase, TanhBase, ClipBase,\
                  BatchNormMergable
from .pattern.base import PatternNet, ActPatternNet
from .pattern.layer import SequentialPatternNet, YSequentialPatternNet, DensePatternNet, Conv2DPatternNet,\
                           Conv2DTransposePatternNet, ReLUPatternNet, IdentityPatternNet, ParallelPatternNet,\
                           LeakyReLUPatternNet
from .explain.base import Interpretable, ActInterpretable
from .explain.layer import SequentialInterpretable, YSequentialInterpretable, DenseInterpretable,\
                           Conv2DTransposeInterpretable, Conv2DInterpretable, BatchNormInterpretable,\
                           ParallelInterpretable, ConcatInterpretable


class Dense(DenseInterpretable, DensePatternNet, BatchNormMergable):
    pass

class Conv2D(Conv2DInterpretable, Conv2DPatternNet, BatchNormMergable):
    pass

class Conv2DTranspose(Conv2DTransposePatternNet, Conv2DTransposeInterpretable, BatchNormMergable):
    pass

class BatchNorm(ActPatternNet, BatchNormInterpretable):
    pass

class Identity(IdentityPatternNet, ActInterpretable):
    pass

class Activation(Interpretable, nn.Activation):
    pass

class MaxOut(Block):
    pass

class ReLU(ReLUPatternNet, ActInterpretable):
    pass

class LeakyReLU(LeakyReLUPatternNet, ActInterpretable):
    pass

class Tanh(ActPatternNet, TanhBase):
    pass

class BatchNorm(nn.BatchNorm, ActPatternNet, ActInterpretable):
    pass

class Clip(ClipBase, ActPatternNet, ActInterpretable):
    pass

class MaxPool2D(ActPatternNet, nn.MaxPool2D):
    pass

class Concat(ConcatInterpretable, ActPatternNet):
    pass

class Parallel(ParallelInterpretable, ParallelPatternNet):
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

class Sequential(SequentialInterpretable, SequentialPatternNet, SequentialIntermediate, BatchNormMergable):
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

class YSequentialIntermediate(Intermediate, YSequentialBase):
    def forward(self, x, y, depth=-1):
        data = self._data_net(x)
        cond = self._cond_net(y)
        combo = nd.concat(data, cond, dim=self._concat_dim)
        return self._main_net.forward(combo, depth=depth)

class YSequential(YSequentialInterpretable, YSequentialPatternNet, YSequentialIntermediate):
    _Subnet = Sequential


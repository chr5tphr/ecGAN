import mxnet as mx
from mxnet import nd, gluon, autograd
from mxnet.gluon import nn
import numpy as np

from .base import PatternNet, ActPatternNet
from ..base import ReLUBase, IdentityBase
from ..func import im2col_indices

class DensePatternNet(PatternNet, nn.Dense):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.act is not None:
            raise RuntimeError('Inline activations are not supported!')

    def _shape_pattern(self):
        return self.weight.shape

    def _weight_pattern(self):
        return self.weight.data().flatten()

    def learn_pattern(self):
        if self._in is None:
            raise RuntimeError('Block has not yet executed forward_logged!')
        x = self._in[0].flatten()
        y = self._out.flatten()
        self._learn_pattern(x, y)

    def _forward_pattern(self, x, w):
        return nd.FullyConnected(x, w, None, no_bias=True, num_hidden=self._units, flatten=self._flatten)

    def _backward_pattern(self, y, pattern):
        return nd.FullyConnected(y, pattern.T, no_bias=True,
                                 num_hidden=pattern.shape[1], flatten=self._flatten)

class Conv2DPatternNet(PatternNet, nn.Conv2D):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.act is not None:
            raise RuntimeError('Inline activations are not supported!')

    def _shape_pattern(self):
        chan = self.weight.shape[0]
        ksize = np.prod(self.weight.shape[1:])
        return (chan, ksize)

    def _weight_pattern(self):
        return self.weight.data().flatten()

    def learn_pattern(self):
        if self._in is None:
            raise RuntimeError('Block has not yet executed forward_logged!')
        for x,y in zip(self._in[0], self._out):
            # -> patch_size x number_of_patches -> transposed
            x = im2col_indices(nd.expand_dims(x, 0), self._kwargs['kernel'][0], self._kwargs['kernel'][1], self._kwargs['pad'][0], self._kwargs['stride'][0]).T
            # -> outsize x number_of_patches -> transposed
            y = y.flatten().T
            self._learn_pattern(x, y)

    def _forward_pattern(self, x, w):
        kwargs = self._kwargs
        kwargs['no_bias'] = True
        w = w.reshape(self.weight.shape)
        return nd.Convolution(x, w, name='fwd', **kwargs)

    def _backward_pattern(self, y, pattern):
        kwargs = self._kwargs
        kwargs['no_bias'] = True
        kwargs['num_filter'] = self.weight.shape[1]
        pattern = pattern.reshape(self.weight.shape)
        return nd.Deconvolution(y, pattern, name='fwd', **kwargs)

class Conv2DTransposePatternNet(PatternNet, nn.Conv2DTranspose):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.act is not None:
            raise RuntimeError('Inline activations are not supported!')

    def _shape_pattern(self):
        chan = self.weight.shape[0]
        ksize = np.prod(self.weight.shape[1:])
        return (chan, ksize)

    def _weight_pattern(self):
        return self.weight.data().flatten()

    def learn_pattern_linear(self):
        if self._in is None:
            raise RuntimeError('Block has not yet executed forward_logged!')
        for x,y in zip(self._in[0], self._out):
            # -> patch_size x number_of_patches -> transposed
            x = x.flatten().T
            # -> outsize x number_of_patches -> transposed
            y = im2col_indices(nd.expand_dims(y, 0), self._kwargs['kernel'][0], self._kwargs['kernel'][1], self._kwargs['pad'][0], self._kwargs['stride'][0]).T
            self._learn_pattern_linear(x, y)

    def _forward_pattern(self, x, w):
        kwargs = self._kwargs
        kwargs['no_bias'] = True
        w = w.reshape(self.weight.shape)
        return nd.Deconvolution(x, w, name='fwd', **kwargs)

    def _backward_pattern(self, y, pattern):
        kwargs = self._kwargs
        kwargs['no_bias'] = True
        kwargs['num_filter'] = self.weight.shape[0]
        pattern = pattern.reshape(self.weight.shape)
        return nd.Convolution(y, pattern, name='fwd', **kwargs)

class BatchNormPatternNet(PatternNet, nn.BatchNorm):
    def init_pattern(self, *args):
        pass

    def learn_pattern(self, *args):
        pass

    def compute_pattern(self):
        pass

    def fit_pattern(self, x):
        return self(x)

    # TODO: do this correctly
    def forward_pattern(self, *args):
        x_neut, x_acc, x_regs = self._args_forward_pattern(*args)

        z_neut = self.forward(x_neut)
        z_acc = self.forward(x_acc)
        z_regs = {}
        for reg_name, x_reg in x_regs.items():
            z_regs[reg_name] = self.forward(x_reg)

        return z_neut, z_acc, z_regs

class SequentialPatternNet(PatternNet, nn.Sequential):
    def init_pattern(self):
        for block in self._children:
            block.init_pattern()

    def forward_pattern(self, *args):
        x = args
        for block in self._children:
            x = block.forward_pattern(*x)
        return x

    def learn_pattern(self):
        for block in self._children:
            block.learn_pattern()

    def fit_pattern(self, x):
        for block in self._children:
            x = block.fit_pattern(x)
        return x

    def compute_pattern(self):
        for block in self._children:
            block.compute_pattern()

    def explain_pattern(self, *args, num_reg=1):
        x = args[0]
        x.attach_grad()
        with autograd.record():
            y = self.forward_pattern(x)
        #y[1].backward(out_grad=y[0])
        y[1].backward()
        return x.grad

class ReLUPatternNet(ActPatternNet, ReLUBase):
    def _forward_pattern(self, x_neut, x_reg):
        return nd.where(x_neut>=0., x_reg, nd.zeros_like(x_neut, ctx=x_neut.context))

class IdentityPatternNet(ActPatternNet, IdentityBase):
    def _forward_pattern(self, x_neut, x_reg):
        return x_reg


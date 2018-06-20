import mxnet as mx
from mxnet import nd, gluon, autograd
from mxnet.gluon import nn
import numpy as np

from .base import PatternNet, ActPatternNet
from ..base import ReLUBase, IdentityBase, YSequentialBase
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

    def _prepare_data_pattern(self):
        if self._in is None:
            raise RuntimeError('Block has not yet executed forward_logged!')
        x = self._in[0].flatten()
        y = self._out.flatten()
        yield x, y

    def _forward_pattern(self, x, w, bias=None):
        return nd.FullyConnected(x, w, bias, no_bias=(bias is None),
                                 num_hidden=self._units, flatten=self._flatten)

    def _backward_pattern(self, y, pattern, pias=None):
        return nd.FullyConnected(y, pattern.T, pias, no_bias=(pias is None),
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

    def _prepare_data_pattern(self):
        if self._in is None:
            raise RuntimeError('Block has not yet executed forward_logged!')
        for x,y in zip(self._in[0], self._out):
            # -> patch_size x number_of_patches -> transposed
            x = im2col_indices(nd.expand_dims(x, 0), self._kwargs['kernel'][0], self._kwargs['kernel'][1], self._kwargs['pad'][0], self._kwargs['stride'][0]).T
            # -> outsize x number_of_patches -> transposed
            y = y.flatten().T
            yield x, y

    def _forward_pattern(self, x, w, bias=None):
        kwargs = self._kwargs.copy()
        kwargs['no_bias'] = bias is None
        w = w.reshape(self.weight.shape)
        return nd.Convolution(x, w, bias, name='fwd', **kwargs)

    def _backward_pattern(self, y, pattern, pias=None):
        kwargs = self._kwargs.copy()
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

    def _prepare_data_pattern(self):
        if self._in is None:
            raise RuntimeError('Block has not yet executed forward_logged!')
        for x,y in zip(self._in[0], self._out):
            # -> patch_size x number_of_patches -> transposed
            x = x.flatten().T
            # -> outsize x number_of_patches -> transposed
            y = im2col_indices(nd.expand_dims(y, 0), self._kwargs['kernel'][0], self._kwargs['kernel'][1], self._kwargs['pad'][0], self._kwargs['stride'][0]).T
            yield x, y

    def _forward_pattern(self, x, w):
        kwargs = self._kwargs.copy()
        kwargs['no_bias'] = True
        w = w.reshape(self.weight.shape)
        return nd.Deconvolution(x, w, name='fwd', **kwargs)

    def _backward_pattern(self, y, pattern):
        kwargs = self._kwargs.copy()
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
        for block in self._children.values():
            block.init_pattern()

    def forward_pattern(self, *args):
        x = args
        for block in self._children.values():
            x = block.forward_pattern(*x)
        return x

    def forward_attribution_pattern(self, *args):
        x = args
        for block in self._children.values():
            x = block.forward_attribution_pattern(*x)
        return x

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

    def explain_pattern(self, *args):
        x = args[0]
        x.attach_grad()
        with autograd.record():
            y = self.forward_pattern(x)
        y[1].backward(out_grad=y[0])
        return x.grad

    def explain_attribution_pattern(self, *args):
        x = args[0]
        x.attach_grad()
        with autograd.record():
            y = self.forward_attribution_pattern(x)
        y[1].backward(out_grad=y[0])
        return x.grad

    def backward_pattern(self, y_sig):
        for block in self._children.values()[::-1]:
            y_sig = block.backward_pattern(y_sig)
        return y_sig

class YSequentialPatternNet(PatternNet, YSequentialBase):
    def init_pattern(self):
        self._data_net.init_pattern()
        self._cond_net.init_pattern()
        self._main_net.init_pattern()

    def forward_pattern(self, x, y):
        data = self._data_net.forward_pattern(x)
        cond = self._cond_net.forward_pattern(y)
        combo = nd.concat(data, cond, dim=self._concat_dim)
        return self._main_net.forward_pattern(combo)

    def forward_attribution_pattern(self, x, y):
        data = self._data_net.forward_attribution_pattern(x)
        cond = self._cond_net.forward_attribution_pattern(y)
        combo = nd.concat(data, cond, dim=self._concat_dim)
        return self._main_net.forward_attribution_pattern(combo)

    def learn_pattern(self):
        self._data_net.learn_pattern()
        self._cond_net.learn_pattern()
        self._main_net.learn_pattern()

    def fit_pattern(self, x, y):
        data = self._data_net.fit_pattern(x)
        cond = self._cond_net.fit_pattern(y)
        combo = nd.concat(data, cond, dim=self._concat_dim)
        out = self._main_net.fit_pattern(combo)
        self._err = self._data_net._err + self._cond_net._err + self._main_net._err
        return out

    def fit_assess_pattern(self, x, y):
        data = self._data_net.fit_assess_pattern(x)
        cond = self._cond_net.fit_assess_pattern(y)
        combo = nd.concat(data, cond, dim=self._concat_dim)
        out = self._main_net.fit_assess_pattern(combo)
        self._err = self._data_net._err + self._cond_net._err + self._main_net._err
        return out

    def stats_assess_pattern(self):
        self._data_net.stats_assess_pattern()
        self._cond_net.stats_assess_pattern()
        self._main_net.stats_assess_pattern()

    def assess_pattern(self):
        quals = self._data_net.assess_pattern() \
              + self._cond_net.assess_pattern() \
              + self._main_net.assess_pattern()
        return quals

    def compute_pattern(self):
        self._data_net.compute_pattern()
        self._cond_net.compute_pattern()
        self._main_net.compute_pattern()

    def explain_pattern(self, x, y):
        x.attach_grad()
        with autograd.record():
            z = self.forward_pattern(x, y)
        z[1].backward(out_grad=z[0])
        return x.grad

    def explain_attribution_pattern(self, x, y):
        x.attach_grad()
        with autograd.record():
            z = self.forward_attribution_pattern(x, y)
        z[1].backward(out_grad=z[0])
        return x.grad

    def backward_pattern(self, y_sig):
        y_sig_dc = self._main_net.backward_pattern(y_sig)

        dim_d = self._data_net._out.shape[self._concat_dim]

        y_sid_d = y_sig_dc.slice_axis(axis=self._concat_dim, begin=0, end=dim_d)
        y_sid_c = y_sig_dc.slice_axis(axis=self._concat_dim, begin=dim_d, end=None)

        x_sig_d = self._data_net.backward_pattern(y_sid_d)
        x_sig_c = self._cond_net.backward_pattern(y_sid_c)

        return x_sig_d, x_sig_c

class ReLUPatternNet(ActPatternNet, ReLUBase):
    def _forward_pattern(self, x_neut, x_reg):
        return nd.where(x_neut>=0., x_reg, nd.zeros_like(x_neut, ctx=x_neut.context))

    def backward_pattern(self, y_sig):
        if self._out is None:
            raise RuntimeError('Block has not yet executed forward_logged!')
        y_neut = self._out
        return self._forward_pattern(y_neut, y_sig)

class IdentityPatternNet(ActPatternNet, IdentityBase):
    def _forward_pattern(self, x_neut, x_reg):
        return x_reg

    def backward_pattern(self, y_sig):
        return y_sig


import mxnet as mx
from mxnet import nd, gluon, autograd
from mxnet.gluon import nn
import numpy as np

from .base import Interpretable
from ..base import YSequentialBase
from ..func import linspace

class DenseInterpretable(Interpretable, nn.Dense):
    def _forward_interpretable(self, x, w, bias=None):
        return nd.FullyConnected(x, w, bias,
                                 no_bias=(bias is None),
                                 num_hidden=self._units,
                                 flatten=self._flatten)

    def _weight_interpretable(self):
        return self.weight.data()

class Conv2DInterpretable(Interpretable, nn.Conv2D):
    def _forward_interpretable(self, x, w, bias=None):
        kwargs = self._kwargs.copy()
        kwargs['no_bias'] = True
        w = w.reshape(self.weight.shape)
        return nd.Convolution(x, w, name='fwd', **kwargs)

    def _weight_interpretable(self):
        return self.weight.data()

class Conv2DTransposeInterpretable(Interpretable, nn.Conv2DTranspose):
    def _forward_interpretable(self, x, w, bias=None):
        kwargs = self._kwargs.copy()
        kwargs['no_bias'] = True
        w = w.reshape(self.weight.shape)
        return nd.Deconvolution(x, w, name='fwd', **kwargs)

    def _weight_interpretable(self):
        return self.weight.data()

class BatchNormInterpretable(Interpretable, nn.BatchNorm):
    pass

class SequentialInterpretable(Interpretable, nn.Sequential):
    def relevance_layerwise(self, data=None, out=None, ret_all=False, **kwargs):
        if data is not None:
            self.forward_logged(data)
        elif self._in is None:
            raise RuntimeError('Block has not yet executed forward_logged!')
        method = kwargs.get('method', 'dtd')
        R = [self._out if out is None else out]
        for child in list(self._children.values())[::-1]:
            R.append(child.relevance(R[-1], **kwargs))
        return R if ret_all else R[-1]

    def relevance_sensitivity(self, data, out=None, **kwargs):
        data.attach_grad()
        with autograd.record():
            y = self.forward(data)
        y.backward(out_grad=out)
        return data.grad

    def relevance_intgrads(self, data, out=None, num=50, base=None, **kwargs):
        if base is None:
            base = nd.zeros_like(data)

        alpha = linspace(0., 1., num, ctx=data.context)
        diff = data - base

        res = nd.zeros_like(data)
        for a in alpha:
            res += self.relevance_sensitivity(data=base + a*diff, out=out)

        return res

    def relevance_dtd(self, *args, **kwargs):
        return self.relevance_layerwise(*args, **kwargs)

    def relevance_lrp(self, *args, **kwargs):
        return self.relevance_layerwise(*args, **kwargs)

    def forward_logged(self, x, depth=-1):
        self._in = [x]
        rdep = depth if depth > 0 else (len(self._children.values()) + depth)
        for i, block in enumerate(self._children.values()):
            x = block.forward_logged(x)
            if i == depth:
                break
        self._out = x
        return self._out

class YSequentialInterpretable(Interpretable, YSequentialBase):
    def forward_logged(self, x, y, depth=-1):
        self._in = [x, y]
        data = self._data_net.forward_logged(x)
        cond = self._cond_net.forward_logged(y)
        combo = nd.concat(data, cond, dim=self._concat_dim)
        self._out = self._main_net.forward_logged(combo, depth=depth)
        return self._out

    def relevance_layerwise(self, y=None, method='dtd', ret_all=False, **kwargs):
        if self._in is None:
            raise RuntimeError('Block has not yet executed forward_logged!')
        Rout = self._out if y is None else y
        R = self._main_net.relevance(Rout, method=method, ret_all=True, **kwargs)

        dim_d = self._data_net._out.shape[self._concat_dim]

        Rtd = R[-1].slice_axis(axis=self._concat_dim, begin=0, end=dim_d)
        Rtc = R[-1].slice_axis(axis=self._concat_dim, begin=dim_d, end=None)

        Rd = self._data_net.relevance(Rtd, method=method, ret_all=True, **kwargs)
        Rc = self._cond_net.relevance(Rtc, method=method, ret_all=True, **kwargs)

        R += Rd

        return (R, Rc) if ret_all else (R[-1], Rc[-1])

    def relevance_sensitivity(self, *args, **kwargs):
        for a in args:
            a.attach_grad()
        with autograd.record():
            y = self.forward(*args)
        y.backward()
        return [a.grad for a in args]

    def relevance_dtd(self, *args, **kwargs):
        return self.relevance_layerwise(*args, method='dtd', **kwargs)

    def relevance_lrp(self, *args, **kwargs):
        return self.relevance_layerwise(*args, method='lrp', **kwargs)


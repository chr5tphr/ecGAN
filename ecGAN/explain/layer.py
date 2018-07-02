import mxnet as mx
from mxnet import nd, gluon, autograd
from mxnet.gluon import nn
import numpy as np

from .base import Interpretable
from ..base import YSequentialBase, SequentialBase, ConcatBase, ParallelBase
from ..func import linspace, Mlist

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

class SequentialInterpretable(Interpretable, SequentialBase):
    def relevance_layerwise(self, data=None, out=None, **kwargs):
        if data is not None:
            self.forward_logged(data)
        elif self._in is None:
            raise RuntimeError('Block has not yet executed forward_logged!')

        R = self._out if out is None else out
        for child in list(self._children.values())[::-1]:
            R = child.relevance(R, **kwargs)

        return R

    def relevance_sensitivity(self, data, out=None, **kwargs):
        data = Mlist(data)
        data.attach_grad()
        with autograd.record():
            y = self.forward(data)
        y.backward(out_grad=out)
        return data.grad

    def relevance_intgrads(self, data, out=None, num=50, base=None, **kwargs):
        data = Mlist(data)
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

class ConcatInterpretable(Interpretable, ConcatBase):
    def relevance_layerwise(self, data=None, out=None, **kwargs):
        if data is not None:
            self.forward_logged(data)
        elif self._in is None:
            raise RuntimeError('Block has not yet executed forward_logged!')

        Rout = self._out if out is None else out
        dims = [0] + [x.shape[self._concat_dim] for x in Rout] + [None]
        R = [x.slice_axis(axis=self._concat_dim, begin=begin, end=end) for x, begin, end in zip(Rout, dims[:-1], dims[1:])]

        return R

    def relevance_dtd(self, *args, **kwargs):
        return self.relevance_layerwise(*args, **kwargs)

    def relevance_lrp(self, *args, **kwargs):
        return self.relevance_layerwise(*args, **kwargs)

class ParallelInterpretable(Interpretable, ParallelBase):
    def relevance_layerwise(self, data=None, cond=None, out=None, **kwargs):
        if data is not None:
            self.forward_logged(data)
        elif self._in is None:
            raise RuntimeError('Block has not yet executed forward_logged!')

        Rout = self._out if out is None else out
        R = []
        for child, rout in zip(self._children.values(), Rout):
            R.append(child.relevance(rout, **kwargs))

        return R

    def relevance_dtd(self, *args, **kwargs):
        return self.relevance_layerwise(*args, **kwargs)

    def relevance_lrp(self, *args, **kwargs):
        return self.relevance_layerwise(*args, **kwargs)

class YSequentialInterpretable(Interpretable, YSequentialBase):
    def relevance_layerwise(self, data=None, cond=None, out=None, **kwargs):
        if data is not None:
            self.forward_logged(data, cond)
        elif self._in is None:
            raise RuntimeError('Block has not yet executed forward_logged!')

        Rout = self._out if out is None else out
        R = self._main_net.relevance(Rout, **kwargs)

        dim_d = self._data_net._out.shape[self._concat_dim]

        Rtd = R[-1].slice_axis(axis=self._concat_dim, begin=0, end=dim_d)
        Rtc = R[-1].slice_axis(axis=self._concat_dim, begin=dim_d, end=None)

        Rd = self._data_net.relevance(Rtd, **kwargs)
        Rc = self._cond_net.relevance(Rtc, **kwargs)

        return Rd[-1], Rc[-1]

    def relevance_sensitivity(self, data, cond, out=None, **kwargs):
        data.attach_grad()
        cond.attach_grad()
        with autograd.record():
            y = self.forward(data, cond)
        y.backward(out_grad=out)
        return data.grad, cond.grad

    def relevance_intgrads(self, data, cond, out=None, num=50, base=None, cbase=None, **kwargs):
        if base is None:
            base = nd.zeros_like(data)
        if cbase is None:
            cbase = nd.zeros_like(cond)

        alpha = linspace(0., 1., num, ctx=data.context)
        diff = data - base
        cdiff = cond - cbase

        res = nd.zeros_like(data)
        cres = nd.zeros_like(cond)
        for a in alpha:
            sens, csens = self.relevance_sensitivity(data=base + a*diff, cond=cbase + a*cdiff, out=out)
            res += sens
            cres += csens

        return res, cres

    def relevance_dtd(self, *args, **kwargs):
        return self.relevance_layerwise(*args, **kwargs)

    def relevance_lrp(self, *args, **kwargs):
        return self.relevance_layerwise(*args, **kwargs)


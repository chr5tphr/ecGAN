import mxnet as mx
from mxnet import nd, gluon, autograd
from mxnet.gluon import nn
import numpy as np
from logging import getLogger

from ...func import Mlist
from ..base import Explainable
from ...base import Linear

class LayerwiseExplainable(Explainable):
    def __init__(self, *args, **kwargs):
        self._explain = kwargs.pop('explain', False)
        self._ekwargs = kwargs.pop('ekwargs', {})
        super().__init__(*args, **kwargs)

    def relevance_layerwise(self, *args, **kwargs):
        raise NotImplementedError

class LinearLayerwiseExplainable(LayerwiseExplainable, Linear):
    def relevance_layerwise(self, *args, **kwargs):
        method = self._explain
        func = getattr(self, 'layerwise_relevance_'+method)
        ekwargs = self._ekwargs.copy()
        ekwargs.update(kwargs)
        R = func(*args, **ekwargs)
        return R

    def check_bias(self, ctx=None):
        return self.bias is not None and (self.bias.data(ctx=ctx) <= 0.).asnumpy().all()

    def layerwise_relevance_zb(self, out, lo=-1, hi=1, use_bias=False, **kwargs):
        if self._in is None:
            raise RuntimeError('Block has not yet executed forward_logged!')
        R = out
        a = self._in[0]
        weight = self.weight.data(ctx=a.context)
        wplus = nd.maximum(0., weight)
        wminus = nd.minimum(0., weight)

        bias = None
        bplus = None
        bminus = None
        if use_bias is not None:
            bias = self.bias.data(ctx=a.context)
            bplus = nd.maximum(0., bias)
            bminus = nd.minimum(0., bias)

        upper = nd.ones_like(a)*hi
        lower = nd.ones_like(a)*lo
        a.attach_grad()
        upper.attach_grad()
        lower.attach_grad()
        with autograd.record():
            zlh = ( self._forward(a, weight, bias)
                  - self._forward(lower, wplus, bplus)
                  - self._forward(upper, wminus, bminus)
                  )
        zlh.backward(out_grad=R/(zlh + (zlh == 0.)))
        return a*a.grad + upper*upper.grad + lower*lower.grad

    def layerwise_relevance_zplus(self, out, use_bias=False, **kwargs):
        if self._in is None:
            raise RuntimeError('Block has not yet executed forward_logged!')
        R = out
        a = self._in[0]
        weight = self.weight.data(ctx=a.context)
        wplus = nd.maximum(0., weight)

        bplus = None
        if use_bias is not None:
            bias = self.bias.data(ctx=a.context)
            bplus = nd.maximum(0., bias)

        a.attach_grad()
        with autograd.record():
            z = self._forward(data=a, weight=wplus, bias=bplus)
        c, = autograd.grad(z, a, head_grads=R/(z + (z == 0.)))
        return a*c

    def layerwise_relevance_zclip(self, out, use_bias=False, **kwargs):
        if self._in is None:
            raise RuntimeError('Block has not yet executed forward_logged!')
        R = out
        a = self._in[0]
        z = self._out
        weight = self.weight.data(ctx=a.context)
        wplus = nd.maximum(0., weight)
        wminus = nd.minimum(0., weight)

        bplus = None
        bminus = None
        if use_bias is not None:
            bias = self.bias.data(ctx=a.context)
            bplus = nd.maximum(0., bias)
            bminus = nd.minimum(0., bias)

        alpha = z > 0.
        beta = z < 0.

        a.attach_grad()
        with autograd.record():
            zplus = self._forward(data=a, weight=wplus, bias=bplus)
        cplus, = autograd.grad(zplus, a, head_grads=alpha*R/(zplus + (zplus == 0.)))

        with autograd.record():
            zminus = self._forward(data=a, weight=wminus, bias=bminus)
        cminus, = autograd.grad(zminus, a, head_grads=beta*R/(zminus + (zminus == 0.)))

        return a*(cplus - cminus)

    def layerwise_relevance_wsquare(self, out, use_bias=False, **kwargs):
        if self._in is None:
            raise RuntimeError('Block has not yet executed forward_logged!')
        R = out
        a = self._in[0].ones_like()
        weight = self.weight.data(ctx=a.context)
        wsquare = weight**2
        bsquare = None

        if use_bias is not None:
            bias = self.bias.data(ctx=a.context)
            bsquare = bias**2

        a.attach_grad()
        with autograd.record():
            z = self._forward(data=a, weight=wsquare, bias=bsquare)
        c, = autograd.grad(z, a, head_grads=R/(z + (z == 0.)))
        return c

    def layerwise_relevance_alphabeta(self, out, alpha=1., beta=0., use_bias=False, **kwargs):
        if self._in is None:
            raise RuntimeError('Block has not yet executed forward_logged!')
        R = out
        a = self._in[0]
        weight = self.weight.data(ctx=a.context)
        wplus = nd.maximum(0., weight)
        wminus = nd.minimum(0., weight)

        bplus = None
        bminus = None
        if use_bias is not None:
            bias = self.bias.data(ctx=a.context)
            bplus = nd.maximum(0., bias)
            bminus = nd.minimum(0., bias)

        a.attach_grad()
        with autograd.record():
            zplus = self._forward(data=a, weight=wplus, bias=bplus)
        cplus, = autograd.grad(zplus, a, head_grads=alpha*R/(zplus + (zplus == 0.)))

        with autograd.record():
            zminus = self._forward(data=a, weight=wminus, bias=bminus)
        cminus, = autograd.grad(zminus, a, head_grads=beta*R/(zminus + (zminus == 0.)))

        return a*(cplus - cminus)

class PoolingLayerwiseExplainable(LayerwiseExplainable):
    def relevance_layerwise(self, out, *args, **kwargs):
        R = out
        a = self._in[0]
        pkwargs = self._kwargs.copy()
        pkwargs['pool_type'] = 'sum'
        # suppress mxnet warnings about sum-pooling nob being supported with cudnn
        pkwargs['cudnn_off'] = True
        a.attach_grad()
        with autograd.record():
            z = nd.Pooling(a, **pkwargs)
        z.backward(out_grad=R/(z + (z == 0.)))
        return a * a.grad

class ActLayerwiseExplainable(LayerwiseExplainable):
    def relevance_layerwise(self, out, *args, **kwargs):
        return out


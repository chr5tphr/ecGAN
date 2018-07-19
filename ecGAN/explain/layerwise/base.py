import mxnet as mx
from mxnet import nd, gluon, autograd
from mxnet.gluon import nn
import numpy as np
from logging import getLogger

from ...base import Block
from ...func import Mlist
from ..base import Explainable

class LayerwiseExplainable(Explainable):
    def __init__(self, *args, **kwargs):
        self._explain = kwargs.pop('explain', False)
        super().__init__(*args, **kwargs)

    def relevance_layerwise(self, *args, **kwargs):
        raise NotImplementedError

class LinearLayerwiseExplainable(LayerwiseExplainable):
    def relevance_layerwise(self, *args, **kwargs):
        method = self._explain
        func = getattr(self, 'layerwise_relevance_'+method)
        R = func(*args, **kwargs)
        return R

    def layerwise_relevance_zb(self, out, lo=-1, hi=1, **kwargs):
        if self._in is None:
            raise RuntimeError('Block has not yet executed forward_logged!')
        R = out
        a = self._in[0]
        weight = self._weight()
        wplus = nd.maximum(0., weight)
        wminus = nd.minimum(0., weight)
        upper = nd.ones_like(a)*hi
        lower = nd.ones_like(a)*lo
        a.attach_grad()
        upper.attach_grad()
        lower.attach_grad()
        with autograd.record():
            zlh = ( self._forward(a, weight)
                  - self._forward(lower, wplus)
                  - self._forward(upper, wminus)
                  )
        zlh.backward(out_grad=R/(zlh + (zlh == 0.)))
        return a*a.grad + upper*upper.grad + lower*lower.grad

    def layerwise_relevance_zplus(self, out, **kwargs):
        if self._in is None:
            raise RuntimeError('Block has not yet executed forward_logged!')
        R = out
        a = self._in[0]
        weight = self._weight()
        wplus = nd.maximum(0., weight)
        a.attach_grad()
        with autograd.record():
            z = self._forward(data=a, weight=wplus)
        c, = autograd.grad(z, a, head_grads=R/(z + (z == 0.)))
        return a*c

    def layerwise_relevance_zclip(self, out, **kwargs):
        if self._in is None:
            raise RuntimeError('Block has not yet executed forward_logged!')
        R = out
        a = self._in[0]
        z = self._out
        weight = self._weight()
        wplus = nd.maximum(0., weight)
        wminus = nd.minimum(0., weight)
        alpha = z > 0.
        beta = z < 0.

        a.attach_grad()
        with autograd.record():
            zplus = self._forward(data=a, weight=wplus)
        cplus, = autograd.grad(zplus, a, head_grads=alpha*R/(zplus + (zplus == 0.)))

        with autograd.record():
            zminus = self._forward(data=a, weight=wminus)
        cminus, = autograd.grad(zminus, a, head_grads=beta*R/(zminus + (zminus == 0.)))

        return a*(cplus - cminus)

    def layerwise_relevance_wsquare(self, out, **kwargs):
        if self._in is None:
            raise RuntimeError('Block has not yet executed forward_logged!')
        R = out
        a = self._in[0].ones_like()
        weight = self._weight()
        wsquare = weight**2
        a.attach_grad()
        with autograd.record():
            z = self._forward(data=a, weight=wsquare)
        c, = autograd.grad(z, a, head_grads=R/(z + (z == 0.)))
        return c

    def layerwise_relevance_alphabeta(self, out, alpha=1., beta=0., **kwargs):
        if self._in is None:
            raise RuntimeError('Block has not yet executed forward_logged!')
        R = out
        a = self._in[0]
        weight = self._weight()
        wplus = nd.maximum(0., weight)
        wminus = nd.minimum(0., weight)

        a.attach_grad()
        with autograd.record():
            zplus = self._forward(data=a, weight=wplus)
        cplus, = autograd.grad(zplus, a, head_grads=alpha*R/(zplus + (zplus == 0.)))

        with autograd.record():
            zminus = self._forward(data=a, weight=wminus)
        cminus, = autograd.grad(zminus, a, head_grads=beta*R/(zminus + (zminus == 0.)))

        return a*(cplus - cminus)

class ActLayerwiseExplainable(LayerwiseExplainable):
    def relevance_layerwise(self, out, *args, **kwargs):
        return out


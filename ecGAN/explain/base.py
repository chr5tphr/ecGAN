import mxnet as mx
from mxnet import nd, gluon, autograd
from mxnet.gluon import nn
import numpy as np

from ..base import Block

class Interpretable(Block):
    def __init__(self, *args, **kwargs):
        self._isinput = kwargs.pop('isinput', False)
        super().__init__(*args, **kwargs)
        self._in = None
        self._out = None

    def relevance(self, *args, **kwargs):
        method = kwargs.pop('method', 'dtd')
        func = getattr(self, 'relevance_'+method)
        return func(*args, **kwargs)

    def relevance_sensitivity(self, R):
        if self._in is None:
            raise RuntimeError('Block has not yet executed forward_logged!')
        a = self._in[0]
        a.attach_grad()
        with autograd.record():
            z = self(a)
        return autograd.grad(z, a, head_grads=R)

    def relevance_dtd(self, R, lo=-1, hi=1):
        if self._in is None:
            raise RuntimeError('Block has not yet executed forward_logged!')
        a = self._in[0]
        if self._isinput: #zb
            weight = self._weight_interpretable()
            wplus = nd.maximum(0., weight)
            wminus = nd.minimum(0., weight)
            upper = nd.ones_like(a)*hi
            lower = nd.ones_like(a)*lo
            a.attach_grad()
            upper.attach_grad()
            lower.attach_grad()
            with autograd.record():
                zlh = ( self._forward_interpretable(a, weight)
                      - self._forward_interpretable(lower, wplus)
                      - self._forward_interpretable(upper, wminus)
                      )
            zlh.backward(out_grad=R/zlh)
            return a*a.grad + upper*upper.grad + lower*lower.grad
        else: #z+
            weight = self._weight_interpretable()
            wplus = nd.maximum(0., weight)
            a.attach_grad()
            with autograd.record():
                z = self._forward_interpretable(a, wplus)
            c = autograd.grad(z, a, head_grads=R/z)
            return a*c

    def relevance_lrp(self, R, alpha=1., beta=0.):
        if self._in is None:
            raise RuntimeError('Block has not yet executed forward_logged!')
        a = self._in[0]
        weight = self._weight_interpretable()
        wplus = nd.maximum(0., weight)
        wminus = nd.minimum(0., weight)
        with autograd.record():
            zplus = self._forward_interpretable(a, wplus)
            zminus = self._forward_interpretable(a, wminus)
        cplus = autograd.grad(zplus, a, head_grads=alpha*R/zplus)
        cminus = autograd.grad(zminus, a, head_grads=beta*R/zminus)
        return a*(cplus - cminus)

    def _forward_interpretable(self, *args, **kwargs):
        raise NotImplementedError

    def _weight_interpretable(self):
        raise NotImplementedError

class ActInterpretable(Interpretable):
    def relevance(self, *args, **kwargs):
        return args[0]


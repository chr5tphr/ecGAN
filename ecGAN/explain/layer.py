import mxnet as mx
from mxnet import nd, gluon, autograd
from mxnet.gluon import nn
import numpy as np

from .base import Interpretable
from ..layer import YSequentialBase
from ..func import linspace

class SequentialInterpretable(Interpretable, nn.Sequential):
    def relevance_layerwise(self, y=None, method='dtd', ret_all=False, **kwargs):
        if self._in is None:
            raise RuntimeError('Block has not yet executed forward_logged!')
        R = [self._out if y is None else y]
        for child in self._children[::-1]:
            R.append(child.relevance(R[-1], method=method))
        return R if ret_all else R[-1]

    def relevance_sensitivity(self, *args, **kwargs):
        a = args[0]
        a.attach_grad()
        with autograd.record():
            y = self.forward(a)
        y.backward()
        return a.grad

    def relevance_intgrads(self, x, *args, num=50, base=None, **kwargs):
        if base is None:
            base = nd.zeros_like(x)

        alpha = linspace(0., 1., num)
        diff = x - base

        res = nd.zeros_like(x)
        for a in alpha:
            res += self.relevance_sensitivity(base + a*diff)

        return res

    def relevance_dtd(self, *args, **kwargs):
        return self.relevance_layerwise(*args, method='dtd', **kwargs)

    def relevance_lrp(self, *args, **kwargs):
        return self.relevance_layerwise(*args, method='lrp', **kwargs)

    def forward_logged(self, x, depth=-1):
        self._in = [x]
        rdep = depth if depth > 0 else (len(self._children) + depth)
        for i, block in enumerate(self._children):
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

    def relevance_layerwise(self, y=None, method='dtd', ret_all=False):
        if self._in is None:
            raise RuntimeError('Block has not yet executed forward_logged!')
        Rout = self._out if y is None else y
        R = self._main_net.relevance(Rout, method=method, ret_all=True)

        dim_d = self._data_net._out.shape[self._concat_dim]

        Rtd = R[-1].slice_axis(axis=self._concat_dim, begin=0, end=dim_d)
        Rtc = R[-1].slice_axis(axis=self._concat_dim, begin=dim_d, end=None)

        Rd = self._data_net.relevance(Rtd, method=method, ret_all=True)
        Rc = self._cond_net.relevance(Rtc, method=method, ret_all=True)

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


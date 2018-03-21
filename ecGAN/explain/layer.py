import mxnet as mx
from mxnet import nd, gluon, autograd
from mxnet.gluon import nn
import numpy as np

from .base import Interpretable
from ..base import YSequentialBase
from ..func import linspace

class DenseInterpretable(Interpretable, nn.Dense):
    def relevance_sensitivity(self, R):
        if self._in is None:
            raise RuntimeError('Block has not yet executed forward_logged!')
        a = self._in[0]
        a.attach_grad()
        with autograd.record():
            z = self(a)
        return autograd.grad(z, a, head_grads=R)

    def relevance_lrp(self, R, alpha=1., beta=0.):
        if self._in is None:
            raise RuntimeError('Block has not yet executed forward_logged!')
        a = self._in[0]
        wplus = nd.maximum(0., self.weight.data())
        wminus = nd.minimum(0., self.weight.data())
        with autograd.record():
            zplus = nd.FullyConnected(a, wplus, None, no_bias=True, num_hidden=self._units, flatten=self._flatten)
            zminus = nd.FullyConnected(a, wminus, None, no_bias=True, num_hidden=self._units, flatten=self._flatten)
        cplus = autograd.grad(zplus, a, head_grads=alpha*R/zplus)
        cminus = autograd.grad(zminus, a, head_grads=beta*R/zminus)
        return a*(cplus - cminus)

    def relevance_dtd(self, R, lo=-1, hi=1):
        if self._in is None:
            raise RuntimeError('Block has not yet executed forward_logged!')
        a = self._in[0]
        if self._isinput: #zb
            wplus = nd.maximum(0., self.weight.data())
            wminus = nd.minimum(0., self.weight.data())
            upper = nd.ones_like(a)*hi
            lower = nd.ones_like(a)*lo
            a.attach_grad()
            upper.attach_grad()
            lower.attach_grad()
            with autograd.record():
                zlh = (  self(a)
                       - nd.FullyConnected(lower,
                                           wplus,
                                           None,
                                           no_bias=True,
                                           num_hidden=self._units,
                                           flatten=self._flatten)
                       - nd.FullyConnected(upper,
                                           wminus,
                                           None,
                                           no_bias=True,
                                           num_hidden=self._units,
                                           flatten=self._flatten) )
            zlh.backward(out_grad=R/zlh)
            return a*a.grad + upper*upper.grad + lower*lower.grad
        else: #z+
            wplus = nd.maximum(0., self.weight.data())
            a.attach_grad()
            with autograd.record():
                z = nd.FullyConnected(a,
                                      wplus,
                                      None,
                                      no_bias=True,
                                      num_hidden=self._units,
                                      flatten=self._flatten)
                if self.act is not None:
                    z = self.act(z)
            c = autograd.grad(z, a, head_grads=R/z)
            return a*c

class Conv2DInterpretable(Interpretable, nn.Conv2D):
    pass

class Conv2DTransposeInterpretable(Interpretable, nn.Conv2DTranspose):
    pass

class BatchNormInterpretable(Interpretable, nn.BatchNorm):
    pass

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


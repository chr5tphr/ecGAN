import mxnet as mx
from mxnet import nd, gluon, autograd
from mxnet.gluon import nn

from .func import Interpretable, PatternNet

class DenseAdapter(nn.Dense):
    def __init__(self, *args, **kwargs):
        super(nn.Dense, self).__init__(*args, **kwargs)
        super().__init__(*args, **kwargs)

class DensePatternNet(PatternNet, DenseAdapter):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        with self.name_scope():
            self.mean_x = self.params.get('mean_x',
                                          shape=(1,in_units),
                                          init=mx.initializer.Zero(),
                                          grad_req='null')
            self.mean_y = self.params.get('mean_y',
                                          shape=(1,units),
                                          init=mx.initializer.Zero(),
                                          grad_req='null')
            self.var_y = self.params.get('var_y',
                                         shape=(1,units),
                                         init=mx.initializer.Zero(),
                                         grad_req='null')
            self.cov = self.params.get('cov',
                                       shape=self.weight.shape,
                                       init=mx.initializer.Zero(),
                                       grad_req='null')
            self.num_samples = self.params.get('num_samples',
                                               shape=(1,),
                                               init=mx.initializer.Zero(),
                                               grad_req='null')

    def learn_pattern_linear(self):
        if self._in is None:
            raise RuntimeError('Block has not yet executed forward_logged!')
        x = self._in
        y = self._out
        meanx = self.mean_x
        meany = self.mean_y
        n = self.num_samples
        m = x.shape[0]

        meanx_ = x.mean(axis=0, keepdims=True)
        meany_ = y.mean(axis=0, keepdims=True)
        dx = x - meanx_
        dy = y - meany_

        C_ = nd.dot(dx, dy, transpose_a=True)
        self._cov += C_ + nd.dot((meanx - meanx_), (meany - meany_), transpose_a=True) * n * m / (n+m)

        vary_ = nd.sum(dy**2, axis=0)
        self._vary += vary_ + ((meany - meany_) * (meany - meany_)) * n * m / (n+m)

        self.mean_x = (n * meanx + m * meanx_) / (n+m)
        self.mean_y = (n * meany + m * meany_) / (n+m)
        self.num_samples += m

    def forward_pattern_linear(self, *args):
        x = args[0]
        # omit number of samples, since present in both cov and var
        a = self.cov / self.var_y
        z = nd.FullyConnected(x, a, None, no_bias=True, num_hidden=self._units, flatten=self._flatten)
        if self.act is not None:
            z = self.act(z)
        return z

    def assess_pattern_linear(self):
        pass

class DenseInterpretable(Interpretable, DenseAdapter):
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

class Dense(DenseInterpretable, DensePatternNet):
    pass

class Identity(Interpretable, nn.Block):
    def forward(self, *args, **kwargs):
        return args[0]
    def relevance(self, *args, **kwargs):
        return args[0]

class Clip(Interpretable, nn.Block):
    def forward(self, x):
        return nd.clip(x, 0., 1.)
    def relevance(self, R):
        return R

class Conv2D(Interpretable, nn.Conv2D):
    pass

class Conv2DTranspose(Interpretable, nn.Conv2DTranspose):
    pass

class BatchNorm(Interpretable, nn.BatchNorm):
    pass

class LeakyReLU(Interpretable, nn.BatchNorm):
    pass

class Activation(Interpretable, nn.Activation):
    pass

class MaxOut(nn.Block):
    pass

import mxnet as mx
from mxnet import nd, gluon, autograd
from mxnet.gluon import nn

from .func import Interpretable, PatternNet

#class DenseAdapter(nn.Dense):
#    def __init__(self, *args, **kwargs):
#        super(nn.Dense, self).__init__(*args, **kwargs)
#        super().__init__(*args, **kwargs)

class DensePatternNet(PatternNet, nn.Dense):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    ##############
    ### Linear ###
    ##############

    def init_pattern_linear(self):
        units, in_units = self.weight.shape
        with self.name_scope():
            self.mean_x = self.pparams.get('mean_x',
                                          shape=(1, in_units),
                                          init=mx.initializer.Zero(),
                                          grad_req='null')
            self.mean_y = self.pparams.get('mean_y',
                                          shape=(1, units),
                                          init=mx.initializer.Zero(),
                                          grad_req='null')
            self.var_y = self.pparams.get('var_y',
                                         shape=(1, units),
                                         init=mx.initializer.Zero(),
                                         grad_req='null')
            self.cov = self.pparams.get('cov',
                                       shape=self.weight.shape,
                                       init=mx.initializer.Zero(),
                                       grad_req='null')
            self.num_samples = self.pparams.get('num_samples',
                                               shape=(1,),
                                               init=mx.initializer.Zero(),
                                                grad_req='null')

    def learn_pattern_linear(self):
        if self._in is None:
            raise RuntimeError('Block has not yet executed forward_logged!')
        x = self._in[0].flatten()
        y = self._out.flatten()
        mean_x = self.mean_x.data()
        mean_y = self.mean_y.data()
        n = self.num_samples.data()
        var_y = self.var_y.data()
        cov = self.cov.data()
        m = x.shape[0]

        mean_x_cur = x.mean(axis=0, keepdims=True)
        mean_y_cur = y.mean(axis=0, keepdims=True)
        dx = x - mean_x_cur
        dy = y - mean_y_cur

        cov_cur = nd.dot(dy, dx, transpose_a=True)
        cov += cov_cur + nd.dot((mean_y - mean_y_cur), (mean_x - mean_x_cur), transpose_a=True) * n * m / (n+m)

        var_y_cur = nd.sum(dy**2, axis=0)
        var_y += var_y_cur + ((mean_y - mean_y_cur) * (mean_y - mean_y_cur)) * n * m / (n+m)

        mean_x = (n * mean_x + m * mean_x_cur) / (n+m)
        mean_y = (n * mean_y + m * mean_y_cur) / (n+m)
        n += m

        self.mean_x.set_data(mean_x)
        self.mean_y.set_data(mean_y)
        self.num_samples.set_data(n)
        self.var_y.set_data(var_y)
        self.cov.set_data(cov)

    def forward_pattern_linear(self, *args):
        x = args[0]
        if len(args) < 2:
            x_neut = x
        else:
            x_neut = args[1]
        # omit number of samples, since present in both cov and var
        a = self.cov.data() / (self.var_y.data().T + 1e-12)
        z_neut = self.forward(x_neut)
        z = nd.FullyConnected(x, a, None, no_bias=True, num_hidden=self._units, flatten=self._flatten)

        # this depends on ReLU!
        z = nd.where(z_neut>0, z, nd.zeros_like(z))
        return z, z_neut

    def assess_pattern_linear(self):
        pass

    #####################
    ### Two-Component ###
    #####################

    def init_pattern_twocomponent(self):
        units, in_units = self.weight.shape
        with self.name_scope():
            self.mean_x_pos = self.pparams.get('mean_x_pos',
                                          shape=(1, in_units),
                                          init=mx.initializer.Zero(),
                                          grad_req='null')
            self.mean_x_neg = self.pparams.get('mean_x_neg',
                                          shape=(1, in_units),
                                          init=mx.initializer.Zero(),
                                          grad_req='null')
            self.mean_y = self.pparams.get('mean_y',
                                          shape=(1, units),
                                          init=mx.initializer.Zero(),
                                          grad_req='null')
            self.cov_pos = self.pparams.get('cov_pos',
                                       shape=self.weight.shape,
                                       init=mx.initializer.Zero(),
                                       grad_req='null')
            self.cov_neg = self.pparams.get('cov_neg',
                                       shape=self.weight.shape,
                                       init=mx.initializer.Zero(),
                                       grad_req='null')
            self.num_samples = self.pparams.get('num_samples',
                                               shape=(1,),
                                               init=mx.initializer.Zero(),
                                               grad_req='null')

    def learn_pattern_twocomponent(self):
        if self._in is None:
            raise RuntimeError('Block has not yet executed forward_logged!')
        x = self._in[0].flatten()
        y = self._out.flatten()
        mean_x_pos = self.mean_x_pos.data()
        mean_x_neg = self.mean_x_neg.data()
        mean_y = self.mean_y.data()
        n = self.num_samples.data()
        cov_pos = self.cov_pos.data()
        cov_neg = self.cov_neg.data()
        m = x.shape[0]

        x_pos = nd.maximum(0., x)
        x_neg = nd.minimum(0., x)

        mean_x_pos_cur = x_pos.mean(axis=0, keepdims=True)
        mean_x_neg_cur = x_neg.mean(axis=0, keepdims=True)
        mean_y_cur = y.mean(axis=0, keepdims=True)

        dx_pos = x_pos - mean_x_pos_cur
        dx_neg = x_neg - mean_x_neg_cur
        dy = y - mean_y_cur

        cov_pos_cur = nd.dot(dy, dx_pos, transpose_a=True)
        cov_neg_cur = nd.dot(dy, dx_neg, transpose_a=True)
        cov_pos += cov_pos_cur + nd.dot((mean_y - mean_y_cur), (mean_x_pos - mean_x_pos_cur), transpose_a=True) * n * m / (n+m)
        cov_neg += cov_neg_cur + nd.dot((mean_y - mean_y_cur), (mean_x_neg - mean_x_neg_cur), transpose_a=True) * n * m / (n+m)

        mean_x_pos = (n * mean_x_pos + m * mean_x_pos_cur) / (n+m)
        mean_x_neg = (n * mean_x_neg + m * mean_x_neg_cur) / (n+m)
        mean_y = (n * mean_y + m * mean_y_cur) / (n+m)
        n += m

        self.mean_x_pos.set_data(mean_x_pos)
        self.mean_x_neg.set_data(mean_x_neg)
        self.mean_y.set_data(mean_y)
        self.num_samples.set_data(n)
        self.cov_pos.set_data(cov_pos)
        self.cov_neg.set_data(cov_neg)

    def forward_pattern_twocomponent(self, *args):
        x = args[0]
        if len(args) < 2:
            x_neut = x
        else:
            x_neut = args[1]
        cov_pos = self.cov_pos.data()
        cov_neg = self.cov_neg.data()
        weight = self.weight.data()
        a_pos = cov_pos / ((weight * cov_pos).sum(axis=1, keepdims=True) + 1e-12)
        a_neg = cov_neg / ((weight * cov_neg).sum(axis=1, keepdims=True) + 1e-12)

        z_neut = self.forward(x_neut)
        z_pos = nd.FullyConnected(x, a_pos, None, no_bias=True, num_hidden=self._units, flatten=self._flatten)
        z_neg = nd.FullyConnected(x, a_neg, None, no_bias=True, num_hidden=self._units, flatten=self._flatten)

        # this depends on ReLU!
        z = nd.where(z_neut>0, z_pos, z_neg)

        return z, z_neut

    def assess_pattern_twocomponent(self):
        pass


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

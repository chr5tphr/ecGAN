import re
import mxnet as mx
from mxnet import nd, autograd
from mxnet.gluon import nn, ParameterDict

from ..base import Block
from ..func import stats_batchwise

class PatternNet(Block):
    def __init__(self, *args, **kwargs):
        #self.estimator = kwargs.pop('estimator', 'linear')
        self._regimes = kwargs.pop('regimes', [PatternRegime('linear', lambda y: nd.ones_like(y))])
        #self._regimes = kwargs.pop('regimes', [PatternRegime('pos', lambda y: y>0)])
        super().__init__(*args, **kwargs)
        self._pparams = ParameterDict(getattr(self, '_prefix', ''))
        self.num_samples = None
        self.mean_y = None
        self._err = None
        self.w_qual = None

    def hybrid_forward(self, F, x, weight, bias=None, **kwargs):
        # exists since there is a bug with the way mxnet uses params and our pparams
        return super().hybrid_forward(F, x, weight, bias)

    @property
    def pparams(self):
        return self._pparams

    def collect_pparams(self, select=None):
        self._check_container_with_block()
        ret = ParameterDict(self.pparams.prefix)
        if not select:
            ret.update(self.pparams)
        else:
            pattern = re.compile(select)
            ret.update({name:value for name, value in self.pparams.items() if pattern.match(name)})
        for cld in self._children.values():
            try:
                ret.update(cld.collect_pparams(select=select))
            except AttributeError:
                pass
        return ret


    def init_pattern(self):
        outsize, insize = self._shape_pattern()
        with self.name_scope():
            self.num_samples = self.pparams.get('num_samples',
                                                shape=(1,),
                                                init=mx.initializer.Zero(),
                                                grad_req='null')
            self.mean_y = self.pparams.get('mean_y',
                                           shape=(1, outsize),
                                           init=mx.initializer.Zero(),
                                           grad_req='null')
            self.var_y = self.pparams.get('var_y',
                                          shape=(1, outsize),
                                          init=mx.initializer.Zero(),
                                          grad_req='null')
            self.mean_d = self.pparams.get('mean_d',
                                          shape=(1, insize),
                                          init=mx.initializer.Zero(),
                                          grad_req='null')
            self.cov_dd = self.pparams.get('cov_dd',
                                          shape=(insize, insize),
                                          init=mx.initializer.Zero(),
                                          grad_req='null')
            self.cov_dy = self.pparams.get('cov_dy',
                                           shape=(outsize, insize),
                                           init=mx.initializer.Zero(),
                                           grad_req='null')
            self.w_qual = self.pparams.get('w_qual',
                                           shape=(outsize, insize),
                                           init=mx.initializer.Xavier(),
                                           grad_req='write')
            for regime in self._regimes:
                regime.mean_x = self.pparams.get('mean_x_%s'%str(regime),
                                                 shape=(1, insize),
                                                 init=mx.initializer.Zero(),
                                                 grad_req='null')
                regime.mean_xy = self.pparams.get('mean_xy_%s'%str(regime),
                                                  shape=(outsize, insize),
                                                  init=mx.initializer.Zero(),
                                                  grad_req='null')
                regime.num_y = self.pparams.get('num_y_%s'%str(regime),
                                                shape=(1, outsize),
                                                init=mx.initializer.Zero(),
                                                grad_req='null')
                regime.pattern = self.pparams.get('pattern_%s'%str(regime),
                                                  shape=(outsize, insize),
                                                  #init=mx.initializer.Constant(1.),
                                                  init=mx.initializer.Xavier(),
                                                  grad_req='write')
                regime.pias = self.pparams.get('pias_%s'%str(regime),
                                               shape=(insize,),
                                               init=mx.initializer.Zero(),
                                               grad_req='write')

    def forward_pattern(self, x):
        z = None
        for regime in self._regimes:
            regime.pattern_ref = self._weight_pattern().copy()
            a_reg = regime.pattern_ref
            z_reg = self._forward_pattern(x, a_reg)
            # regimes are assumed to be disjunct
            if z is None:
                z = nd.zeros_like(z_reg)
            z = nd.where(regime(z_reg), z_reg, z)
        return z

    def overload_weight_pattern(self):
        for regime in self._regimes:
            regime.pattern_ref[:] = regime.pattern.data(ctx=regime.pattern_ref.context)

    def overload_weight_attribution_pattern(self):
        for regime in self._regimes:
            regime.pattern_ref *= regime.pattern.data(ctx=regime.pattern_ref.context)

    def backward_pattern(self, y_sig):
        if self._out is None:
            raise RuntimeError('Block has not yet executed forward_logged!')
        y_cond = self._out
        return self._signal_pattern(y_sig, y_cond)

    def compute_pattern(self):
        weight = self._weight_pattern()
        for regime in self._regimes:
            cov = regime.mean_xy.data() - nd.dot(self.mean_y.data(), regime.mean_x.data(), transpose_a=True)
            var_y = (weight * cov).sum(axis=1, keepdims=True) + 1e-12
            pat = cov / var_y
            regime.pattern.set_data(pat)

    def fit_pattern(self, x):
        y = self(x)
        #computes only gradients for batch step!
        self._err = []
        for regime in self._regimes:
            loss = mx.gluon.loss.L2Loss()
            with autograd.record():
                y_reg = y * regime(y)
                pattern = regime.pattern.data(ctx=y.context)
                pias = regime.pias.data(ctx=y.context)
                signal = self._backward_pattern(y_reg, pattern, pias)
                err = loss(signal, x)
            err.backward()
            self._err.append(err)
        return y
        #y = self(x)
        ##computes only gradients for batch step!
        #loss = mx.gluon.loss.L2Loss()
        #with autograd.record():
        #    signal = self._signal_pattern(y)
        #    err = loss(signal, x)
        #err.backward()
        #self._err = err
        #return y

    def fit_assess_pattern(self, x):
        y = self(x)
        #computes only gradients for batch step!
        w_qual = self.w_qual.data(ctx=x.context)
        signal = self._signal_pattern(y)
        distractor = x - signal.reshape(x.shape)
        loss = mx.gluon.loss.L2Loss()
        with autograd.record():
            vtd = self._forward_pattern(distractor, w_qual)
            err = loss(vtd, y)
        err.backward()
        self._err = err
        return y

    def _prepare_data_pattern(self):
        raise NotImplementedError

    def stats_assess_pattern(self):
        for x, y in self._prepare_data_pattern():
            s = self._signal_pattern(y)
            d = x - s

            attrs = ['num_samples', 'mean_d', 'mean_y', None, 'var_y', 'cov_dd', None, 'cov_dy']

            args = [d, y] + [None if name is None else getattr(self, name).data(ctx=d.context) for name in attrs]
            retval = stats_batchwise(*args)
            for name, val in zip(attrs, retval):
                if name is not None:
                    getattr(self, name).set_data(val)

    def _signal_pattern(self, y, y_cond=None):
        if y_cond is None:
            y_cond = y
        # shape is set in first regime
        signal = 0.
        for regime in self._regimes:
            y_reg = y * regime(y_cond)
            pattern = regime.pattern.data(ctx=y.context)
            pias = regime.pias.data(ctx=y.context)
            s_reg = self._backward_pattern(y_reg, pattern, pias)
            # regimes are assumed to be disjunct
            signal = signal + s_reg
        return signal

    def explain_pattern(self):
        raise NotImplementedError

    def explain_attribution_pattern(self):
        raise NotImplementedError

    def assess_pattern(self):
        v = self.w_qual.data()
        dd_cov = self.cov_dd.data(ctx=v.context)
        dy_cov = self.cov_dy.data(ctx=v.context)
        y_var = self.var_y.data(ctx=v.context)
        vd_var = (v * nd.dot(v, dd_cov)).sum(axis=1)
        corr = (dy_cov * v).sum(axis=1) / (vd_var * y_var)**0.5
        return 1. - corr

    def learn_pattern(self):
        for x, y in self._prepare_data_pattern():
            for regime in self._regimes:
                mean_x = regime.mean_x.data()
                mean_xy = regime.mean_xy.data()
                num_y = regime.num_y.data()
                num_x = num_y.sum()

                cond_y = regime(y)
                # number of times each sample's x for w.t dot x was inside the regime
                num_n = cond_y.sum(axis=1, keepdims=True)
                # => weighted sum over x
                wsum_x = nd.dot(num_n, x, transpose_a=True)

                # y's in regime
                reg_y = y * cond_y
                # sum of xy's in regime
                # sum_xy = (reg_y.expand_dims(axis=2) * x.expand_dims(axis=1)).sum(axis=0)
                sum_xy = nd.dot(reg_y, x, transpose_a=True)


                #TODO more stable running mean
                num_x_cur = num_n.sum()
                mean_x = (num_x * mean_x + wsum_x) / (num_x + num_x_cur + 1e-12)

                num_y_cur = cond_y.sum(axis=0)
                mean_xy = (num_y.T * mean_xy + sum_xy) / (num_y + num_y_cur + 1e-12).T

                num_y += num_y_cur

                regime.num_y.set_data(num_y)
                regime.mean_x.set_data(mean_x)
                regime.mean_xy.set_data(mean_xy)

            num = self.num_samples.data()
            num_cur = x.shape[0]
            mean_y = self.mean_y.data()
            sum_y = y.sum(axis=0, keepdims=True)
            mean_y = (num * mean_y + sum_y) / (num + num_cur + 1e-12)
            num += num_cur
            self.mean_y.set_data(mean_y)
            self.num_samples.set_data(num)

    def _backward_pattern(self, y, pattern, pias=None):
        raise NotImplementedError

    def _shape_pattern(self):
        raise NotImplementedError

    def _weight_pattern(self):
        raise NotImplementedError

    @staticmethod
    def _args_forward_pattern(*args):
        if len(args) == 1:
            return args[0], args[0]
        elif len(args) == 2:
            return args
        else:
            raise RuntimeError('Number of input arguments not correct!')

    def _forward_pattern(self, x, pattern, pias=None):
        raise NotImplementedError

class ActPatternNet(PatternNet):
    def init_pattern(self, *args):
        pass

    def forward_pattern(self, *args):
        return self.forward(*args)

    def learn_pattern(self, *args):
        pass

    def fit_pattern(self, x):
        return self(x)

    def compute_pattern(self):
        pass

    def fit_assess_pattern(self, x):
        return self(x)

    def stats_assess_pattern(self):
        pass

    def assess_pattern(self):
        return None

    def overload_weight_pattern(self):
        pass

    def overload_weight_attribution_pattern(self):
        pass

    def backward_pattern(self, y_sig):
        raise NotImplementedError

class PatternRegime(object):
    def __init__(self, name, condition):
        self.name = name
        if condition is not None:
            self.condition = condition
        self.mean_x = None
        self.mean_xy = None
        self.num_y = None
        self.pattern = None
        self.pattern_ref = None

    def __str__(self):
        return self.name

    def __repr__(self):
        return '%s: %s'%(str(self.__class__), str(self))

    def __call__(self, *args):
        return self.condition(*args)

    def copy(self):
        return PatternRegime(self.name, self.condition)

    def condition(self, *args):
        raise NotImplementedError


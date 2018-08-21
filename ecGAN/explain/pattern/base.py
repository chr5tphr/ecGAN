import re
import mxnet as mx
from mxnet import nd, autograd
from mxnet.gluon import nn, ParameterDict

from ...base import Block
from ...func import stats_batchwise

class PatternNet(Block):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._pparams = ParameterDict(getattr(self, '_prefix', ''))
        self._err = None

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

    def init_pattern(self, *args):
        raise NotImplementedError

    def forward_pattern(self, *args):
        raise NotImplementedError

    def learn_pattern(self, *args):
        raise NotImplementedError

    def fit_pattern(self, x):
        raise NotImplementedError

    def compute_pattern(self, ctx=None):
        raise NotImplementedError

    def fit_assess_pattern(self, x):
        raise NotImplementedError

    def stats_assess_pattern(self):
        raise NotImplementedError

    def assess_pattern(self):
        raise NotImplementedError

    def overload_weight_reset(self):
        raise NotImplementedError

    def overload_weight_pattern(self):
        raise NotImplementedError

    def overload_weight_attribution_pattern(self):
        raise NotImplementedError

    def backward_pattern(self, y_sig):
        raise NotImplementedError

    def explain_pattern(self, data, attribution=False):
        raise NotImplementedError

class LinearPatternNet(PatternNet):
    def __init__(self, *args, **kwargs):
        #self.estimator = kwargs.pop('estimator', 'linear')
        self._regimes = kwargs.pop('regimes', [PatternRegime('linear', lambda y: nd.ones_like(y))])
        self._noregconst = kwargs.pop('noregconst', 0.)
        #self._regimes = kwargs.pop('regimes', [PatternRegime('pos', lambda y: y>0)])
        super().__init__(*args, **kwargs)
        self.num_samples = None
        self.mean_y = None
        self.w_qual = None

    def init_pattern(self):
        outsize, insize = self._shape_pattern()
        with self.name_scope():
            # Screw MXNet's messing with __setattr__ of Blocks!
            # (Parameters are put into _reg_params when __setattr__ (see source of MXNet's Block's __setattr__))
            object.__setattr__(self, 'num_samples', self.pparams.get('num_samples',
                                                                     shape=(1,),
                                                                     init=mx.initializer.Zero(),
                                                                     grad_req='null'))
            object.__setattr__(self, 'mean_y', self.pparams.get('mean_y',
                                                                shape=(1, outsize),
                                                                init=mx.initializer.Zero(),
                                                                grad_req='null'))
            object.__setattr__(self, 'var_y', self.pparams.get('var_y',
                                                               shape=(1, outsize),
                                                               init=mx.initializer.Zero(),
                                                               grad_req='null'))
            object.__setattr__(self, 'mean_d', self.pparams.get('mean_d',
                                                                shape=(1, insize),
                                                                init=mx.initializer.Zero(),
                                                                grad_req='null'))
            object.__setattr__(self, 'cov_dd', self.pparams.get('cov_dd',
                                                                shape=(insize, insize),
                                                                init=mx.initializer.Zero(),
                                                                grad_req='null'))
            object.__setattr__(self, 'cov_dy', self.pparams.get('cov_dy',
                                                                shape=(outsize, insize),
                                                                init=mx.initializer.Zero(),
                                                                grad_req='null'))
            object.__setattr__(self, 'w_qual', self.pparams.get('w_qual',
                                                                shape=(outsize, insize),
                                                                init=mx.initializer.Xavier(),
                                                                grad_req='write'))
            for regime in self._regimes:
                regime.mean_x = self.pparams.get('mean_x_%s'%str(regime),
                                                 shape=(outsize, insize),
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
        bias = self.bias.data(ctx=x.context) if self.bias is not None else None
        for regime in self._regimes:
            regime.pattern_ref = self._weight(ctx=x.context).copy()
            a_reg = regime.pattern_ref
            z_reg = self._forward(x, a_reg, bias)
            # regimes are assumed to be disjunct
            if z is None:
                z = nd.ones_like(z_reg) * self._noregconst
            z = nd.where(regime(z_reg), z_reg, z)
        return z

    def overload_weight_reset(self):
        for regime in self._regimes:
            regime.pattern_ref[:] = self._weight(ctx=regime.pattern_ref.context).reshape(regime.pattern_ref.shape)

    def overload_weight_pattern(self):
        for regime in self._regimes:
            regime.pattern_ref[:] = regime.pattern.data(ctx=regime.pattern_ref.context).reshape(regime.pattern_ref.shape)

    def overload_weight_attribution_pattern(self):
        for regime in self._regimes:
            regime.pattern_ref *= regime.pattern.data(ctx=regime.pattern_ref.context).reshape(regime.pattern_ref.shape)

    def backward_pattern(self, y_sig):
        if self._out is None:
            raise RuntimeError('Block has not yet executed forward_logged!')
        y_cond = self._out
        return self._signal_pattern(y_sig, y_cond)

    def compute_pattern(self, ctx=None):
        weight = self._weight(ctx=ctx).flatten()
        for regime in self._regimes:
            cov = regime.mean_xy.data(ctx=ctx) - regime.mean_x.data(ctx=ctx) * self.mean_y.data(ctx=ctx).T
            var_y = (weight * cov).sum(axis=1, keepdims=True)
            pat = cov / (var_y + (var_y == 0.))
            regime.pattern.set_data(pat)

    def fit_pattern(self, x, xnb=None):
        if xnb is None:
            xnb = x
        y   = self(x)
        ynb = self._forward(x, self._weight(ctx=x.context))
        #computes only gradients for batch step!
        self._err = []
        for regime in self._regimes:
            loss = mx.gluon.loss.L2Loss()
            with autograd.record():
                y_reg = y * regime(y)
                pattern = regime.pattern.data(ctx=y.context)
                pias = regime.pias.data(ctx=y.context)
                signal = self._backward_pattern(y_reg, pattern, pias)
                err = loss(signal, xnb)
            err.backward()
            self._err.append(err)
        return y, ynb
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
            vtd = self._forward(distractor, w_qual)
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
            ctx = x.context
            for regime in self._regimes:
                mean_x = regime.mean_x.data(ctx=ctx)
                mean_xy = regime.mean_xy.data(ctx=ctx)
                num_y = regime.num_y.data(ctx=ctx)

                cond_y = regime(y)
                # y's in regime
                reg_y  = y * cond_y
                # sum of x's per y in regime
                sum_x  = nd.dot(cond_y, x, transpose_a=True)
                # sum of xy's in regime
                sum_xy = nd.dot(reg_y, x, transpose_a=True)

                #TODO more stable running mean
                num_y_new = num_y + cond_y.sum(axis=0, keepdims=True)
                norm      = num_y_new + (num_y_new == 0.)
                mean_x    = (num_y.T * mean_x  + sum_x ) / norm.T
                mean_xy   = (num_y.T * mean_xy + sum_xy) / norm.T

                regime.num_y  .set_data(num_y_new)
                regime.mean_x .set_data(mean_x)
                regime.mean_xy.set_data(mean_xy)

            num     = self.num_samples.data(ctx=ctx)
            num_new = num + y.shape[0]
            mean_y  = self.mean_y.data(ctx=ctx)
            sum_y   = y.sum(axis=0, keepdims=True)
            mean_y  = (num * mean_y + sum_y) / (num_new + (num_new == 0.))

            self.mean_y.set_data(mean_y)
            self.num_samples.set_data(num_new)

    def _backward_pattern(self, y, pattern, pias=None):
        raise NotImplementedError

    def _shape_pattern(self):
        raise NotImplementedError

    @staticmethod
    def _args_forward_pattern(*args):
        if len(args) == 1:
            return args[0], args[0]
        elif len(args) == 2:
            return args
        else:
            raise RuntimeError('Number of input arguments not correct!')

class ActPatternNet(PatternNet):
    def forward_pattern(self, *args):
        return self.forward(*args)

    def fit_pattern(self, x, xnb=None):
        return (self(x), None if xnb is None else self(xnb))

    def fit_assess_pattern(self, x):
        return self(x)

    def assess_pattern(self):
        return None


    def init_pattern(self):
        pass

    def overload_weight_reset(self):
        pass

    def overload_weight_pattern(self):
        pass

    def overload_weight_attribution_pattern(self):
        pass

    def learn_pattern(self):
        pass

    def stats_assess_pattern(self):
        pass

    def compute_pattern(self, ctx=None):
        pass

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


import re
import mxnet as mx
from mxnet import nd, autograd
from mxnet.gluon import nn, ParameterDict

from ..base import Block

class PatternNet(Block):
    def __init__(self, *args, **kwargs):
        #self.estimator = kwargs.pop('estimator', 'linear')
        self._regimes = kwargs.pop('regimes', [PatternRegime('linear', lambda y: nd.ones_like(y))])
        #self._regimes = kwargs.pop('regimes', [PatternRegime('pos', lambda y: y>0)])
        super().__init__(*args, **kwargs)
        self._pparams = ParameterDict(getattr(self, '_prefix', ''))
        self.num_samples = None
        self.mean_y = None

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
        for cld in self._children:
            try:
                ret.update(cld.collect_pparams(select=select))
            except AttributeError:
                pass
        return ret


    def learn_pattern(self, *args, **kwargs):
        raise NotImplementedError
        #try:
        #    func = getattr(self, 'learn_pattern_'+self.estimator)
        #except AttributeError:
        #    raise NotImplementedError(self.estimator)
        #return func(*args, **kwargs)

    def assess_pattern(self, *args, **kwargs):
        raise NotImplementedError
        #try:
        #    func = getattr(self, 'assess_pattern_'+self.estimator)
        #except AttributeError:
        #    raise NotImplementedError(self.estimator)
        #return func(*args, **kwargs)

    def forward_pattern(self, *args, **kwargs):
        raise NotImplementedError
        #try:
        #    func = getattr(self, 'forward_pattern_'+self.estimator)
        #except AttributeError:
        #    raise NotImplementedError(self.estimator)
        #return func(*args, **kwargs)

    def explain_pattern(self, *args, **kwargs):
        raise NotImplementedError
        #try:
        #    func = getattr(self, 'explain_pattern_'+self.estimator)
        #except AttributeError:
        #    raise NotImplementedError(self.estimator)
        #return func(*args, **kwargs)

    def init_pattern(self, *args, **kwargs):
        raise NotImplementedError
        #try:
        #    func = getattr(self, 'init_pattern_'+self.estimator)
        #except AttributeError:
        #    raise NotImplementedError(self.estimator)
        #return func(*args, **kwargs)

    def compute_pattern(self):
        raise NotImplementedError

    def _init_pattern(self, shape):
        outsize, insize = shape
        with self.name_scope():
            self.num_samples = self.pparams.get('num_samples',
                                               shape=(1,),
                                               init=mx.initializer.Zero(),
                                               grad_req='null')
            self.mean_y = self.pparams.get('mean_y',
                                          shape=(1, outsize),
                                          init=mx.initializer.Zero(),
                                          grad_req='null')
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
                                                    init=mx.initializer.Zero(),
                                                    grad_req='null')

    def _learn_pattern(self, x, y):
        num = self.num_samples.data()
        num_cur = x.shape[0]
        mean_y = self.mean_y.data()
        for regime in self._regimes:
            mean_x = regime.mean_x.data()
            mean_xy = regime.mean_xy.data()
            num_y = regime.num_y.data()
            num_x = num_y.sum()

            cond_y = regime(y)
            # number of times w.t dot x was inside the regime
            num_n = cond_y.sum(axis=1, keepdims=True)
            # weighted sum of x
            wsum_x = nd.dot(num_n, x, transpose_a=True)

            # y's in regime
            reg_y = y * cond_y
            # sum of xy's in regime
            # sum_xy = (reg_y.expand_dims(axis=2) * x.expand_dims(axis=1)).sum(axis=0)
            sum_xy = nd.dot(reg_y, x, transpose_a=True)


            num_x_cur = num_n.sum()
            mean_x = (num_x * mean_x + wsum_x) / (num_x + num_x_cur + 1e-12)

            num_y_cur = cond_y.sum(axis=0)
            mean_xy = (num_y.T * mean_xy + sum_xy) / (num_y + num_y_cur + 1e-12).T

            num_y += num_y_cur

            regime.num_y.set_data(num_y)
            regime.mean_x.set_data(mean_x)
            regime.mean_xy.set_data(mean_xy)

        sum_y = y.sum(axis=0, keepdims=True)
        mean_y = (num * mean_y + sum_y) / (num + num_cur + 1e-12)
        num += num_cur
        self.mean_y.set_data(mean_y)
        self.num_samples.set_data(num)

    @staticmethod
    def _args_forward_pattern(*args):
        if len(args) == 1:
            return args[0], args[0], {}
        elif len(args) == 2:
            return args[0], args[1], {}
        elif len(args) == 3:
            return args
        else:
            raise RuntimeError('Number of input arguments not correct!')

    def _compute_pattern(self, weight):
        for regime in self._regimes:
            cov = regime.mean_xy.data() - nd.dot(self.mean_y.data(), regime.mean_x.data(), transpose_a=True)
            pat = cov / ((weight * cov).sum(axis=1, keepdims=True) + 1e-12)
            regime.pattern.set_data(pat)

#    def _init_pattern_linear(self, shape):
#        outsize, insize = shape
#        with self.name_scope():
#            self.mean_x = self.pparams.get('mean_x',
#                                          shape=(1, insize),
#                                          init=mx.initializer.Zero(),
#                                          grad_req='null')
#            self.mean_y = self.pparams.get('mean_y',
#                                          shape=(1, outsize),
#                                          init=mx.initializer.Zero(),
#                                          grad_req='null')
#            self.var_y = self.pparams.get('var_y',
#                                         shape=(1, outsize),
#                                         init=mx.initializer.Zero(),
#                                         grad_req='null')
#            self.cov = self.pparams.get('cov',
#                                       shape=(outsize, insize),
#                                       init=mx.initializer.Zero(),
#                                       grad_req='null')
#            self.num_samples = self.pparams.get('num_samples',
#                                               shape=(1,),
#                                               init=mx.initializer.Zero(),
#                                               grad_req='null')
#            #self.qual_dtd = self.pparams.get('qual_dtd',
#            #                                shape=(in_units, in_units),
#            #                                init=mx.initializer.Zero(),
#            #                                grad_req='null')
#            #self.qual_dty = self.pparams.get('qual_dty',
#            #                                shape=self.weight.shape,
#            #                                init=mx.initializer.Zero(),
#            #                                grad_req='null')
#            #self.cov_dy = self.pparams.get('qual_dty',
#            #                                shape=self.weight.shape,
#            #                                init=mx.initializer.Zero(),
#            #                                grad_req='null')
#            #self.qual_v = self.pparams.get('qual_v',
#            #                              shape=(1, in_units),
#            #                              init=mx.initializer.One(),
#            #                              grad_req='null')
#
#    def _init_pattern_twocomponent(self, shape):
#        outsize, insize = shape
#        with self.name_scope():
#            self.mean_x_pos = self.pparams.get('mean_x_pos',
#                                          shape=(1, insize),
#                                          init=mx.initializer.Zero(),
#                                          grad_req='null')
#            self.mean_x_neg = self.pparams.get('mean_x_neg',
#                                          shape=(1, insize),
#                                          init=mx.initializer.Zero(),
#                                          grad_req='null')
#            self.mean_y = self.pparams.get('mean_y',
#                                          shape=(1, outsize),
#                                          init=mx.initializer.Zero(),
#                                          grad_req='null')
#            self.mean_xy_pos = self.pparams.get('mean_xy_pos',
#                                          shape=(outsize, insize),
#                                          init=mx.initializer.Zero(),
#                                          grad_req='null')
#            self.mean_xy_neg = self.pparams.get('mean_xy_neg',
#                                          shape=(outsize, insize),
#                                          init=mx.initializer.Zero(),
#                                          grad_req='null')
#
#            self.n_x_pos = self.pparams.get('n_x_pos',
#                                          shape=(1, insize),
#                                          init=mx.initializer.Zero(),
#                                          grad_req='null')
#            self.n_x_neg = self.pparams.get('n_x_neg',
#                                          shape=(1, insize),
#                                          init=mx.initializer.Zero(),
#                                          grad_req='null')
#            self.n_xy_pos = self.pparams.get('n_xy_pos',
#                                          shape=(outsize, insize),
#                                          init=mx.initializer.Zero(),
#                                          grad_req='null')
#            self.n_xy_neg = self.pparams.get('n_xy_neg',
#                                          shape=(outsize, insize),
#                                          init=mx.initializer.Zero(),
#                                          grad_req='null')
#            self.num_samples = self.pparams.get('num_samples',
#                                               shape=(1,),
#                                               init=mx.initializer.Zero(),
#                                               grad_req='null')
#
#    def _learn_pattern_linear(self, x, y):
#        mean_x = self.mean_x.data()
#        mean_y = self.mean_y.data()
#        n = self.num_samples.data()
#        var_y = self.var_y.data()
#        cov = self.cov.data()
#        m = x.shape[0]
#
#        mean_x_cur = x.mean(axis=0, keepdims=True)
#        mean_y_cur = y.mean(axis=0, keepdims=True)
#        dx = x - mean_x_cur
#        dy = y - mean_y_cur
#
#        cov_cur = nd.dot(dy, dx, transpose_a=True)
#        cov += cov_cur + nd.dot((mean_y - mean_y_cur), (mean_x - mean_x_cur), transpose_a=True) * n * m / (n+m)
#
#        var_y_cur = nd.sum(dy**2, axis=0)
#        var_y += var_y_cur + ((mean_y - mean_y_cur) * (mean_y - mean_y_cur)) * n * m / (n+m)
#
#        mean_x = (n * mean_x + m * mean_x_cur) / (n+m)
#        mean_y = (n * mean_y + m * mean_y_cur) / (n+m)
#        n += m
#
#        self.mean_x.set_data(mean_x)
#        self.mean_y.set_data(mean_y)
#        self.num_samples.set_data(n)
#        self.var_y.set_data(var_y)
#        self.cov.set_data(cov)
#
#    def _learn_pattern_twocomponent(self, x, y):
#        mean_x_pos = self.mean_x_pos.data()
#        mean_x_neg = self.mean_x_neg.data()
#        mean_y = self.mean_y.data()
#        mean_xy_pos = self.mean_xy_pos.data()
#        mean_xy_neg = self.mean_xy_neg.data()
#        n = self.num_samples.data()
#        n_x_pos = self.n_x_pos.data()
#        n_x_neg = self.n_x_neg.data()
#        n_xy_pos = self.n_xy_pos.data()
#        n_xy_neg = self.n_xy_neg.data()
#        m = x.shape[0]
#
#        x_pos = nd.maximum(0., x)
#        x_neg = nd.minimum(0., x)
#        m_x_pos = (x >= 0.).sum(axis=0, keepdims=True)
#        m_x_neg = (x < 0.).sum(axis=0, keepdims=True)
#
#        xy = y.expand_dims(axis=2) * x.expand_dims(axis=1)
#        xy_pos = nd.maximum(0., xy)
#        xy_neg = nd.minimum(0., xy)
#        m_xy_pos = (xy >= 0.).sum(axis=0)
#        m_xy_neg = (xy < 0.).sum(axis=0)
#
#        # "/m_x_pos" omitted since multiplied later
#        sum_x_pos_cur = x_pos.sum(axis=0, keepdims=True)
#        sum_x_neg_cur = x_neg.sum(axis=0, keepdims=True)
#        sum_y_cur = y.sum(axis=0, keepdims=True)
#        # note that we omit "<sum> / m_xy_pos" since we would multiply it later anyway 
#        sum_xy_pos_cur = xy_pos.sum(axis=0)
#        sum_xy_neg_cur = xy_neg.sum(axis=0)
#
#        mean_x_pos = (n_x_pos * mean_x_pos + sum_x_pos_cur) / (n_x_pos + m_x_pos + 1e-12)
#        mean_x_neg = (n_x_neg * mean_x_neg + sum_x_neg_cur) / (n_x_neg + m_x_neg + 1e-12)
#        mean_y = (n * mean_y + sum_y_cur) / (n+m)
#        mean_xy_pos = (n_xy_pos * mean_xy_pos + sum_xy_pos_cur) / (n_xy_pos + m_xy_pos + 1e-12)
#        mean_xy_neg = (n_xy_neg * mean_xy_neg + sum_xy_neg_cur) / (n_xy_neg + m_xy_neg + 1e-12)
#
#        n += m
#        n_x_pos += m_x_pos
#        n_x_neg += m_x_neg
#        n_xy_pos += m_xy_pos
#        n_xy_neg += m_xy_neg
#
#        self.mean_x_pos.set_data(mean_x_pos)
#        self.mean_x_neg.set_data(mean_x_neg)
#        self.mean_y.set_data(mean_y)
#        self.mean_xy_pos.set_data(mean_xy_pos)
#        self.mean_xy_neg.set_data(mean_xy_neg)
#        self.num_samples.set_data(n)
#        self.n_x_pos.set_data(n_x_pos)
#        self.n_x_neg.set_data(n_x_neg)
#        self.n_xy_pos.set_data(n_xy_pos)
#        self.n_xy_neg.set_data(n_xy_neg)
#
#
#    def init_pattern_linear(self, *args, **kwargs):
#        raise NotImplementedError
#
#    def learn_pattern_linear(self, *args, **kwargs):
#        raise NotImplementedError
#
#    def assess_pattern_linear(self, *args, **kwargs):
#        raise NotImplementedError
#
#    def forward_pattern_linear(self, *args):
#        raise NotImplementedError
#
#    def explain_pattern_linear(self, *args, **kwargs):
#        raise NotImplementedError
#
#
#    def init_pattern_twocomponent(self, *args, **kwargs):
#        raise NotImplementedError
#
#    def learn_pattern_twocomponent(self, *args, **kwargs):
#        raise NotImplementedError
#
#    def assess_pattern_twocomponent(self, *args, **kwargs):
#        raise NotImplementedError
#
#    def forward_pattern_twocomponent(self, *args):
#        raise NotImplementedError
#
#    def explain_pattern_twocomponent(self, *args, **kwargs):
#        raise NotImplementedError

class ActPatternNet(PatternNet):
    def init_pattern(self, *args):
        pass

    def learn_pattern(self, *args):
        pass

    def compute_pattern(self):
        pass

    def forward_pattern(self, *args):
        x_neut, x_acc, x_regs = self._args_forward_pattern(*args)

        z_neut = self.forward(x_neut)
        z_regs = {}
        z_acc = nd.zeros_like(x_neut)
        for regime in self._regimes:
            z_reg = self.forward(x_regs[regime.name])
            z_acc = nd.where(regime(z_neut), z_reg, z_acc)

        return z_neut, z_acc, z_regs

class PatternRegime(object):
    def __init__(self, name, condition):
        self.name = name
        if condition is not None:
            self.condition = condition
        self.mean_x = None
        self.mean_xy = None
        self.num_y = None
        self.pattern = None

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

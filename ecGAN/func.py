from mxnet import nd, autograd
from mxnet.gluon import nn, ParameterDict

def fuzzy_one_hot(arr, size):
    x = arr.reshape((-1, ))
    return nd.where(nd.one_hot(x, size),
                    nd.uniform(low=0.7, high=1.2, shape=(x.shape[0], size), ctx=x.context),
                    nd.uniform(low=0.0, high=0.3, shape=(x.shape[0], size), ctx=x.context))

def linspace(start=0., stop=1., num=1, ctx=None, dtype=None):
    return nd.arange(start, stop, (stop-start)/num, ctx=ctx, dtype='float32').astype(dtype)

def randint(low=0., high=1., shape=(1, ), ctx=None, dtype='int32'):
    return nd.uniform(low=low, high=high, shape=shape, ctx=ctx).astype(dtype)

def batchwise_covariance(X, Y):
        meanx = meany = vary = n = C = 0
        for x, y in zip(X, Y):
            m = len(x)
            meanx_ = x.mean(axis=0, keepdims=True)
            meany_ = y.mean(axis=0, keepdims=True)
            dx = x - meanx_
            dy = y - meany_

            C_ = nd.dot(dx, dy, transpose_a=True)
            C += C_ + nd.dot((meanx - meanx_), (meany - meany_), transpose_a=True) * n * m / (n+m)

            vary_ = nd.sum(dy**2, axis=0)
            vary += vary_ + ((meany - meany_)**2) * n * m / (n+m)

            meanx = (n * meanx + m * meanx_) / (n+m)
            meany = (n * meany + m * meany_) / (n+m)
            n += m
        return C / n, vary / n

class Intermediate(object):
    def forward(self, *args, depth=-1):
        return self.forward(self, *args)

class Interpretable(object):
    def __init__(self, *args, **kwargs):
        self._isinput = kwargs.pop('isinput', False)
        super().__init__(*args, **kwargs)
        self._in = None
        self._out = None

#    def forward(self, *args, **kwargs):
#        raise NotImplementedError
#
    def forward_logged(self, *args, **kwargs):
        self._in = args
        self._out = self.forward(*args)
        return self._out

    def relevance(self, *args, **kwargs):
        method = kwargs.pop('method', 'dtd')
        func = getattr(self, 'relevance_'+method)
        return func(*args, **kwargs)

    def relevance_sensitivity(self):
        raise NotImplementedError

    def relevance_dtd(self, a, R):
        raise NotImplementedError

class PatternNet(object):
    def __init__(self, *args, **kwargs):
        self.estimator = kwargs.pop('estimator', 'linear')
        super().__init__(*args, **kwargs)
        self._pparams = ParameterDict(getattr(self, '_prefix', ''))
        self.mean_x = None
        self.mean_y = None
        self.num_samples = None
        self.var_y = None
        self.cov = None

    def hybrid_forward(self, F, x, weight, bias=None, **kwargs):
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
        try:
            func = getattr(self, 'learn_pattern_'+self.estimator)
        except AttributeError:
            raise NotImplementedError(self.estimator)
        return func(*args, **kwargs)

    def assess_pattern(self, *args, **kwargs):
        try:
            func = getattr(self, 'assess_pattern_'+self.estimator)
        except AttributeError:
            raise NotImplementedError(self.estimator)
        return func(*args, **kwargs)

    def forward_pattern(self, *args, **kwargs):
        try:
            func = getattr(self, 'forward_pattern_'+self.estimator)
        except AttributeError:
            raise NotImplementedError(self.estimator)
        return func(*args, **kwargs)

    def explain_pattern(self, *args, **kwargs):
        try:
            func = getattr(self, 'explain_pattern_'+self.estimator)
        except AttributeError:
            raise NotImplementedError(self.estimator)
        return func(*args, **kwargs)

    def init_pattern(self, *args, **kwargs):
        try:
            func = getattr(self, 'init_pattern_'+self.estimator)
        except AttributeError:
            raise NotImplementedError(self.estimator)
        return func(*args, **kwargs)

    def init_pattern_linear(self, *args, **kwargs):
        raise NotImplementedError

    def init_pattern_twocomponent(self, *args, **kwargs):
        raise NotImplementedError

    def learn_pattern_linear(self, *args, **kwargs):
        raise NotImplementedError

    def learn_pattern_twocomponent(self, *args, **kwargs):
        raise NotImplementedError

    def assess_pattern_linear(self, *args, **kwargs):
        raise NotImplementedError

    def assess_pattern_twocomponent(self, *args, **kwargs):
        raise NotImplementedError

    def forward_pattern_linear(self, *args):
        raise NotImplementedError

    def forward_pattern_twocomponent(self, *args):
        raise NotImplementedError

    def explain_pattern_linear(self, *args, **kwargs):
        raise NotImplementedError

    def explain_pattern_twocomponent(self, *args, **kwargs):
        raise NotImplementedError


class Sequential(Interpretable, PatternNet, Intermediate, nn.Sequential):
    def forward(self, x, depth=-1):
        rdep = depth if depth > 0 else (len(self._children) + depth)
        for i, block in enumerate(self._children):
            x = block(x)
            if i == depth:
                break
        return x
    #################
    # Interpretable #
    #################
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

    ##############
    # PatternNet #
    ##############
    def init_pattern(self):
        for block in self._children:
            block.estimator = self.estimator
            block.init_pattern()

    def forward_pattern(self, *args):
        self._in = args
        x = args[0]
        for block in self._children:
            block.estimator = self.estimator
            x = block.forward_pattern(x)
        self._out = x
        return self._out

    def learn_pattern(self, *args, **kwargs):
        for block in self._children:
            block.estimator = self.estimator
            block.learn_pattern()

    def explain_pattern(self, *args, **kwargs):
        x = args[0]
        x.attach_grad()
        with autograd.record():
            y = self.forward_pattern(x)
        y.backward()
        return x.grad

class YSequential(Interpretable, Intermediate, nn.Block):
    def __init__(self, **kwargs):
        self._concat_dim = kwargs.pop('concat_dim', 1)
        super().__init__(**kwargs)
        with self.name_scope():
            self._data_net = Sequential()
            self.register_child(self._data_net)

            self._cond_net = Sequential()
            self.register_child(self._cond_net)

            self._main_net = Sequential()
            self.register_child(self._main_net)

    def addData(self, *args, **kwargs):
        with self._data_net.name_scope():
            self._data_net.add(*args, **kwargs)

    def addCond(self, *args, **kwargs):
        with self._cond_net.name_scope():
            self._cond_net.add(*args, **kwargs)

    def add(self, *args, **kwargs):
        with self._main_net.name_scope():
            self._main_net.add(*args, **kwargs)

    def forward_logged(self, x, y, depth=-1):
        self._in = [x, y]
        data = self._data_net.forward_logged(x)
        cond = self._cond_net.forward_logged(y)
        combo = nd.concat(data, cond, dim=self._concat_dim)
        self._out = self._main_net.forward_logged(combo, depth=depth)
        return self._out

    def forward(self, x, y, depth=-1):
        data = self._data_net(x)
        cond = self._cond_net(y)
        combo = nd.concat(data, cond, dim=self._concat_dim)
        return self._main_net.forward(combo, depth=depth)

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

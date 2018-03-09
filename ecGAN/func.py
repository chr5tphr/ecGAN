from mxnet import nd, autograd
from mxnet.gluon import nn, ParameterDict

def get_im2col_indices(x_shape, field_height, field_width, padding=1, stride=1, ctx=None):
    # First figure out what the size of the output should be
    N, C, H, W = x_shape
    assert (H + 2 * padding - field_height) % stride == 0
    assert (W + 2 * padding - field_height) % stride == 0
    out_height = int((H + 2 * padding - field_height) / stride + 1)
    out_width = int((W + 2 * padding - field_width) / stride + 1)

    i0 = nd.repeat(nd.arange(field_height, ctx=ctx), field_width)
    i0 = nd.tile(i0, C)
    i1 = stride * nd.repeat(nd.arange(out_height, ctx=ctx), out_width)
    j0 = nd.tile(nd.arange(field_width, ctx=ctx), field_height * C)
    j1 = stride * nd.tile(nd.arange(out_width, ctx=ctx), out_height)
    i = i0.reshape((-1, 1)) + i1.reshape((1, -1))
    j = j0.reshape((-1, 1)) + j1.reshape((1, -1))

    k = nd.repeat(nd.arange(C, ctx=ctx), field_height * field_width).reshape((-1, 1))

    return (k.astype('int32'), i.astype('int32'), j.astype('int32'))

def im2col_indices(x, field_height, field_width, padding, stride):
    """ An implementation of im2col based on some fancy indexing """
    # Zero-pad the input
    ctx = x.context
    p = padding
    x_padded = nd.pad(x, pad_width=(0, 0, 0, 0, p, p, p, p), mode='constant')

    k, i, j = get_im2col_indices(x.shape, field_height, field_width, padding, stride, ctx=ctx)

    cols = x_padded[:, k, i, j]
    C = x.shape[1]
    cols = cols.transpose((1, 2, 0)).reshape((field_height * field_width * C, -1))
    return cols

def col2im_indices(cols, x_shape, field_height=3, field_width=3, padding=1,
                   stride=1, ctx=None):
    """ An implementation of col2im based on fancy indexing and np.add.at """
    N, C, H, W = x_shape
    H_padded, W_padded = H + 2 * padding, W + 2 * padding
    x_padded = nd.zeros((N, C, H_padded, W_padded), dtype=cols.dtype, ctx=ctx)
    k, i, j = get_im2col_indices(x_shape, field_height, field_width, padding, stride, ctx=ctx)
    cols_reshaped = cols.reshape((C * field_height * field_width, -1, N))
    cols_reshaped = cols_reshaped.transpose((2, 0, 1))
    # The for loop is probably a bottleneck, but cannot be avoided without a nd.add.at function
    #for l in nd.arange(cols.shape[1]):
    #    x_padded[:,k,i[:,l], j[:,l]] += cols_reshaped[:,:,l]
    for col in nd.arange(cols.shape[0], ctx=ctx):
        x_padded[:,k[col],i[col,:], j[col,:]] += cols_reshaped[:,col,:]
    #np.add.at(x_padded, (slice(None), k, i, j), cols_reshaped)
    if padding == 0:
        return x_padded
    return x_padded[:, :, padding:-padding, padding:-padding]

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
        x = args
        for block in self._children:
            block.estimator = self.estimator
            x = block.forward_pattern(*x)
        self._out = x[0]
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

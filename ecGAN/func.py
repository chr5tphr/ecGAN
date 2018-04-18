import re
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

def stats_batchwise(x_bat, y_bat, n, x_mean, y_mean, x_var=None, y_var=None, cov=None, x_mean_skip=False, y_mean_skip=False):
    m = x.shape[0]

    x_bat_mean = x.mean(axis=0, keepdims=True)
    y_bat_mean = y.mean(axis=0, keepdims=True)

    dx = x - x_bat_mean
    dy = y - y_bat_mean

    if x_var is not None:
        x_bat_var = nd.sum(dx**2, axis=0)
        x_var += x_bat_var + ((x_mean - x_bat_mean)**2) * n * m / (n+m)

    if y_var is not None:
        y_bat_var = nd.sum(dy**2, axis=0)
        y_var += y_bat_var + ((y_mean - y_bat_mean)**2) * n * m / (n+m)

    if cov is not None:
        cov_bat = nd.dot(dx, dy, transpose_a=True)
        cov += cov_bat + nd.dot((x_mean - x_bat_mean), (y_mean - y_bat_mean), transpose_a=True) * n * m / (n+m)

    if not x_mean_skip:
        x_mean = (n * x_mean + m * x_bat_mean) / (n+m)

    if not y_mean_skip:
        y_mean = (n * y_mean + m * y_bat_mean) / (n+m)

    n += m

    return n, x_mean, y_mean, x_var, y_var, cov

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

def cov_reg_it(self, x, y, mean_x, mean_xy, num_x, num_y, regime):
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


    num_x_cur = num_n.sum()
    mean_x = (num_x * mean_x + wsum_x) / (num_x + num_x_cur + 1e-12)

    num_y_cur = cond_y.sum(axis=0)
    mean_xy = (num_y.T * mean_xy + sum_xy) / (num_y + num_y_cur + 1e-12).T

    num_y += num_y_cur

    return mean_x, mean_xy, num_y


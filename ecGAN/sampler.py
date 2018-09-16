import mxnet as mx
import numpy as np
import logging

from mxnet import nd

from .func import linspace

getLogger = logging.getLogger

samplers = {}
def register_sampler(func):
    samplers[func.__name__] = func
    return func

def asim(*args):
    return [x.reshape(list(x.shape) + [1, 1]) for x in args]

@register_sampler
def fully_random_uniform(num, K, ctx, d=100, ohkw={}):
    noise      = nd.random_normal(shape=(num, d), ctx=ctx)
    cond_flat  = nd.random.uniform(0, K, shape=num, ctx=ctx).floor().one_hot(K, **ohkw)

    return noise, cond

@register_sampler
def fully_random_uniform(num, K, ctx, d=100, ohkw={}):
    noise      = nd.random_normal(shape=(num, d), ctx=ctx)
    cond_flat  = nd.random.uniform(0, K, shape=num, ctx=ctx).floor().one_hot(K, **ohkw)

    return noise, cond

@register_sampler
def random_uniform(num, K, ctx, d=100, ohkw={}):
    ncol, nrow = (num-1)//K+1, K
    noise_row  = nd.random_normal(shape=(ncol, 1, d), ctx=ctx)
    cond_col   = nd.random.uniform(0, K, shape=(1, nrow), ctx=ctx).floor()

    noise      = noise_row.tile([1, nrow])                   .reshape([ncol*nrow, d])[:num]
    cond       = cond_col .tile([ncol, 1]).one_hot(K, **ohkw).reshape([ncol*nrow, K])[:num]

    return noise, cond

@register_sampler
def random_counter(num, K, ctx, d=100, ohkw={}):
    nrow, ncol = K, (num-1)//K+1
    cond_col   = nd.arange(nrow, ctx=ctx).reshape([1, nrow])

    noise      = nd.random_normal(shape=(num, d), ctx=ctx)
    cond       = cond_col .tile([ncol, 1]).one_hot(K, **ohkw).reshape([ncol*nrow,   K])[:num]

    return noise, cond

@register_sampler
def counter(num, K, ctx, d=100, ohkw={}):
    nrow, ncol = K, (num-1)//K+1
    noise_row  = nd.random_normal(shape=(ncol, 1, d), ctx=ctx)
    cond_col   = nd.arange(nrow, ctx=ctx).reshape([1, nrow])

    noise      = noise_row.tile([1, nrow])                   .reshape([ncol*nrow, d])[:num]
    cond       = cond_col .tile([ncol, 1]).one_hot(K, **ohkw).reshape([ncol*nrow, K])[:num]

    return noise, cond

@register_sampler
def grow(num, K, ctx, d=100, ohkw={}):
    nrow, ncol  = K, (num-1)//K+1
    noise_one   = nd.random_normal(shape=(1, 1, d), ctx=ctx)
    noise       = noise_one.tile([ncol, nrow]).reshape([ncol*nrow, d])[:num]

    onval = ohkw.get('on_value', 1.0)
    offval = ohkw.get('off_value', -1.0)

    cond_col_d  = nd.arange(nrow, ctx=ctx).reshape([1, nrow]).tile([ncol, 1])
    cond_col    = cond_col_d.one_hot(K, **ohkw)
    alpha       = linspace(offval, onval, ncol, end=True, ctx=ctx).reshape([ncol, 1, 1]) * cond_col_d.one_hot(K) + offval * cond_col_d.one_hot(K, off_value=1., on_value=0.)
    cond        = alpha.reshape([ncol*nrow, K])[:num]

    return noise, cond

@register_sampler
def transform(num, K, ctx, d=100, ohkw={}):
    nrow, ncol  = K, (num-1)//K+1
    noise_one   = nd.random_normal(shape=(1, 1, d), ctx=ctx)
    noise       = noise_one.tile([ncol, nrow]).reshape([ncol*nrow, d])[:num]

    onval = ohkw.get('on_value', 1.0)
    offval = ohkw.get('off_value', -1.0)

    cond_col_ds = nd.arange(nrow, ctx=ctx).reshape([1, nrow]).tile([ncol, 1])
    cond_col_dt = cond_col_ds[:, list(range(1,nrow)) + [0]]
    cond_col_s  = cond_col_ds.one_hot(K)
    cond_col_t  = cond_col_dt.one_hot(K)

    alpha       = linspace(offval, onval, ncol, end=True, ctx=ctx).reshape([ncol, 1, 1]) * cond_col_t
    beta        = linspace(onval, offval, ncol, end=True, ctx=ctx).reshape([ncol, 1, 1]) * cond_col_s
    offvals     = offval * cond_col_ds.one_hot(K, off_value=1., on_value=0.) * cond_col_dt.one_hot(K, off_value=1., on_value=0.)
    cond        = (beta + alpha + offvals).reshape([ncol*nrow, K])[:num]

    return noise, cond

import mxnet as mx
import numpy as np
import logging

from .func import linspace

getLogger = logging.getLogger

samplers = {}
def register_sampler(func):
    samplers[func.__name__] = func
    return func

def asim(*args):
    return [x.reshape(list(x.shape) + [1, 1]) for x in args]

@register_sampler
def fully_random_uniform(num, K, ctx=None)
    noise      = nd.random_normal(shape=(num, 100), ctx=ctx)
    cond_flat  = nd.random.uniform(0, K, shape=num, ctx=ctx).floor().one_hot(K)

    return noise, cond

@register_sampler
def random_uniform(num, K, ctx=None):
    ncol, nrow = (num-1)//K+1, K
    noise_row  = nd.random_normal(shape=(ncol, 1, 100), ctx=ctx)
    cond_col   = nd.random.uniform(0, K, shape=(1, nrow), ctx=ctx).floor()

    noise      = noise_row.tile([1, nrow])           .reshape([ncol*nrow, 100])[:num]
    cond       = cond_col .tile([ncol, 1]).one_hot(K).reshape([ncol*nrow,  10])[:num]

    return noise, cond

@register_sampler
def random_counter(num, K, ctx=None):
    nrow, ncol = K, (num-1)//K+1
    cond_col   = nd.arange(nrow, ctx=ctx).reshape([1, nrow])

    noise      = nd.random_normal(shape=(num, 100), ctx=ctx)
    cond       = cond_col .tile([ncol, 1]).one_hot(K).reshape([ncol*nrow,  10])[:num]

    return noise, cond

@register_sampler
def counter(num, K, ctx=None):
    nrow, ncol = K, (num-1)//K+1
    noise_row  = nd.random_normal(shape=(ncol, 1, 100), ctx=ctx)
    cond_col   = nd.arange(nrow, ctx=ctx).reshape([1, nrow])

    noise      = noise_row.tile([1, nrow])           .reshape([ncol*nrow, 100])[:num]
    cond       = cond_col .tile([ncol, 1]).one_hot(K).reshape([ncol*nrow,  10])[:num]

    return noise, cond

@register_sampler
def grow(num, K, ctx=None):
    nrow, ncol  = K, (num-1)//K+1
    noise_one   = nd.random_normal(shape=(1, 1, 100), ctx=ctx)
    noise       = noise_one.tile([ncol, nrow]).reshape([ncol*nrow, 100])[:num]

    cond_col    = nd.arange(nrow, ctx=ctx).reshape([1, nrow]).tile([ncol, 1]).one_hot(K)
    alpha       = linspace(0, 1, ncol).reshape([ncol, 1, 1])
    cond        = (alpha * cond_col).reshape([ncol*nrow, K])[:num]

    return noise, cond

@register_sampler
def transform(num, K, ctx=None):
    nrow, ncol  = K, (num-1)//K+1
    noise_one   = nd.random_normal(shape=(1, 1, 100), ctx=ctx)
    noise       = noise_one.tile([ncol, nrow]).reshape([ncol*nrow, 100])[:num]

    cond_col_s  = nd.arange(nrow, ctx=ctx).reshape([1, nrow]).tile([ncol, 1]).one_hot(K)
    cond_col_t  = nd.concatenate([cond_col_s[1:], cond_col_s[-1:]], axis=1)

    alpha       = linspace(0, 1, ncol).reshape([ncol, 1, 1])
    cond        = (alpha[::-1] * cond_col_s + alpha * cond_col_t ).reshape([ncol*nrow, K])[:num]

    return noise, cond

import mxnet as mx

from mxnet import nd

def fuzzy_one_hot(x,size,ctx):
    return nd.where(nd.one_hot(x,size),
                    nd.uniform(low=0.7,high=1.2,shape=(x.shape[0],size),ctx=ctx),
                    nd.uniform(low=0.0,high=0.3,shape=(x.shape[0],size),ctx=ctx))

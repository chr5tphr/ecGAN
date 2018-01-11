import mxnet as mx

from mxnet import nd, gluon
from mxnet.gluon import nn


def fuzzy_one_hot(x,size):
    return nd.where(nd.one_hot(x,size),
                    nd.uniform(low=0.7,high=1.2,shape=(x.shape[0],size),ctx=x.context),
                    nd.uniform(low=0.0,high=0.3,shape=(x.shape[0],size),ctx=x.context))


class Interpretable(nn.Block):
    pass

class MaxOut(nn.Block):
    pass

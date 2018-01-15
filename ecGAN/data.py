import mxnet as mx
import numpy as np

from mxnet import nd

data_funcs = {}
def register_data_func(func):
    data_funcs[func.__name__] = func
    return func

@register_data_func
def get_mnist(train,ret_label=True,tanh=True):
    def transform(data,label):
        data = (data.astype(np.float32)/255.)
        if tanh:
            data = data*2. - 1.
        label = label.astype(np.float32)
        return (data,label) if ret_label else data

    return mx.gluon.data.vision.MNIST(train=train,transform=transform)

@register_data_func
def get_mnist_cond(train):
    def transform(data,label):
        data = (data.astype(np.float32)/255.)*2. - 1.
        label = label.astype(np.float32)
        return data,label

    return mx.gluon.data.vision.MNIST(train=train,transform=transform)

@register_data_func
def get_mnist_single(train,label):
    def transform(tdata,tlabel):
        if tlabel != label:
            return None
        return (tdata.astype(np.float32)/255.)*2. - 1.

    ldata = [dat for dat in mx.gluon.data.vision.MNIST(train=train,transform=transform) if dat is not None]
    return ldata

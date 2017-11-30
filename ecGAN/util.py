import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import mxnet as mx
from mxnet import nd
import logging

data_funcs = {}
def register_data_func(func):
    data_funcs[func.__name__] = func
    return func

def plot_data(data):
    snum = int(len(data)**.5)
    if type(data) is nd.NDArray:
        data = data.asnumpy()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.imshow(data[:snum**2].reshape(snum,snum,28,28).transpose(0,2,1,3).reshape(snum*28,snum*28),cmap='Greys')
    ax.axis('off')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.hlines(np.arange(1,snum)*28,0,snum*28)
    ax.vlines(np.arange(1,snum)*28,0,snum*28)
    fig.tight_layout()
    return fig

@register_data_func
def get_mnist(train,batch_size):
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

def mkfilelogger(lname,fname,level=logging.INFO):
    logger = logging.getLogger(lname)
    logger.setLevel(level)
    frmt = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fhdl = logging.FileHandler(fname)
    shdl = logging.StreamHandler()
    fhdl.setLevel(level)
    shdl.setLevel(level)
    fhdl.setFormatter(frmt)
    shdl.setFormatter(frmt)
    logger.addHandler(shdl)
    logger.addHandler(fhdl)
    return logger

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import mxnet as mx
from mxnet import nd

def plot_data(data):
    snum = int(len(data)**.5)
    if type(data) is nd.NDArray:
        data = data.asnumpy()
        plt.imshow(data[:snum**2].reshape(snum,snum,28,28).transpose(0,2,1,3).reshape(snum*28,snum*28),cmap='Greys')
        plt.axis('off')
        plt.tight_layout()
        plt.xticks([])
        plt.yticks([])
        plt.hlines(np.arange(1,snum)*28,0,snum*28)
        plt.vlines(np.arange(1,snum)*28,0,snum*28)
        plt.show()

def get_mnist(train,batch_size):
    def transform(data,label):
        data = (data.astype(np.float32)/255.)*2. - 1.
        label = label.astype(np.float32)
        return data,label

    return mx.gluon.data.vision.MNIST(train=train,transform=transform)

def get_mnist_single(train,label):
    def transform(tdata,tlabel):
        if tlabel != label:
            return None
        return (tdata.astype(np.float32)/255.)*2. - 1.

    ldata = [dat for dat in mx.gluon.data.vision.MNIST(train=train,transform=transform) if dat is not None][:500]
    return ldata

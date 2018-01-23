import mxnet as mx
import numpy as np

from mxnet import nd
from mxnet.gluon.data import Dataset

data_funcs = {}
def register_data_func(func):
    data_funcs[func.__name__] = func
    return func

@register_data_func
def mnist(train,ret_label=True,bbox=(-1,1),preload=False,ctx=None):
    def transform(data,label):
        data = (data.astype(np.float32)/255.) * (bbox[1]-bbox[0]) + bbox[0]
        label = label.astype(np.float32)
        return (data,label) if ret_label else data

    dataset = mx.gluon.data.vision.MNIST(train=train,transform=transform)
    if (not preload) or (ctx is None):
        return dataset

    return PreloadedDataset(dataset,ctx)


class PreloadedDataset(Dataset):
    def __init__(self,dataset,ctx,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self._length = len(dataset)

        self._data = nd.zeros(dataset._data.shape,dtype='float32',ctx=ctx)
        self._label = nd.zeros(dataset._label.shape,dtype='int32',ctx=ctx)

        for i in range(self._length):
            dat,lab = dataset[i]
            self._data[i] = dat.astype('float32')
            self._label[i] = lab.astype('int32')

    def __getitem__(self,idx):
        return (self._data[idx],self._label[idx])

    def __len__(self):
        return self._length

@register_data_func
def mnist_cond(train):
    def transform(data,label):
        data = (data.astype(np.float32)/255.)*2. - 1.
        label = label.astype(np.float32)
        return data,label

    return mx.gluon.data.vision.MNIST(train=train,transform=transform)

@register_data_func
def mnist_single(train,label):
    def transform(tdata,tlabel):
        if tlabel != label:
            return None
        return (tdata.astype(np.float32)/255.)*2. - 1.

    ldata = [dat for dat in mx.gluon.data.vision.MNIST(train=train,transform=transform) if dat is not None]
    return ldata

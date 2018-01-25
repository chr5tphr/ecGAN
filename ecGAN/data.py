import mxnet as mx
import numpy as np

from mxnet import nd
from mxnet.gluon.data import Dataset

data_funcs = {}
def register_data_func(func):
    data_funcs[func.__name__] = func
    return func

@register_data_func
def mnist(train,ctx,bbox=(-1,1),labels=None):
    def transform(data,label):
        data = (data.astype('float32')/255.) * (bbox[1]-bbox[0]) + bbox[0]
        label = label.astype('int32')
        return (data,label)

    dataset = mx.gluon.data.vision.MNIST(train=train,transform=transform)
    return PreloadedDataset(dataset,ctx,labels=labels)


class PreloadedDataset(Dataset):
    def __init__(self,dataset,ctx,labels=None,*args,**kwargs):
        super().__init__(*args,**kwargs)

        if labels is not None:
            llen = 0
            for cond in labels:
                llen += (dataset._label == cond).sum()
            self._length = llen
        else:
            self._length = len(dataset)

        self._data = nd.zeros([self._length] + list(dataset._data.shape)[1:],dtype='float32',ctx=ctx)
        self._label = nd.zeros([self._length] + list(dataset._label.shape)[1:],dtype='int32',ctx=ctx)

        uniques = set()
        i = 0
        for dat,dlab in dataset:
            lab = dlab.item()
            if labels is None or ([lab == cond for cond in labels]):
                self._data[i] = dat
                self._label[i] = lab
                i += 1
                uniques.add(lab)
        self.classes = list(uniques)


    def __getitem__(self,idx):
        return (self._data[idx],self._label[idx])

    def __len__(self):
        return self._length

# @register_data_func
# def mnist_cond(train):
#     def transform(data,label):
#         data = (data.astype(np.float32)/255.)*2. - 1.
#         label = label.astype(np.float32)
#         return data,label
#
#     return mx.gluon.data.vision.MNIST(train=train,transform=transform)
#
# @register_data_func
# def mnist_single(train,label):
#     def transform(tdata,tlabel):
#         if tlabel != label:
#             return None
#         return (tdata.astype(np.float32)/255.)*2. - 1.
#
#     ldata = [dat for dat in mx.gluon.data.vision.MNIST(train=train,transform=transform) if dat is not None]
#     return ldata

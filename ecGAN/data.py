import mxnet as mx
import numpy as np

from mxnet import nd
from mxnet.gluon.data import Dataset, ArrayDataset as ArrayDatasetBase

data_funcs = {}
def register_data_func(func):
    data_funcs[func.__name__] = func
    return func

@register_data_func
def mnist(train, ctx, bbox=(-1, 1), labels=None, pad=True):
    def transform(data, label):
        data = ((data.astype('float32')/255.) * (bbox[1]-bbox[0]) + bbox[0]).reshape((1, 28, 28))
        if pad:
            data = nd.pad(data.reshape(1,1,28,28), 'constant', constant_value=bbox[0], pad_width=[0,0,0,0, 2,2,2,2])[0]
        label = label.astype('int32')
        return (data, label)

    shape = [1,28,28]
    if pad:
        shape = [1,32,32]
    dataset = mx.gluon.data.vision.MNIST(train=train, transform=transform)
    return PreloadedDataset(dataset, ctx, labels=labels, shape=shape, label_shape=[])

@register_data_func
def cifar10(train, ctx, bbox=(-1, 1), labels=None, grey=False):
    def transform(data, label):
        data = ((data.astype('float32')/255.) * (bbox[1]-bbox[0]) + bbox[0]).reshape((3, 32, 32))
        if grey:
            data = data.mean(axis=0, keepdims=True)
        label = label.astype('int32')
        return (data, label)
    C = 1 if grey else 3

    dataset = mx.gluon.data.vision.CIFAR10(train=train, transform=transform)
    return PreloadedDataset(dataset, ctx, labels=labels, shape=(C, 32, 32), label_shape=[])

@register_data_func
def toydata(train, ctx, bbox=[-1., 1.], N=1000, F=[1,32,32], K=10, seed=0xdeadbeef, **kwargs):
    rs = np.random.RandomState(seed)
    f = np.prod(F)
    means = rs.uniform(bbox[0], bbox[1], size=[K, f])
    if not train:
        rs = np.random.RandomState(seed + 1)
    data_l = np.concatenate([rs.multivariate_normal(m, np.eye(f)*(bbox[1]-bbox[0])/100, size=N//K) for m in means], axis=0)
    label_l = np.repeat(np.arange(K, dtype=int), N//K)
    perm = rs.permutation((N//K)*K)
    data = data_l[perm].reshape([N] + F).clip(bbox[0], bbox[1]).astype(np.float32)
    label = label_l[perm].astype(np.int64)
    return ArrayDataset(data, label, classes=np.arange(K).tolist())

class ArrayDataset(ArrayDatasetBase):
    def __init__(self, *args, classes=[], **kwargs):
        super().__init__(*args, **kwargs)
        self.classes = classes

class PreloadedDataset(Dataset):
    def __init__(self, dataset, ctx, labels=None, shape=None, label_shape=None, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if labels is not None:
            llen = 0
            for cond in labels:
                llen += (dataset._label == cond).sum()
            self._length = llen
        else:
            self._length = len(dataset)

        if shape is None:
            shape = dataset._data.shape[1:]
        if label_shape is None:
            label_shape = dataset._label.shape[1:]

        self._data = nd.zeros([self._length] + list(shape), dtype='float32', ctx=ctx)
        self._label = nd.zeros([self._length] + list(label_shape), dtype='int32', ctx=ctx)

        uniques = set()
        i = 0
        for dat, dlab in dataset:
            lab = dlab.item()
            if labels is None or np.any([lab == cond for cond in labels]):
                self._data[i] = dat
                self._label[i] = lab
                i += 1
                uniques.add(lab)
        self.classes = list(uniques)


    def __getitem__(self, idx):
        return (self._data[idx], self._label[idx])

    def __len__(self):
        return self._length

# @register_data_func
# def mnist_cond(train):
#     def transform(data, label):
#         data = (data.astype(np.float32)/255.)*2. - 1.
#         label = label.astype(np.float32)
#         return data, label
#
#     return mx.gluon.data.vision.MNIST(train=train, transform=transform)
#
# @register_data_func
# def mnist_single(train, label):
#     def transform(tdata, tlabel):
#         if tlabel != label:
#             return None
#         return (tdata.astype(np.float32)/255.)*2. - 1.
#
#     ldata = [dat for dat in mx.gluon.data.vision.MNIST(train=train, transform=transform) if dat is not None]
#     return ldata

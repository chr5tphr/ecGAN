import matplotlib as mpl
from matplotlib import pyplot as plt
import mxnet as mx
from mxnet import gluon, autograd, nd
from mxnet.gluon import nn
import numpy as np
import h5py

class Generator(nn.Sequential):
    def __init__(self,**kwargs):
        super().__init__(**kwargs)
        with self.name_scope():
            self.add(gluon.nn.Dense(64,activation='relu'))
            self.add(gluon.nn.Dense(784,activation='tanh'))

class Discriminator(nn.Sequential):
    def __init__(self,**kwargs):
        super().__init__(**kwargs)
        with self.name_scope():
            self.add(nn.Dense(64, activation='relu'))
            self.add(nn.Dense(64, activation='relu'))
            self.add(nn.Dense(2))

def get_all_data(train,batch_size):
    def transform(data,label):
        data = (data.astype(np.float32)/255.)*2. - 1.
        label = label.astype(np.float32)
        return data,label

    return mx.gluon.data.DataLoader(mx.gluon.data.vision.MNIST(train=train,transform=transform),batch_size,shuffle=True)

# data = mx.gluon.data.vision.MNIST(train=True,transform=transform)
# data._data
# super(type(data))

def get_data(train,batch_size):
    def transform(data,label):
        if label != 5:
            return None
        return (data.astype(np.float32)/255.)*2. - 1.

    ldata = [dat for dat in mx.gluon.data.vision.MNIST(train=train,transform=transform) if dat is not None][:500]
    return mx.gluon.data.DataLoader(ldata,batch_size,shuffle=True,last_batch='discard')

ctx = mx.cpu()

batch_size = 20
train_data = get_data(train=True,batch_size=batch_size)

# X = nd.random_normal(shape=(1000, 2))
# A = nd.array([[1, 2], [-0.1, 0.5]])
# b = nd.array([1, 2])
# X = nd.dot(X,A) + b
# Y = nd.ones(shape=(1000,1))
#
# # and stick them into an iterator
# batch_size = 4
# train_data = mx.io.NDArrayIter(X, Y, batch_size, shuffle=True)
#
# plt.scatter(X[:, 0].asnumpy(),X[:,1].asnumpy())
# plt.show()
# print("The covariance matrix is")
# print(nd.dot(A, A.T))

netG = Generator()

netD = Discriminator()

# loss
loss = gluon.loss.SoftmaxCrossEntropyLoss()

# initialize the generator and the discriminator
netG.initialize(mx.init.Normal(0.02), ctx=ctx)
netD.initialize(mx.init.Normal(0.02), ctx=ctx)

# trainer for the generator and the discriminator
trainerG = gluon.Trainer(netG.collect_params(), 'adam', {'learning_rate': 0.01})
trainerD = gluon.Trainer(netD.collect_params(), 'adam', {'learning_rate': 0.05})

real_label = mx.nd.ones((batch_size,), ctx=ctx)
fake_label = mx.nd.zeros((batch_size,), ctx=ctx)
metric = mx.metric.Accuracy()

#%%
# set up logging
from datetime import datetime
import os
import time


stamp =  datetime.now().strftime('%Y_%m_%d-%H_%M')
for epoch in range(10):
    tic = time.time()
    # train_data.reset()
    for i, data in enumerate(train_data):
        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################
        # train with real_t

        data = data.as_in_context(ctx).reshape((-1,784))

        # data = batch.data[0].as_in_context(ctx)
        noise = nd.random_normal(shape=(data.shape[0], 32), ctx=ctx)

        with autograd.record():
            real_output = netD(data)
            errD_real = loss(real_output, real_label)

            fake = netG(noise)
            fake_output = netD(fake.detach())
            errD_fake = loss(fake_output, fake_label)
            errD = errD_real + errD_fake
            errD.backward()

        trainerD.step(batch_size)
        metric.update([real_label,], [real_output,])
        metric.update([fake_label,], [fake_output,])

        ############################
        # (2) Update G network: maximize log(D(G(z)))
        ###########################
        with autograd.record():
            output = netD(fake)
            errG = loss(output, real_label)
            errG.backward()

        trainerG.step(batch_size)

    name, acc = metric.get()
    metric.reset()
    print('\nbinary training acc at epoch %d: %s=%f' % (epoch, name, acc))
    print('time: %f' % (time.time() - tic))
    noise = nd.random_normal(shape=(25, 32), ctx=ctx)
    fake = netG(noise)
    plt.imshow(fake.asnumpy().reshape(5,5,28,28).transpose(0,2,1,3).reshape(5*28,5*28),cmap='Greys')
    plt.show()
    # plt.scatter(X[:, 0].asnumpy(),X[:,1].asnumpy())
    # plt.scatter(fake[:,0].asnumpy(),fake[:,1].asnumpy())
    # plt.show()

# noise = mx.nd.random_normal(shape=(100, 2), ctx=ctx)
# fake = netG(noise)
#
# plt.scatter(X[:, 0].asnumpy(),X[:,1].asnumpy())
# plt.scatter(fake[:,0].asnumpy(),fake[:,1].asnumpy())
# plt.show()

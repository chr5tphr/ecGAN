import numpy as np
import time
import mxnet as mx
from mxnet import gluon, autograd, nd
from mxnet.gluon import nn
from string import Template

nets = {}
def register_net(obj):
    nets[obj.__name__] = obj
    return obj

@register_net
class GenFC(nn.Sequential):
    def __init__(self,**kwargs):
        super().__init__(**kwargs)
        with self.name_scope():
            self.add(gluon.nn.Dense(256))
            self.add(nn.BatchNorm(axis=1,center=True,scale=True))
            self.add(nn.LeakyReLU(0.01))
            self.add(nn.Dropout(.5))
            self.add(nn.Dense(784,activation='tanh'))

@register_net
class DiscrFC(nn.Sequential):
    def __init__(self,**kwargs):
        super().__init__(**kwargs)
        with self.name_scope():
            self.add(nn.Dense(256))
            # self.add(nn.BatchNorm(axis=1,center=True,scale=True))
            self.add(nn.LeakyReLU(0.01))
            self.add(nn.Dropout(.5))
            self.add(nn.Dense(256))
            # self.add(nn.BatchNorm(axis=1,center=True,scale=True))
            self.add(nn.LeakyReLU(0.01))
            self.add(nn.Dropout(.5))
            self.add(nn.Dense(2))

def train_GAN(data,batch_size,netG,netD,ctx,nepochs,**kwargs):

    logger = kwargs.get('logger')
    config = kwargs.get('config',{})

    chkfreq = config.get('chkfreq',0)
    start_epoch = config.get('start_epoch',0)

    train_data = mx.gluon.data.DataLoader(data,batch_size,shuffle=True,last_batch='discard')

    # loss
    loss = gluon.loss.SoftmaxCrossEntropyLoss()

    # trainer for the generator and the discriminator
    trainerG = gluon.Trainer(netG.collect_params(), 'adam', {'learning_rate': 0.01})
    trainerD = gluon.Trainer(netD.collect_params(), 'adam', {'learning_rate': 0.05})

    real_label = nd.uniform(low=.7,high=1.2,shape=(batch_size,),ctx=ctx)
    fake_label = nd.uniform(low=0.,high=.3,shape=(batch_size,),ctx=ctx)
    metric = mx.metric.Accuracy()

    for epoch in range(start_epoch, start_epoch + nepochs):
        tic = time.time()
        # train_data.reset()
        for i, data in enumerate(train_data):
            ############################
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            ###########################

            data = data.as_in_context(ctx).reshape((-1,784))

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
        if logger is not None:
            logger.info('netD training acc epoch %04d: %s=%f , time: %f',epoch, name, acc, (time.time() - tic))

        if (chkfreq > 0) and ( ( (epoch + 1) % chkfreq) == 0):
            if config.saveG:
                netG.save_params(config.sub('saveG',epoch=epoch+1))
            if config.saveD:
                netD.save_params(config.sub('saveD',epoch=epoch+1))

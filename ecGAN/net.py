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

class GAN(object):
    def __init__(self,netG,netD,ctx,**kwargs):
        self.logger = kwargs.get('logger')
        self.config = kwargs.get('config',{})

        self.save_freq = self.config.save_freq
        self.start_epoch = self.config.get('start_epoch',0)

        self.netG = netG
        self.netD = netD

        self.ctx = ctx

    def train(self,data,batch_size,nepochs):

        logger = self.logger
        config = self.config

        save_freq = self.save_freq
        start_epoch = self.start_epoch

        netG = self.netG
        netD = self.netD

        ctx = self.ctx

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
            if logger:
                logger.info('netD training acc epoch %04d: %s=%f , time: %f',epoch, name, acc, (time.time() - tic))

            if ( (save_freq > 0) and not ( (epoch + 1) % save_freq) ) or  ((epoch + 1) >= (start_epoch + nepochs)) :
                if config.saveG:
                    fpath = config.sub('saveG',epoch=epoch+1)
                    netG.save_params(fpath)
                    if logger:
                        logger.info('Saved generator \'%s\' checkpoint epoch %04f in file \'%s\'.',config.netG,epoch+1,fpath)
                if config.saveD:
                    fpath = config.sub('saveD',epoch=epoch+1)
                    netD.save_params(fpath)
                    if logger:
                        logger.info('Saved discriminator \'%s\' checkpoint epoch %04f in file \'%s\'.',config.netD,epoch+1,fpath)

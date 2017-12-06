import mxnet as mx
from time import time
from mxnet import gluon, autograd, nd

from .func import fuzzy_one_hot
from .util import Config
from .net import nets

models = {}
def register_model(obj):
    models[obj.__name__] = obj
    return obj

class Model(object):
    def __init__(self,**kwargs):
        self.logger = kwargs.get('logger')
        self.config = kwargs.get('config',Config())
        self.ctx = kwargs.get('ctx')
        if not self.ctx:
            self.ctx = mx.context.Context(config.device, config.device_id)

        self.save_freq = self.config.save_freq
        self.start_epoch = self.config.get('start_epoch',0)

        self.nets = {}
        for key, desc in self.config.nets.items():
            self.nets[key] = nets[desc.type]()
            if desc.get('param'):
                fpath = self.config.sub('nets.%s.param'%(key))
                self.nets[key].load_params(fpath,ctx=self.ctx)
                if self.logger:
                    self.logger.debug('Loading parameters for %s \'%s\' from %s.',key,desc.type,fpath)
            else:
                self.nets[key].initialize(mx.init.Xavier(magnitude=2.24),ctx=self.ctx)
                if self.logger:
                    self.logger.debug('Initializing %s \'%s\'.',key,desc.type)

    def checkpoint(self, epoch):
        for key,tnet in self.nets.items():
            if self.config.nets[key].get('save'):
                fpath = self.config.sub('nets.%s.save'%(key),epoch=epoch)
                tnet.save_params(fpath)
                if self.logger:
                    self.logger.info('Saved %s \'%s\' checkpoint epoch %s in file \'%s\'.',key,self.config.nets[key].type,epoch,fpath)
            else:
                if self.logger:
                    self.logger.debug('Not saving %s \'%s\' epoch %s.',key,self.config.nets[key].type,epoch,fpath)


@register_model
class Classifier(Model):
    def __init__(self,**kwargs):
        super().__init__(**kwargs)
        self.netC = self.nets['classifier']

    def train(self,data,batch_size,nepochs):
        config = self.config

        netC = self.netC
        ctx = self.ctx

        data_iter = gluon.data.DataLoader(data,batch_size,shuffle=True,last_batch='discard')
        loss = gluon.loss.SoftmaxCrossEntropyLoss()
        trainer = gluon.Trainer(netC.collect_params(), 'adam', {'learning_rate': 0.01})
        epoch = self.start_epoch

        metric = mx.metric.Accuracy()

        if self.logger:
            self.logger.info('Starting training of model %s, classifier %s at epoch %d',
                            config.model,config.nets.classifier.type,epoch)

        try:
            for epoch in range(self.start_epoch, self.start_epoch + nepochs):
                tic = time()
                for i, (data, label) in enumerate(data_iter):

                    data = data.as_in_context(ctx).reshape((-1,784))
                    label = label.as_in_context(ctx)

                    with autograd.record():
                        output = netC(data)
                        err = loss(output, label)

                        err.backward()

                    trainer.step(batch_size)
                    metric.update([label,], [output,])

                name, acc = metric.get()
                metric.reset()
                if self.logger:
                    self.logger.info('netD training acc epoch %04d: %s=%.4f , time: %.2f',epoch, name, acc, (time() - tic))
                if ( (self.save_freq > 0) and not ( (epoch + 1) % self.save_freq) ) or  ((epoch + 1) >= (self.start_epoch + nepochs)) :
                    self.checkpoint(epoch+1)
        except KeyboardInterrupt:
            self.logger.info('Training interrupted by user.')
            self.checkpoint('I%d'%epoch)

    def test(self,data,batch_size):
        data_iter = gluon.data.DataLoader(data,batch_size,last_batch='discard')

        metric = mx.metric.Accuracy()
        for i, (data, label) in enumerate(data_iter):
            data = data.as_in_context(self.ctx).reshape((-1,784))
            label = label.as_in_context(self.ctx)

            output = self.netC(data)
            metric.update([label,], [output,])

        name, acc = metric.get()

        if self.logger:
            self.logger.info('%s test acc: %s=%.4f , time: %.2f',self.config.nets.classifier.type, name, acc)

@register_model
class GAN(Model):
    def __init__(self,**kwargs):
        super().__init__(**kwargs)
        self.netG = self.nets['generator']
        self.netD = self.nets['discriminator']

    def train(self,data,batch_size,nepochs):

        config = self.config

        netG = self.netG
        netD = self.netD

        ctx = self.ctx

        data_iter = gluon.data.DataLoader(data,batch_size,shuffle=True,last_batch='discard')

        # loss
        loss = gluon.loss.SoftmaxCrossEntropyLoss(sparse_label=False)

        # trainer for the generator and the discriminator
        trainerG = gluon.Trainer(netG.collect_params(), 'adam', {'learning_rate': 0.01})
        trainerD = gluon.Trainer(netD.collect_params(), 'adam', {'learning_rate': 0.05})

        real_label = nd.ones(batch_size, ctx=ctx)
        fake_label = nd.zeros(batch_size, ctx=ctx)

        real_label_oh = fuzzy_one_hot(real_label, 2)
        fake_label_oh = fuzzy_one_hot(fake_label, 2)

        metric = mx.metric.Accuracy()

        epoch = self.start_epoch

        if self.logger:
            self.logger.info('Starting training of model %s, discriminator %s, generator %s at epoch %d',
                        config.model,config.nets.discriminator.type,config.nets.generator.type,epoch)

        try:
            for epoch in range(self.start_epoch, self.start_epoch + nepochs):
                tic = time()
                # train_data.reset()
                for i, data in enumerate(data_iter):
                    ############################
                    # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
                    ###########################

                    data = data.as_in_context(ctx).reshape((-1,784))

                    noise = nd.random_normal(shape=(data.shape[0], 32), ctx=ctx)

                    with autograd.record():
                        real_output = netD(data)
                        errD_real = loss(real_output, real_label_oh)

                        fake = netG(noise)
                        fake_output = netD(fake.detach())
                        errD_fake = loss(fake_output, fake_label_oh)
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
                        errG = loss(output, real_label_oh)
                        errG.backward()

                    trainerG.step(batch_size)

                name, acc = metric.get()
                metric.reset()
                if self.logger:
                    self.logger.info('netD training acc epoch %04d: %s=%.4f , time: %.2f',epoch, name, acc, (time() - tic))
                if ( (self.save_freq > 0) and not ( (epoch + 1) % self.save_freq) ) or  ((epoch + 1) >= (self.start_epoch + nepochs)) :
                    self.checkpoint(epoch+1)
        except KeyboardInterrupt:
            self.logger.info('Training interrupted by user.')
            self.checkpoint('I%d'%epoch)

@register_model
class CGAN(GAN):
    def train(self,data,batch_size,nepochs):

        config = self.config
        netG = self.netG
        netD = self.netD
        ctx = self.ctx

        data_iter = gluon.data.DataLoader(data,batch_size,shuffle=True,last_batch='discard')

        # loss
        loss = gluon.loss.SoftmaxCrossEntropyLoss(sparse_label=False)

        # trainer for the generator and the discriminator
        trainerG = gluon.Trainer(netG.collect_params(), 'adam', {'learning_rate': 0.01})
        trainerD = gluon.Trainer(netD.collect_params(), 'adam', {'learning_rate': 0.05})

        real_label = nd.ones(batch_size, ctx=ctx)
        fake_label = nd.zeros(batch_size, ctx=ctx)

        real_label_oh = fuzzy_one_hot(real_label, 2)
        fake_label_oh = fuzzy_one_hot(fake_label, 2)
        metric = mx.metric.Accuracy()

        epoch = self.start_epoch

        if self.logger:
            self.logger.info('Starting training of model %s, discriminator %s, generator %s at epoch %d',
                        config.model,config.nets.discriminator.type,config.nets.generator.type,epoch)

        try:
            for epoch in range(self.start_epoch, self.start_epoch + nepochs):
                tic = time()
                # train_data.reset()
                for i, (data,cond_dense) in enumerate(data_iter):
                    ############################
                    # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
                    ###########################

                    data = data.as_in_context(ctx).reshape((-1,784))
                    cond_dense = cond_dense.as_in_context(ctx)
                    cond = nd.one_hot(cond_dense, 10)
                    cond2 = nd.one_hot(cond_dense, 10)
                    cond3 = nd.one_hot(cond_dense, 10)

                    noise = nd.random_normal(shape=(data.shape[0], 32), ctx=ctx)

                    with autograd.record():
                        real_output = netD(data, cond.detach())
                        errD_real = loss(real_output, real_label_oh)

                        fake = netG(noise, cond)
                        fake_output = netD(fake.detach(), cond2.detach())
                        errD_fake = loss(fake_output, fake_label_oh)
                        errD = errD_real + errD_fake
                        errD.backward()

                    trainerD.step(batch_size)
                    metric.update([real_label,], [real_output,])
                    metric.update([fake_label,], [fake_output,])

                    ############################
                    # (2) Update G network: maximize log(D(G(z)))
                    ###########################
                    with autograd.record():
                        output = netD(fake, cond3.detach())
                        errG = loss(output, real_label_oh)
                        errG.backward()

                    trainerG.step(batch_size)

                name, acc = metric.get()
                metric.reset()
                if self.logger:
                    self.logger.info('netD training acc epoch %04d: %s=%.4f , time: %.2f',epoch, name, acc, (time() - tic))
                if ( (self.save_freq > 0) and not ( (epoch + 1) % self.save_freq) ) or  ((epoch + 1) >= (self.start_epoch + nepochs)) :
                    self.checkpoint(epoch+1)
        except KeyboardInterrupt:
            self.logger.info('Training interrupted by user.')
            self.checkpoint('I%d'%epoch)

import mxnet as mx
from time import time
from mxnet import gluon, autograd, nd

from .func import fuzzy_one_hot

models = {}
def register_model(obj):
    models[obj.__name__] = obj
    return obj

@register_model
class Classifier(object):
    def __init__(self,netC,ctx,**kwargs):
        self.logger = kwargs.get('logger')
        self.config = kwargs.get('config',{})

        self.save_freq = self.config.save_freq
        self.start_epoch = self.config.get('start_epoch',0)

        self.netC = netC

        self.ctx = ctx

    def checkpoint(self,epoch):
        if self.config.saveC:
            fpath = self.config.sub('saveC',epoch=epoch)
            self.netC.save_params(fpath)
            if self.logger:
                self.logger.info('Saved classifier \'%s\' checkpoint epoch %s in file \'%s\'.',self.config.netC,epoch,fpath)

    def train(self,data,batch_size,nepochs):
        logger = self.logger
        config = self.config
        save_freq = self.save_freq
        start_epoch = self.start_epoch
        net = self.net
        ctx = self.ctx

        data_iter = gluon.data.DataLoader(data,batch_size,shuffle=True,last_batch='discard')
        loss = gluon.loss.SoftmaxCrossEntropyLoss()
        trainer = gluon.Trainer(net.collect_params(), 'adam', {'learning_rate': 0.01})
        epoch = start_epoch

        metric = mx.metric.Accuracy()

        if logger:
            logger.info('Starting training of model %s, classifier %s at epoch %d',config.model,config.netC,start_epoch)

        try:
            for epoch in range(start_epoch, start_epoch + nepochs):
                tic = time()
                for i, (data, label) in enumerate(data_iter):

                    data = data.as_in_context(ctx).reshape((-1,784))
                    label = label.as_in_context(ctx)

                    with autograd.record():
                        output = net(data)
                        err = loss(output, label)

                        err.backward()

                    trainer.step(batch_size)
                    metric.update([label,], [output,])

                name, acc = metric.get()
                metric.reset()
                if logger:
                    logger.info('netD training acc epoch %04d: %s=%.4f , time: %.2f',epoch, name, acc, (time() - tic))
                if ( (save_freq > 0) and not ( (epoch + 1) % save_freq) ) or  ((epoch + 1) >= (start_epoch + nepochs)) :
                    self.checkpoint(epoch+1)
        except KeyboardInterrupt:
            logger.info('Training interrupted by user.')
            self.checkpoint('I%d'%epoch)

@register_model
class GAN(object):
    def __init__(self,netG,netD,ctx,**kwargs):
        self.logger = kwargs.get('logger')
        self.config = kwargs.get('config',{})

        self.save_freq = self.config.save_freq
        self.start_epoch = self.config.get('start_epoch',0)

        self.netG = netG
        self.netD = netD

        self.ctx = ctx

    def checkpoint(self,epoch):
        if self.config.saveG:
            fpath = self.config.sub('saveG',epoch=epoch)
            self.netG.save_params(fpath)
            if self.logger:
                self.logger.info('Saved generator \'%s\' checkpoint epoch %s in file \'%s\'.',self.config.netG,epoch,fpath)
        if self.config.saveD:
            fpath = self.config.sub('saveD',epoch=epoch)
            self.netD.save_params(fpath)
            if self.logger:
                self.logger.info('Saved discriminator \'%s\' checkpoint epoch %s in file \'%s\'.',self.config.netD,epoch,fpath)

    def train(self,data,batch_size,nepochs):

        logger = self.logger
        config = self.config

        save_freq = self.save_freq
        start_epoch = self.start_epoch

        netG = self.netG
        netD = self.netD

        ctx = self.ctx

        data_iter = gluon.data.DataLoader(data,batch_size,shuffle=True,last_batch='discard')

        # loss
        loss = gluon.loss.SoftmaxCrossEntropyLoss(sparse_labels=False)

        # trainer for the generator and the discriminator
        trainerG = gluon.Trainer(netG.collect_params(), 'adam', {'learning_rate': 0.01})
        trainerD = gluon.Trainer(netD.collect_params(), 'adam', {'learning_rate': 0.05})

        real_label = nd.ones(batch_size, ctx=ctx)
        fake_label = nd.zeros(batch_size, ctx=ctx)

        real_label_oh = fuzzy_one_hot(real_label, 2)
        fake_label_oh = fuzzy_one_hot(fake_label, 2)

        metric = mx.metric.Accuracy()

        epoch = start_epoch

        if logger:
            logger.info('Starting training of model %s, discriminator %s, generator %s at epoch %d',config.model,config.netD,config.netG,start_epoch)

        try:
            for epoch in range(start_epoch, start_epoch + nepochs):
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
                        errG = loss(output, real_label)
                        errG.backward()

                    trainerG.step(batch_size)

                name, acc = metric.get()
                metric.reset()
                if logger:
                    logger.info('netD training acc epoch %04d: %s=%.4f , time: %.2f',epoch, name, acc, (time() - tic))
                if ( (save_freq > 0) and not ( (epoch + 1) % save_freq) ) or  ((epoch + 1) >= (start_epoch + nepochs)) :
                    self.checkpoint(epoch+1)
        except KeyboardInterrupt:
            logger.info('Training interrupted by user.')
            self.checkpoint('I%d'%epoch)

@register_model
class CGAN(GAN):
    def train(self,data,batch_size,nepochs):

        logger = self.logger
        config = self.config

        save_freq = self.save_freq
        start_epoch = self.start_epoch

        netG = self.netG
        netD = self.netD

        ctx = self.ctx

        data_iter = gluon.data.DataLoader(data,batch_size,shuffle=True,last_batch='discard')

        # loss
        loss = gluon.loss.SoftmaxCrossEntropyLoss(sparse_labels=False)

        # trainer for the generator and the discriminator
        trainerG = gluon.Trainer(netG.collect_params(), 'adam', {'learning_rate': 0.01})
        trainerD = gluon.Trainer(netD.collect_params(), 'adam', {'learning_rate': 0.05})

        real_label = nd.ones(batch_size, ctx=ctx)
        fake_label = nd.zeros(batch_size, ctx=ctx)

        real_label_oh = fuzzy_one_hot(real_label, 2)
        fake_label_oh = fuzzy_one_hot(fake_label, 2)
        metric = mx.metric.Accuracy()

        epoch = start_epoch

        if logger:
            logger.info('Starting training of model %s, discriminator %s, generator %s at epoch %d',config.model,config.netD,config.netG,start_epoch)

        try:
            for epoch in range(start_epoch, start_epoch + nepochs):
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
                        errG = loss(output, real_label)
                        errG.backward()

                    trainerG.step(batch_size)

                name, acc = metric.get()
                metric.reset()
                if logger:
                    logger.info('netD training acc epoch %04d: %s=%.4f , time: %.2f',epoch, name, acc, (time() - tic))
                if ( (save_freq > 0) and not ( (epoch + 1) % save_freq) ) or  ((epoch + 1) >= (start_epoch + nepochs)) :
                    self.checkpoint(epoch+1)
        except KeyboardInterrupt:
            logger.info('Training interrupted by user.')
            self.checkpoint('I%d'%epoch)

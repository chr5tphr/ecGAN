import mxnet as mx
import numpy as np
from time import time
from mxnet import gluon, autograd, nd
from imageio import imwrite

from .func import fuzzy_one_hot, linspace, randint
from .explain.base import Interpretable
from .pattern.base import PatternNet
from .layer import Intermediate, YSequential
from .util import Config, config_ctx
from .net import nets

models = {}
def register_model(obj):
    models[obj.__name__] = obj
    return obj

class TrainingRoutine(object):
    def __init__(self):
        self.data_iter = None
        self.trainers = []
        self.batch_size = None
        self.start_epoch = None
        self.nepochs = None
        self.save_freq = None
        self.logger = None

    def __call__(data):
        for epoch in range(start_epoch, start_epoch + nepochs):
            try:
                for i, (data, label) in enumerate(data_iter):
                    self.step(data)
                    for trainer in self.trainers:
                        trainer.step(self.batch_size)

                if ( (self.save_freq > 0) and not ( (epoch + 1) % self.save_freq) ) or ((epoch + 1) >= (self.start_epoch + nepochs)) :
                    self.save(epoch+1)
            except KeyboardInterrupt:
                self.save(epoch+1, intr=True)


class Model(object):
    def __init__(self, **kwargs):
        self.logger = kwargs.get('logger')
        self.config = kwargs.get('config', Config())
        self.ctx = kwargs.get('ctx')
        if not self.ctx:
            self.ctx = config_ctx(self.config)

        self.save_freq = self.config.save_freq
        self.gen_freq = self.config.gen_freq
        self.start_epoch = self.config.get('start_epoch', 0)
        self.data_bbox = self.config.data.bbox

        self.nets = {}
        for key, desc in self.config.nets.items():
            self.nets[key] = nets[desc.type](**(desc.get('kwargs', {})))
            if not self.config.init and desc.get('param'):
                fpath = self.config.sub('nets.%s.param'%(key))
                self.nets[key].load_params(fpath, ctx=self.ctx)
                if self.logger:
                    self.logger.debug('Loading parameters for %s \'%s\' from %s.', key, desc.type, fpath)
            else:
                self.nets[key].initialize(mx.init.Xavier(rnd_type='gaussian', magnitude=2.24), ctx=self.ctx)
                if self.logger:
                    self.logger.debug('Initializing %s \'%s\'.', key, desc.type)

    def checkpoint(self, epoch):
        for key, tnet in self.nets.items():
            if self.config.nets[key].get('save'):
                fpath = self.config.sub('nets.%s.save'%(key), epoch=epoch)
                tnet.save_params(fpath)
                if self.logger:
                    self.logger.info('Saved %s \'%s\' checkpoint epoch %s in file \'%s\'.', key, self.config.nets[key].type, epoch, fpath)
            else:
                if self.logger:
                    fpath = self.config.sub('nets.%s.save'%(key), epoch=epoch)
                    self.logger.debug('Not saving %s \'%s\' epoch %s.', key, self.config.nets[key].type, epoch, fpath)

    def load_pattern_params(self):
        for nrole, net in self.nets.items():
            net.init_pattern()
            nname = self.config.nets.get(nrole, {}).get('name', '')
            ntype = self.config.nets.get(nrole, {}).get('type', '')
            if self.config.pattern.get('load') is not None and not self.config.pattern.get('init', False):
                net.collect_pparams().load(self.config.sub('pattern.load', net_name=nname, net_type=ntype), ctx=self.ctx)
            else:
                net.collect_pparams().initialize(ctx=self.ctx)

    def save_pattern_params(self, epoch):
        if self.config.pattern.get('save') is not None:
            for nrole, net in self.nets.items():
                nname = self.config.nets.get(nrole, {}).get('name', '')
                ntype = self.config.nets.get(nrole, {}).get('type', '')
                fpath = self.config.sub('pattern.save', net_name=nname,
                                        net_type=ntype, epoch=epoch)
                net.collect_pparams().save(fpath)
                if self.logger:
                    self.logger.info('Saved pattern of \'%s %s\' epoch %s in file \'%s\'.', ntype, nname, epoch, fpath)
        else:
            if self.logger:
                self.logger.debug('Not saving pattern.')

    def generate_sample(self, epoch):
        raise NotImplementedError('Not supported for class %s'%self.__class__)

    def explain(self, data, method):
        raise NotImplementedError('Not supported for class %s'%self.__class__)

    def test(self, data, batch_size):
        raise NotImplementedError('Not supported for class %s'%self.__class__)

@register_model
class Classifier(Model):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.netC = self.nets['classifier']

    def train(self, data, batch_size, nepochs):
        config = self.config

        netC = self.netC
        ctx = self.ctx

        data_iter = gluon.data.DataLoader(data, batch_size, shuffle=True, last_batch='discard')
        loss = gluon.loss.SoftmaxCrossEntropyLoss()
        trainer = gluon.Trainer(netC.collect_params(), 'adam', {'learning_rate': 0.01})
        epoch = self.start_epoch

        metric = mx.metric.Accuracy()

        if self.logger:
            self.logger.info('Starting training of model %s, classifier %s at epoch %d',
                            config.model, config.nets.classifier.type, epoch)

        try:
            for epoch in range(self.start_epoch, self.start_epoch + nepochs):
                tic = time()
                for i, (data, label) in enumerate(data_iter):

                    data = data.as_in_context(ctx)
                    label = label.as_in_context(ctx)

                    with autograd.record():
                        output = netC(data)
                        err = loss(output, label)

                        err.backward()

                    trainer.step(batch_size)
                    metric.update([label, ], [output, ])

                name, acc = metric.get()
                metric.reset()
                if self.logger:
                    self.logger.info('netD training acc epoch %04d: %s=%.4f , time: %.2f', epoch, name, acc, (time() - tic))
                if ( (self.save_freq > 0) and not ( (epoch + 1) % self.save_freq) ) or  ((epoch + 1) >= (self.start_epoch + nepochs)) :
                    self.checkpoint(epoch+1)
        except KeyboardInterrupt:
            self.logger.info('Training interrupted by user.')
            self.checkpoint('I%d'%epoch)

    def test(self, data, batch_size):
        data_iter = gluon.data.DataLoader(data, batch_size, last_batch='discard')

        metric = mx.metric.Accuracy()
        for i, (data, label) in enumerate(data_iter):
            data = data.as_in_context(self.ctx).reshape((-1, 784))
            label = label.as_in_context(self.ctx)

            output = self.netC(data)
            metric.update([label, ], [output, ])

        name, acc = metric.get()

        if self.logger:
            self.logger.info('%s test acc: %s=%.4f', self.config.nets.classifier.type, name, acc)

    def explain(self, data, label=None):
        method = self.config.explanation.method
        if not isinstance(self.netC, Interpretable):
            raise NotImplementedError('\'%s\' is not yet Interpretable!'%self.config.nets.classifier.type)

        if method == 'sensitivity':
            # loss = gluon.loss.SoftmaxCrossEntropyLoss()
            # output = self.netC(data)
            # output.attach_grad()
            # with autograd.record():
            #     err = loss(output, label)
            # dEdy = autograd.grad(err, output)
            dEdy = nd.ones(300, ctx=self.ctx)
        else:
            dEdy = None

        R = self.netC.relevance(data, dEdy, method=method, ret_all=True)

        if self.config.debug:
            Rsums = []
            for rel in R:
                Rsums.append(rel.sum().asscalar())
            if self.logger:
                self.logger.debug('Explanation sums: %s', ', '.join([str(fl) for fl in Rsums]))

        return R[-1]

    def learn_pattern(self, data, batch_size):
        estimator = self.config.pattern.estimator
        if not isinstance(self.netC, PatternNet):
            raise NotImplementedError('\'%s\' is not a PatterNet!'%self.config.nets.classifier.type)

        data_iter = gluon.data.DataLoader(data, batch_size)
        #self.netC.estimator = estimator

        self.netC.init_pattern()
        self.netC.collect_pparams().initialize(mx.initializer.Zero(), ctx=self.ctx)

        for i, (data, label) in enumerate(data_iter):
            data = data.as_in_context(self.ctx)
            label = label.as_in_context(self.ctx)

            self.netC.forward_logged(data)
            self.netC.learn_pattern()

        self.netC.compute_pattern()
        self.netC.collect_pparams().save(self.config.sub('pattern.save',
                                                    net_name=self.config.nets.classifier.name,
                                                    net_type=self.config.nets.classifier.type))

        if self.logger:
            self.logger.info('Learned signal estimator %s for net %s', estimator, self.config.nets.classifier.type)

    def fit_pattern(self, data, batch_size):
        if not isinstance(self.netC, PatternNet):
            raise NotImplementedError('\'%s\' is not a PatternNet!'%self.config.nets.classifier.type)

        data_iter = gluon.data.DataLoader(data, batch_size)

        trainer = gluon.Trainer(self.netC.collect_pparams(),
            self.config.pattern.get('optimizer', 'SGD'),
            self.config.pattern.get('optkwargs', {'learning_rate': 0.01}))

        start_epoch = self.config.pattern.get('start_epoch', 0)
        nepochs = self.config.pattern.get('nepochs', 1)
        save_freq = self.config.pattern.get('save_freq', 1)

        for epoch in range(start_epoch, start_epoch + nepochs):
            tic = time()
            for i, (data, label) in enumerate(data_iter):
                data = data.as_in_context(self.ctx)
                label = label.as_in_context(self.ctx)

                self.netC.fit_pattern(data)
                trainer.step(batch_size, ignore_stale_grad=True)

            if self.logger:
                self.logger.info('pattern training epoch %04d , time: %.2f', epoch, (time() - tic))
            if ( (save_freq > 0) and not ( (epoch + 1) % save_freq) ) or  ((epoch + 1) >= (start_epoch + nepochs)) :
                self.save_pattern_params(epoch+1)

        if self.logger:
            self.logger.info('Learned pattern for net %s',
                             self.config.nets.classifier.type)

    def explain_pattern(self, data):
        if not isinstance(self.netC, PatternNet):
            raise NotImplementedError('\'%s\' is not yet Interpretable!'%self.config.nets.classifier.type)

        R = self.netC.explain_pattern(data)

        return R

@register_model
class Regressor(Model):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.netR = self.nets['regressor']

    def train(self, data, batch_size, nepochs):
        config = self.config

        netR = self.netR
        ctx = self.ctx

        data_iter = gluon.data.DataLoader(data, batch_size, shuffle=True, last_batch='discard')
        loss = gluon.loss.L2Loss()
        trainer = gluon.Trainer(netR.collect_params(), 'adam', {'learning_rate': 0.01})
        epoch = self.start_epoch

        metric = mx.metric.Loss()

        if self.logger:
            self.logger.info('Starting training of model %s, regressor %s at epoch %d',
                            config.model, config.nets.regressor.type, epoch)

        try:
            for epoch in range(self.start_epoch, self.start_epoch + nepochs):
                tic = time()
                for i, (data, label) in enumerate(data_iter):

                    data = data.as_in_context(ctx)
                    label = label.as_in_context(ctx)

                    with autograd.record():
                        output = netR(data)
                        err = loss(output, label)

                        err.backward()

                    trainer.step(batch_size)
                    metric.update([label, ], [output, ])

                name, acc = metric.get()
                metric.reset()
                if self.logger:
                    self.logger.info('netR training acc epoch %04d: %s=%.4f , time: %.2f', epoch, name, acc, (time() - tic))
                if ( (self.save_freq > 0) and not ( (epoch + 1) % self.save_freq) ) or  ((epoch + 1) >= (self.start_epoch + nepochs)) :
                    self.checkpoint(epoch+1)
        except KeyboardInterrupt:
            self.logger.info('Training interrupted by user.')
            self.checkpoint('I%d'%epoch)

    def test(self, data, batch_size):
        data_iter = gluon.data.DataLoader(data, batch_size, last_batch='discard')

        metric = mx.metric.Loss()
        for i, (data, label) in enumerate(data_iter):
            data = data.as_in_context(self.ctx)
            label = label.as_in_context(self.ctx)

            output = self.netR(data)
            metric.update([label, ], [output, ])

        name, acc = metric.get()

        if self.logger:
            self.logger.info('%s test acc: %s=%.4f', self.config.nets.regressor.type, name, acc)

    def learn_pattern(self, data, batch_size):
        estimator = self.config.pattern.estimator
        if not isinstance(self.netR, PatternNet):
            raise NotImplementedError('\'%s\' is not a PatterNet!'%self.config.nets.regressor.type)

        data_iter = gluon.data.DataLoader(data, batch_size)
        #self.netR.estimator = estimator

        self.netR.init_pattern()
        self.netR.collect_pparams().initialize(ctx=self.ctx)

        for i, (data, label) in enumerate(data_iter):
            data = data.as_in_context(self.ctx)
            label = label.as_in_context(self.ctx)

            self.netR.forward_logged(data)
            self.netR.learn_pattern()

        self.netR.compute_pattern()
        self.netR.collect_pparams().save(self.config.sub('pattern.save',
                                                    net_name=self.config.nets.regressor.name,
                                                    net_type=self.config.nets.regressor.type))

        if self.logger:
            self.logger.info('Learned signal estimator %s for net %s', estimator, self.config.nets.regressor.type)

    def fit_pattern(self, data, batch_size):
        if not isinstance(self.netR, PatternNet):
            raise NotImplementedError('\'%s\' is not a PatterNet!'%self.config.nets.regressor.type)

        data_iter = gluon.data.DataLoader(data, batch_size)

        trainer = gluon.Trainer(self.netR.collect_pparams(),
            self.config.pattern.get('optimizer', 'SGD'),
            self.config.pattern.get('optkwargs', {'learning_rate': 0.01}))

        start_epoch = self.config.pattern.get('start_epoch', 0)
        nepochs = self.config.pattern.get('nepochs', 1)
        save_freq = self.config.pattern.get('save_freq', 1)

        for epoch in range(start_epoch, start_epoch + nepochs):
            tic = time()
            for i, (data, label) in enumerate(data_iter):
                data = data.as_in_context(self.ctx)
                label = label.as_in_context(self.ctx)

                self.netR.fit_pattern(data)
                trainer.step(batch_size, ignore_stale_grad=True)

            if self.logger:
                self.logger.info('pattern training epoch %04d , time: %.2f', epoch, (time() - tic))
            if ( (save_freq > 0) and not ( (epoch + 1) % save_freq) ) or  ((epoch + 1) >= (start_epoch + nepochs)) :
                self.save_pattern_params(epoch+1)

        if self.logger:
            self.logger.info('Learned pattern for net %s',
                             self.config.nets.regressor.type)

    def explain_pattern(self, data):
        if not isinstance(self.netR, PatternNet):
            raise NotImplementedError('\'%s\' is not yet Interpretable!'%self.config.nets.regressor.type)

        R = self.netR.explain_pattern(data)

        return R

@register_model
class GAN(Model):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.netG = self.nets['generator']
        self.netD = self.nets['discriminator']

    def train(self, data, batch_size, nepochs):

        config = self.config

        netG = self.netG
        netD = self.netD

        ctx = self.ctx

        one_hot = fuzzy_one_hot if config.fuzzy_labels else nd.one_hot

        data_iter = gluon.data.DataLoader(data, batch_size, shuffle=True, last_batch='discard')

        # loss
        if config.semi_supervised:
            loss = gluon.loss.SoftmaxCrossEntropyLoss(sparse_label=False)
        else:
            loss = gluon.loss.SigmoidBinaryCrossEntropyLoss()

        # trainer for the generator and the discriminator
        trainerG = gluon.Trainer(netG.collect_params(),
            config.nets.generator.get('optimizer', 'adam'),
            config.nets.generator.get('optkwargs', {'learning_rate': 0.01}))
        trainerD = gluon.Trainer(netD.collect_params(),
            config.nets.discriminator.get('optimizer', 'adam'),
            config.nets.discriminator.get('optkwargs', {'learning_rate': 0.05}))


        if not config.semi_supervised:
            real_dense = nd.ones(batch_size, ctx=ctx)
            fake_dense = nd.zeros(batch_size, ctx=ctx)

            real_label = real_dense
            fake_label = fake_dense

            # real_label = one_hot(real_dense, 2).reshape((batch_size, -1, 1, 1))
            # fake_label = one_hot(fake_dense, 2).reshape((batch_size, -1, 1, 1))

        metric = mx.metric.Accuracy()

        epoch = self.start_epoch
        K = len(data.classes)

        if self.logger:
            self.logger.info('Starting training of model %s, discriminator %s, generator %s at epoch %d',
                        config.model, config.nets.discriminator.type, config.nets.generator.type, epoch)

        try:
            for epoch in range(self.start_epoch, self.start_epoch + nepochs):
                tic = time()
                for i, (data, real_dense) in enumerate(data_iter):
                    ############################
                    # (1) Update D
                    ###########################

                    num = data.shape[0]

                    data = data.as_in_context(ctx)
                    if config.semi_supervised:
                        real_dense = real_dense.as_in_context(ctx).reshape((-1, ))
                        real_label = one_hot(real_dense, 2*K).reshape((num, -1, 1, 1))

                        # classes K to 2K are fake
                        fake_dense = real_dense + K
                        fake_label = one_hot(fake_dense, 2*K).reshape((num, -1, 1, 1))

                    noise = nd.random_normal(shape=(num, 100, 1, 1), ctx=ctx)

                    with autograd.record():
                        real_output = netD(data)
                        errD_real = loss(real_output, real_label)

                        fake = netG(noise)
                        fake_output = netD(fake.detach())
                        errD_fake = loss(fake_output, fake_label)
                        errD = errD_real + errD_fake
                    errD.backward()

                    trainerD.step(batch_size)
                    metric.update([real_dense, ], [real_output, ])
                    metric.update([fake_dense, ], [fake_output, ])

                    ############################
                    # (2) Update G
                    ###########################
                    with autograd.record():
                        if config.feature_matching:
                            # feature matching
                            real_intermed = netD.forward(data, depth=-2)
                            fake_intermed = netD.forward(fake, depth=-2)
                            errG = ((real_intermed - fake_intermed)**2).sum()
                        else:
                            output = netD(fake)
                            errG = loss(output, real_label)
                    errG.backward()

                    trainerG.step(batch_size)

                name, acc = metric.get()
                metric.reset()
                if self.logger:
                    self.logger.info('netD training acc epoch %04d: %s=%.4f , time: %.2f', epoch, name, acc, (time() - tic))
                if ( (self.save_freq > 0) and not ( (epoch + 1) % self.save_freq) ) or  ((epoch + 1) >= (self.start_epoch + nepochs)):
                    self.checkpoint(epoch+1)
                if ( (self.gen_freq > 0) and not ( (epoch + 1) % self.gen_freq) ) or  ((epoch + 1) >= (self.start_epoch + nepochs)):
                    self.generate_sample(epoch+1)
        except KeyboardInterrupt:
            self.logger.info('Training interrupted by user.')
            self.checkpoint('I%d'%epoch)
            self.generate_sample('I%d'%epoch)

    def generate_sample(self, epoch, label=None):
        if self.config.genout:
            bbox = self.data_bbox
            noise = nd.random_normal(shape=(30, 100, 1, 1), ctx=self.ctx)
            gdat = self.netG(*([noise] + ([] if label is None else [label])))
            fpath = self.config.sub('genout', epoch=epoch)
            gdat = ((gdat - bbox[0]) * 255/(bbox[1]-bbox[0])).asnumpy().clip(0, 255).astype(np.uint8)
            imwrite(fpath, gdat.reshape(5, 6, 28, 28).transpose(0, 2, 1, 3).reshape(5*28, 6*28))
            if self.logger:
                # if label is not None:
                #     self.logger.debug('Generated using labels: %s', str(label))
                self.logger.info('Saved data generated by \'%s\' epoch %s in \'%s\'.',
                                 self.config.nets.generator.type, epoch, fpath)

    # def generated_sensitivity(self, epoch):
    #     if self.config.genout:
    #         noise = nd.random_normal(shape=(30, 100), ctx=self.ctx)
    #         gdat = self.netG(noise)
    #         gdat.attach_grad()
    #         with autograd.record():
    #             dis = self.netD(gdat)
    #         dis.backward()
    #         img = draw_heatmap(gdat.grad.asnumpy().reshape(5, 6, 28, 28).transpose(0, 2, 1, 3).reshape(5*28, 6*28))
    #         fpath = self.config.sub('relout', method='sensitivity', epoch=epoch)
    #         imwrite(fpath, img)
    #         if self.logger:
    #             self.logger.info('Saved generated sensitivity by \'%s\' epoch %s in \'%s\'.',
    #                              self.config.nets.generator.type, epoch, fpath)

    def explain(self, data=None, label=None):
        netTop = self.nets.get(self.config.explanation.top_net, self.netD)
        method = self.config.explanation.method

        if not isinstance(netTop, Interpretable):
            raise NotImplementedError('\'%s\' is not yet Interpretable!'%
                (self.config.nets.get(self.config.explanation.top_net, self.config.nets.discriminator).type))

        if data is None:
            noise = nd.random_normal(shape=(30, 100, 1, 1), ctx=self.ctx)
            targs = [noise]
            if (isinstance(self.netG, YSequential)) and (label is not None):
                targs += [label]
            gdata = self.netG.forward_logged(*targs)
        else:
            gdata = data

        targs = [gdata]
        if (isinstance(netTop, YSequential)) and (label is not None):
            targs += [label]
        netTop.forward_logged(*targs)

        if method == 'sensitivity':
            dEdy = nd.ones((30, netTop._outnum), ctx=self.ctx)
        else:
            dEdy = None

        Rtc = [None]
        Rc = [None]

        Rret = netTop.relevance(dEdy, method=method, ret_all=True)
        if isinstance(Rret, tuple):
            R, Rtc = Rret
        else:
            R = Rret
        Rt = R[-1]

        if data is None:
            Rret = self.netG.relevance(R[-1], method=method, ret_all=True)
            if isinstance(Rret, tuple):
                Rg, Rc = Rret
            else:
                Rg = Rret
            R += Rg

        if self.config.debug:
            Rsums = []
            for rel in R:
                Rsums.append(rel.sum().asscalar())
            if self.logger:
                self.logger.debug('Explanation sums: %s', ', '.join([str(fl) for fl in Rsums]))

        if data is None:
            return (Rt, Rtc[-1], R[-1], Rc[-1], noise, gdata)
        else:
            return (R[-1], Rc[-1])

    def predict(self, data=None, label=None):
        netTop = self.nets.get(self.config.explanation.top_net, self.netD)

        if data is None:
            noise = nd.random_normal(shape=(30, 100), ctx=self.ctx)
            gdata = self.netG(*([noise] + ([] if label is None else [label])))

        output = netTop(gdata)
        if output.shape[1] > 1:
            output = output.softmax(axis=1)

        return output

@register_model
class WGAN(GAN):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.ncritic = self.config.get('ncritic', 5)

    def train(self, data, batch_size, nepochs):
        config = self.config

        netG = self.netG
        netD = self.netD

        ctx = self.ctx

        data_iter = gluon.data.DataLoader(data, batch_size, shuffle=True, last_batch='discard')

        # trainer for the generator and the discriminator
        trainerG = gluon.Trainer(netG.collect_params(),
            config.nets.generator.get('optimizer', 'rmsprop'),
            config.nets.generator.get('optkwargs', {'learning_rate': 0.00005}))
        trainerD = gluon.Trainer(netD.collect_params(),
            config.nets.discriminator.get('optimizer', 'rmsprop'),
            config.nets.discriminator.get('optkwargs', {'learning_rate': 0.00005, 'clip_weights': 1e-2}))

        metric = mx.metric.Loss()

        epoch = self.start_epoch

        if self.logger:
            self.logger.info('Starting training of model %s, discriminator %s, generator %s at epoch %d',
                        config.model, config.nets.discriminator.type, config.nets.generator.type, epoch)

        iter_g = 0

        try:
            for epoch in range(self.start_epoch, self.start_epoch + nepochs):
                tic = time()
                for i, data in enumerate(data_iter):
                    ################
                    # (1) Update D
                    ################
                    if type(data) in [tuple, list]:
                        data = data[0]

                    data = data.as_in_context(ctx).reshape((-1, 784))

                    noise = nd.random_normal(shape=(data.shape[0], 100), ctx=ctx)

                    with autograd.record():
                        real_output = netD(data)
                        errD_real = real_output.mean(axis=0)

                    fake = netG(noise)

                    with autograd.record():
                        fake_output = netD(fake.detach())
                        errD_fake = fake_output.mean(axis=0)

                        errD = - (errD_real - errD_fake)
                        errD.backward()

                    trainerD.step(batch_size)
                    # for key, param in paramsD.items():
                        # param.set_data(param.data(ctx=ctx).clip(-0.01, 0.01))
                    metric.update(None, [-errD, ])

                    ################
                    # (2) Update G
                    ################
                    diter = 100 if ((iter_g < 25) or not (iter_g % 500)) else self.ncritic
                    if not (i % diter):
                        noise = nd.random_normal(shape=(data.shape[0], 100), ctx=ctx)
                        with autograd.record():
                            fake = netG(noise)
                            output = netD(fake)
                            errG = -output.mean(axis=0)
                            errG.backward()

                        trainerG.step(batch_size)
                        iter_g += 1

                name, est = metric.get()
                metric.reset()
                if self.logger:
                    self.logger.info('netD training est epoch %04d: %s=%.4f , time: %.2f', epoch, name, est, (time() - tic))
                if ( (self.save_freq > 0) and not ( (epoch + 1) % self.save_freq) ) or  ((epoch + 1) >= (self.start_epoch + nepochs)):
                    self.checkpoint(epoch+1)
                if ( (self.gen_freq > 0) and not ( (epoch + 1) % self.gen_freq) ) or  ((epoch + 1) >= (self.start_epoch + nepochs)):
                    self.generate_sample(epoch+1)
        except KeyboardInterrupt:
            if self.logger:
                self.logger.info('Training interrupted by user.')
            self.checkpoint('I%d'%epoch)
            self.generate_sample('I%d'%epoch)

@register_model
class WCGAN(WGAN):
    def train(self, data, batch_size, nepochs):
        config = self.config

        netG = self.netG
        netD = self.netD

        ctx = self.ctx

        K = len(data.classes)

        data_iter = gluon.data.DataLoader(data, batch_size, shuffle=True, last_batch='discard')

        # trainer for the generator and the discriminator
        trainerG = gluon.Trainer(netG.collect_params(),
            config.nets.generator.get('optimizer', 'rmsprop'),
            config.nets.generator.get('optkwargs', {'learning_rate': 0.00005}))
        trainerD = gluon.Trainer(netD.collect_params(),
            config.nets.discriminator.get('optimizer', 'rmsprop'),
            config.nets.discriminator.get('optkwargs', {'learning_rate': 0.00005, 'clip_weights': 1e-2}))

        metric = mx.metric.Loss()

        one_hot = fuzzy_one_hot if config.fuzzy_labels else nd.one_hot

        epoch = self.start_epoch

        if self.logger:
            self.logger.info('Starting training of model %s, discriminator %s, generator %s at epoch %d',
                        config.model, config.nets.discriminator.type, config.nets.generator.type, epoch)

        iter_g = 0

        try:
            for epoch in range(self.start_epoch, self.start_epoch + nepochs):
                tic = time()
                for i, (data, cond_dense) in enumerate(data_iter):
                    num = data.shape[0]
                    ################
                    # (1) Update D
                    ################

                    data = data.as_in_context(ctx)
                    cond_dense = cond_dense.as_in_context(ctx)
                    # cond_real = fuzzy_one_hot(cond_dense, K)
                    cond_real = one_hot(cond_dense, K)

                    noise = nd.random_normal(shape=(num, 100), ctx=ctx)
                    # cond_fake = fuzzy_one_hot(nd.uniform(high=K, shape=(data.shape[0], ), ctx=ctx).astype('int32'), K)
                    # cond_fake = one_hot(randint(high=K, shape=(num, ), ctx=ctx), K)

                    # cond_wrong_dense = (cond_dense.reshape((-1, )) + nd.uniform(low=1, high=10, shape=(data.shape[0], ), ctx=ctx).astype('int32')) % 10
                    # cond_wrong = fuzzy_one_hot(cond_wrong_dense, 10)

                    with autograd.record():
                        real_output = netD(data, cond_real)
                        errD_real = real_output.mean(axis=0)

                        # wrong_output = netD(data, cond_wrong)
                        # errD_wrong = wrong_output.mean(axis=0)

                    fake = netG(noise, cond_real)

                    with autograd.record():
                        fake_output = netD(fake.detach(), cond_real)
                        errD_fake = fake_output.mean(axis=0)

                        # errD = - (errD_real - 0.5*(errD_fake + errD_wrong))
                        errD = - (errD_real - errD_fake)
                    errD.backward()

                    trainerD.step(num)
                    # for key, param in paramsD.items():
                        # param.set_data(param.data(ctx=ctx).clip(-0.01, 0.01))
                    metric.update(None, [-errD, ])

                    ################
                    # (2) Update G
                    ################
                    diter = 100 if ((iter_g < 25) or not (iter_g % 500)) else self.ncritic
                    if not (i % diter):
                        noise = nd.random_normal(shape=(num, 100), ctx=ctx)
                        # cond_fake = fuzzy_one_hot(nd.uniform(high=K, shape=(data.shape[0], ), ctx=ctx).astype('int32'), K)
                        cond_fake = one_hot(randint(high=K, shape=(num, ), ctx=ctx), K)
                        with autograd.record():
                            fake = netG(noise, cond_fake)
                            output = netD(fake, cond_fake)
                            errG = -output.mean(axis=0)
                            errG.backward()

                        trainerG.step(num)
                        iter_g += 1

                name, est = metric.get()
                metric.reset()
                if self.logger:
                    self.logger.info('netD training est epoch %04d: %s=%.4f , time: %.2f', epoch, name, est, (time() - tic))
                if ( (self.save_freq > 0) and not ( (epoch + 1) % self.save_freq) ) or  ((epoch + 1) >= (self.start_epoch + nepochs)):
                    self.checkpoint(epoch+1)
                if ( (self.gen_freq > 0) and not ( (epoch + 1) % self.gen_freq) ) or  ((epoch + 1) >= (self.start_epoch + nepochs)):
                    cond = one_hot(linspace(0, K, 30, ctx=ctx, dtype='int32'), K)
                    self.generate_sample(epoch+1, cond)
        except KeyboardInterrupt:
            if self.logger:
                self.logger.info('Training interrupted by user.')
            self.checkpoint('I%d'%epoch)
            cond = one_hot(linspace(0, K, 30, ctx=ctx, dtype='int32'), K)
            self.generate_sample('I%d'%epoch, cond)


@register_model
class CCGAN(GAN):
    def train(self, data, batch_size, nepochs):

        config = self.config
        netG = self.netG
        netD = self.netD
        ctx = self.ctx

        if not isinstance(netD, Intermediate):
            raise TypeError('Discriminator is not an Intermediate!')

        K = len(data.classes)

        data_iter = gluon.data.DataLoader(data, batch_size, shuffle=True, last_batch='discard')

        # loss
        loss = gluon.loss.SoftmaxCrossEntropyLoss(sparse_label=False)

        one_hot = fuzzy_one_hot if config.fuzzy_labels else nd.one_hot

        # trainer for the generator and the discriminator
        trainerG = gluon.Trainer(netG.collect_params(),
            config.nets.generator.get('optimizer', 'adam'),
            config.nets.generator.get('optkwargs', {'learning_rate': 0.01}))
        trainerD = gluon.Trainer(netD.collect_params(),
            config.nets.discriminator.get('optimizer', 'adam'),
            config.nets.discriminator.get('optkwargs', {'learning_rate': 0.05}))

        metric = mx.metric.Accuracy()

        if not config.semi_supervised:
            real_dense = nd.ones(batch_size, ctx=ctx)
            fake_dense = nd.zeros(batch_size, ctx=ctx)

            real_label = one_hot(real_dense, 2).reshape((batch_size, -1, 1, 1))
            fake_label = one_hot(fake_dense, 2).reshape((batch_size, -1, 1, 1))

        epoch = self.start_epoch

        if self.logger:
            self.logger.info('Starting training of model %s, discriminator %s, generator %s at epoch %d',
                        config.model, config.nets.discriminator.type, config.nets.generator.type, epoch)

        try:
            for epoch in range(self.start_epoch, self.start_epoch + nepochs):
                tic = time()
                for i, (data, cond_dense) in enumerate(data_iter):

                    num = data.shape[0]

                    data = data.as_in_context(ctx)
                    cond_dense = cond_dense.as_in_context(ctx).reshape((-1, ))
                    cond = one_hot(cond_dense, K).reshape((num, -1, 1, 1))

                    if config.semi_supervised:
                        real_dense = cond_dense
                        real_label = one_hot(cond_dense, 2*K).reshape((num, -1, 1, 1))

                        # classes K to 2K are fake
                        fake_dense = cond_dense + K
                        fake_label = one_hot(fake_dense, 2*K).reshape((num, -1, 1, 1))

                    noise = nd.random_normal(shape=(num, 100, 1, 1), ctx=ctx)

                    ############################
                    # (1) Update D
                    ###########################
                    with autograd.record():
                        real_output = netD(data)
                        errD_real = loss(real_output, real_label)

                        fake = netG(noise, cond)
                        fake_output = netD(fake.detach())
                        errD_fake = loss(fake_output, fake_label)
                        errD = errD_real + errD_fake
                    errD.backward()

                    trainerD.step(batch_size)
                    metric.update([real_dense, ], [real_output, ])
                    metric.update([fake_dense, ], [fake_output, ])

                    ############################
                    # (2) Update G
                    ###########################
                    # real_intermed = netD.forward(data, cond, depth=-2)
                    with autograd.record():
                        # fake = netG(noise, cond)
                        if config.feature_matching:
                            # feature matching
                            real_intermed = netD.forward(data, depth=-2)
                            fake_intermed = netD.forward(fake, depth=-2)
                            errG = ((real_intermed - fake_intermed)**2).sum()
                        else:
                            output = netD(fake)
                            errG = loss(output, real_label)
                    errG.backward()

                    trainerG.step(batch_size)

                name, acc = metric.get()
                metric.reset()
                if self.logger:
                    self.logger.info('netD training acc epoch %04d: %s=%.4f , time: %.2f', epoch, name, acc, (time() - tic))
                if ( (self.save_freq > 0) and not ( (epoch + 1) % self.save_freq) ) or  ((epoch + 1) >= (self.start_epoch + nepochs)):
                    self.checkpoint(epoch+1)
                if ( (self.gen_freq > 0) and not ( (epoch + 1) % self.gen_freq) ) or  ((epoch + 1) >= (self.start_epoch + nepochs)):
                    cond = one_hot(linspace(0, K, 30, ctx=ctx, dtype='int32'), K).reshape((30, K, 1, 1))
                    self.generate_sample(epoch+1, cond)
        except KeyboardInterrupt:
            if self.logger:
                self.logger.info('Training interrupted by user.')
            self.checkpoint('I%d'%epoch)
            cond = one_hot(linspace(0, K, 30, ctx=ctx, dtype='int32'), K).reshape((30, K, 1, 1))
            self.generate_sample('I%d'%epoch, cond)


@register_model
class CSGAN(GAN):
    def train(self, data, batch_size, nepochs):

        config = self.config
        netG = self.netG
        netD = self.netD
        ctx = self.ctx

        if not isinstance(netD, Intermediate):
            raise TypeError('Discriminator is not an Intermediate!')

        K = len(data.classes)

        data_iter = gluon.data.DataLoader(data, batch_size, shuffle=True, last_batch='discard')

        # loss
        loss = gluon.loss.SoftmaxCrossEntropyLoss(sparse_label=False)

        one_hot = fuzzy_one_hot if config.fuzzy_labels else nd.one_hot

        # trainer for the generator and the discriminator
        trainerG = gluon.Trainer(netG.collect_params(),
            config.nets.generator.get('optimizer', 'adam'),
            config.nets.generator.get('optkwargs', {'learning_rate': 0.01}))
        trainerD = gluon.Trainer(netD.collect_params(),
            config.nets.discriminator.get('optimizer', 'adam'),
            config.nets.discriminator.get('optkwargs', {'learning_rate': 0.05}))

        # real_label = nd.ones(batch_size, ctx=ctx)
        # class K (0toK) is fake
        fake_label_dense = nd.ones(batch_size, ctx=ctx)*K

        # real_label_oh = one_hot(real_label, 2)
        fake_label = one_hot(fake_label_dense, K+1)
        metric = mx.metric.Accuracy()

        epoch = self.start_epoch

        if self.logger:
            self.logger.info('Starting training of model %s, discriminator %s, generator %s at epoch %d',
                        config.model, config.nets.discriminator.type, config.nets.generator.type, epoch)

        try:
            for epoch in range(self.start_epoch, self.start_epoch + nepochs):
                tic = time()
                for i, (data, cond_dense) in enumerate(data_iter):

                    num = data.shape[0]

                    data = data.as_in_context(ctx)
                    cond_dense = cond_dense.as_in_context(ctx)
                    cond = one_hot(cond_dense, K)
                    real_label = one_hot(cond_dense, K+1)

                    noise = nd.random_normal(shape=(num, 100), ctx=ctx)

                    ############################
                    # (1) Update D
                    ###########################
                    with autograd.record():
                        real_output = netD(data, cond)
                        errD_real = loss(real_output, real_label)

                        fake = netG(noise, cond)
                        fake_output = netD(fake.detach(), cond)
                        errD_fake = loss(fake_output, fake_label)
                        errD = errD_real + errD_fake
                    errD.backward()

                    trainerD.step(batch_size)
                    metric.update([cond_dense, ], [real_output, ])
                    metric.update([fake_label_dense, ], [fake_output, ])

                    ############################
                    # (2) Update G
                    ###########################
                    # real_intermed = netD.forward(data, cond, depth=-2)
                    with autograd.record():
                        fake = netG(noise, cond)
                        if config.feature_matching:
                            # feature matching
                            real_intermed = netD.forward(data, cond, depth=-2)
                            fake_intermed = netD.forward(fake, cond, depth=-2)
                            errG = ((real_intermed - fake_intermed)**2).sum()
                        else:
                            output = netD(fake, cond)
                            errG = loss(output, real_label)
                    errG.backward()

                    trainerG.step(batch_size)

                name, acc = metric.get()
                metric.reset()
                if self.logger:
                    self.logger.info('netD training acc epoch %04d: %s=%.4f , time: %.2f', epoch, name, acc, (time() - tic))
                if ( (self.save_freq > 0) and not ( (epoch + 1) % self.save_freq) ) or  ((epoch + 1) >= (self.start_epoch + nepochs)):
                    self.checkpoint(epoch+1)
                if ( (self.gen_freq > 0) and not ( (epoch + 1) % self.gen_freq) ) or  ((epoch + 1) >= (self.start_epoch + nepochs)):
                    cond = one_hot(linspace(0, K, 30, ctx=ctx, dtype='int32'), K)
                    self.generate_sample(epoch+1, cond)
        except KeyboardInterrupt:
            if self.logger:
                self.logger.info('Training interrupted by user.')
            self.checkpoint('I%d'%epoch)
            cond = one_hot(linspace(0, K, 30, ctx=ctx, dtype='int32'), K)
            self.generate_sample('I%d'%epoch, cond)

@register_model
class CGAN(GAN):
    def train(self, data, batch_size, nepochs):

        config = self.config
        netG = self.netG
        netD = self.netD
        ctx = self.ctx

        data_iter = gluon.data.DataLoader(data, batch_size, shuffle=True, last_batch='discard')

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
                        config.model, config.nets.discriminator.type, config.nets.generator.type, epoch)

        try:
            for epoch in range(self.start_epoch, self.start_epoch + nepochs):
                tic = time()
                # train_data.reset()
                for i, (data, cond_dense) in enumerate(data_iter):
                    ############################
                    # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
                    ###########################

                    data = data.as_in_context(ctx).reshape((-1, 784))
                    cond_dense = cond_dense.as_in_context(ctx)
                    cond = nd.one_hot(cond_dense, 10)
                    cond2 = nd.one_hot(cond_dense, 10)
                    cond3 = nd.one_hot(cond_dense, 10)

                    noise = nd.random_normal(shape=(data.shape[0], 100), ctx=ctx)

                    with autograd.record():
                        real_output = netD(data, cond.detach())
                        errD_real = loss(real_output, real_label_oh)

                        fake = netG(noise, cond)
                        fake_output = netD(fake.detach(), cond2.detach())
                        errD_fake = loss(fake_output, fake_label_oh)
                        errD = errD_real + errD_fake
                        errD.backward()

                    trainerD.step(batch_size)
                    metric.update([real_label, ], [real_output, ])
                    metric.update([fake_label, ], [fake_output, ])

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
                    self.logger.info('netD training acc epoch %04d: %s=%.4f , time: %.2f', epoch, name, acc, (time() - tic))
                if ( (self.save_freq > 0) and not ( (epoch + 1) % self.save_freq) ) or  ((epoch + 1) >= (self.start_epoch + nepochs)) :
                    self.checkpoint(epoch+1)
        except KeyboardInterrupt:
            self.logger.info('Training interrupted by user.')
            self.checkpoint('I%d'%epoch)

import mxnet as mx
import numpy as np
from time import time
from mxnet import gluon, autograd, nd
from imageio import imwrite
from logging import getLogger

from .func import fuzzy_one_hot, linspace, randint
from .explain.base import Explainable
from .explain.pattern.base import PatternNet
from .layer import Intermediate, Sequential
from .util import Config, config_ctx
from .net import nets
from .data import ArrayDataset
from .plot import save_aligned_image

models = {}
def register_model(obj):
    models[obj.__name__] = obj
    return obj

class Model(object):
    def __init__(self, **kwargs):
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
            if not desc.get('active', True):
                continue
            self.nets[key] = nets[desc.type](**(desc.get('kwargs', {})))
            if not self.config.init and desc.get('param'):
                fpath = self.config.sub('nets.%s.param'%(key))
                self.nets[key].load_params(fpath, ctx=self.ctx)
                getLogger('ecGAN').debug('Loading parameters for %s \'%s\' from %s.', key, desc.type, fpath)
            else:
                self.nets[key].initialize(mx.init.Xavier(rnd_type='gaussian', magnitude=2.24), ctx=self.ctx)
                getLogger('ecGAN').debug('Initializing %s \'%s\'.', key, desc.type)

    def checkpoint(self, epoch):
        for key, tnet in self.nets.items():
            if self.config.nets[key].get('save'):
                fpath = self.config.sub('nets.%s.save'%(key), epoch=epoch)
                tnet.save_params(fpath)
                getLogger('ecGAN').info('Saved %s \'%s\' checkpoint epoch %s in file \'%s\'.', key, self.config.nets[key].type, epoch, fpath)
            else:
                fpath = self.config.sub('nets.%s.save'%(key), epoch=epoch)
                getLogger('ecGAN').debug('Not saving %s \'%s\' epoch %s.', key, self.config.nets[key].type, epoch)

    def load_pattern_params(self):
        for nrole, net in self.nets.items():
            net.init_pattern()
            nname = self.config.nets.get(nrole, {}).get('name', '')
            ntype = self.config.nets.get(nrole, {}).get('type', '')
            pparams = net.collect_pparams()
            if self.config.pattern.get('load') is not None and not self.config.pattern.get('init', False):
                pparams.load(self.config.sub('pattern.load', net_name=nname, net_type=ntype), ctx=self.ctx, ignore_extra=True, restore_prefix=pparams.prefix)
            else:
                pparams.initialize(ctx=self.ctx)

    def save_pattern_params(self, **kwargs):
        if self.config.pattern.get('save') is not None:
            for nrole, net in self.nets.items():
                nname = self.config.nets.get(nrole, {}).get('name', '')
                ntype = self.config.nets.get(nrole, {}).get('type', '')
                fpath = self.config.sub('pattern.save', net_name=nname,
                                        net_type=ntype, **kwargs)
                pparams = net.collect_pparams()
                pparams.save(fpath, pparams.prefix)
                ktxt = ", ".join(['%s=%s'%(str(key), str(val)) for key,val in kwargs.items()])
                getLogger('ecGAN').info('Saved pattern of \'%s %s\' {%s} in file \'%s\'.', ntype, nname, ktxt, fpath)
        else:
            getLogger('ecGAN').debug('Not saving pattern.')

    def save_generated(self, epoch):
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

        data_iter = gluon.data.DataLoader(data, batch_size, shuffle=True)
        loss = gluon.loss.SoftmaxCrossEntropyLoss()
        trainer = gluon.Trainer(netC.collect_params(),
            config.nets.classifier.get('optimizer', 'adam'),
            config.nets.classifier.get('optkwargs', {'learning_rate': 0.01}))
        epoch = self.start_epoch

        metric = mx.metric.Accuracy()

        getLogger('ecGAN').info('Starting training of model %s, classifier %s at epoch %d',
                            config.model, config.nets.classifier.type, epoch)

        try:
            for epoch in range(self.start_epoch, self.start_epoch + nepochs):
                tic = time()
                for i, (data, label) in enumerate(data_iter):

                    data = data.as_in_context(ctx)
                    label = label.as_in_context(ctx)

                    num = data.shape[0]

                    with autograd.record():
                        output = netC(data)
                        err = loss(output, label)

                        err.backward()

                    trainer.step(num)
                    metric.update([label, ], [output, ])

                name, acc = metric.get()
                metric.reset()
                getLogger('ecGAN').info('netD training acc epoch %04d: %s=%.4f , time: %.2f', epoch, name, acc, (time() - tic))
                if ( (self.save_freq > 0) and not ( (epoch + 1) % self.save_freq) ) or  ((epoch + 1) >= (self.start_epoch + nepochs)) :
                    self.checkpoint(epoch+1)
        except KeyboardInterrupt:
            getLogger('ecGAN').info('Training interrupted by user.')
            self.checkpoint('I%d'%epoch)

    def test(self, data, batch_size):
        data_iter = gluon.data.DataLoader(data, batch_size)

        metric = mx.metric.Accuracy()
        for i, (data, label) in enumerate(data_iter):
            data = data.as_in_context(self.ctx)
            label = label.as_in_context(self.ctx)

            output = self.netC(data)
            metric.update([label, ], [output, ])

        return metric.get()

    def explain(self, data, label=None, mkwargs={}):
        if not isinstance(self.netC, Explainable):
            raise NotImplementedError('\'%s\' is not yet Explainable!'%self.config.nets.classifier.type)

        R = self.netC.relevance(data=data, out=None, **mkwargs)

#        if self.config.debug:
#            Rsums = []
#            for rel in R:
#                Rsums.append(rel.sum().asscalar())
#            getLogger('ecGAN').debug('Explanation sums: %s', ', '.join([str(fl) for fl in Rsums]))

        return R

    def learn_pattern(self, data, batch_size):
        if not isinstance(self.netC, PatternNet):
            raise NotImplementedError('\'%s\' is not a PatterNet!'%self.config.nets.classifier.type)

        data_iter = gluon.data.DataLoader(data, batch_size)

        for i, (data, label) in enumerate(data_iter):
            data = data.as_in_context(self.ctx)
            label = label.as_in_context(self.ctx)

            self.netC.forward_logged(data)
            self.netC.learn_pattern()

        self.netC.compute_pattern()
        self.save_pattern_params(fit_epoch=self.config.pattern.get('start_epoch', 0),
                                 ase_epoch=self.config.pattern.get('aepoch', 0))

        getLogger('ecGAN').info('Learned pattern for net %s', self.config.nets.classifier.type)

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

            etxt = ', '.join([" & ".join(['%.2e'%(err.mean().asscalar()) for err in toperr if isinstance(err, nd.NDArray)])
                              for toperr in self.netC._err if isinstance(toperr, list)])
            getLogger('ecGAN').info('pattern training epoch %04d , time: %.2f, errors: %s',
                             epoch, (time() - tic), etxt)
            if ( (save_freq > 0) and not ( (epoch + 1) % save_freq) ) or  ((epoch + 1) >= (start_epoch + nepochs)) :
                self.save_pattern_params(fit_epoch=epoch+1,
                                         ase_epoch=self.config.pattern.get('aepoch', 0))

        getLogger('ecGAN').info('Learned pattern for net %s',
                             self.config.nets.classifier.type)

    def fit_assess_pattern(self, data, batch_size):
        if not isinstance(self.netC, PatternNet):
            raise NotImplementedError('\'%s\' is not a PatternNet!'%self.config.nets.classifier.type)

        data_iter = gluon.data.DataLoader(data, batch_size)

        trainer = gluon.Trainer(self.netC.collect_pparams(),
            self.config.pattern.get('optimizer', 'SGD'),
            self.config.pattern.get('optkwargs', {'learning_rate': 0.01}))

        start_epoch = self.config.pattern.get('aepoch', 0)
        nepochs = self.config.pattern.get('nepochs', 1)
        save_freq = self.config.pattern.get('save_freq', 1)

        for epoch in range(start_epoch, start_epoch + nepochs):
            tic = time()
            for i, (data, label) in enumerate(data_iter):
                data = data.as_in_context(self.ctx)
                label = label.as_in_context(self.ctx)

                self.netC.fit_assess_pattern(data)
                trainer.step(batch_size, ignore_stale_grad=True)

            etxt = ', '.join(['%.2e'%(err.mean().asscalar()) for err in self.netC._err if isinstance(err, nd.NDArray)])
            getLogger('ecGAN').info('pattern assessment training epoch %04d , time: %.2f, errors: %s',
                             epoch, (time() - tic), etxt)
            if ( (save_freq > 0) and not ( (epoch + 1) % save_freq) ) or  ((epoch + 1) >= (start_epoch + nepochs)) :
                self.save_pattern_params(fit_epoch=self.config.pattern.get('start_epoch', 0),
                                         ase_epoch=epoch+1)

    def stats_assess_pattern(self, data, batch_size):
        if not isinstance(self.netC, PatternNet):
            raise NotImplementedError('\'%s\' is not a PatterNet!'%self.config.nets.classifier.type)

        data_iter = gluon.data.DataLoader(data, batch_size)

        for i, (data, label) in enumerate(data_iter):
            data = data.as_in_context(self.ctx)
            label = label.as_in_context(self.ctx)

            self.netC.forward_logged(data)
            self.netC.stats_assess_pattern()

        self.save_pattern_params(fit_epoch=self.config.pattern.get('start_epoch', 0),
                                 ase_epoch=self.config.pattern.get('aepoch', 0))

    def explain_pattern(self, data, attribution=False):
        if not isinstance(self.netC, PatternNet):
            raise NotImplementedError('\'%s\' is not yet Explainable!'%self.config.nets.classifier.type)

        R = self.netC.explain_pattern(data, attribution=attribution)

        return R

    def assess_pattern(self):
        if not isinstance(self.netC, PatternNet):
            raise NotImplementedError('\'%s\' is not yet Explainable!'%self.config.nets.classifier.type)

        R = self.netC.assess_pattern()

        return R

    def backward_pattern(self, data):
        if not isinstance(self.netC, PatternNet):
            raise NotImplementedError('\'%s\' is no PatternNet!'%self.config.nets.classifier.type)

        y = self.netC.forward_logged(data)
        R = self.netC.backward_pattern(y)

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

        getLogger('ecGAN').info('Starting training of model %s, regressor %s at epoch %d',
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
                getLogger('ecGAN').info('netR training acc epoch %04d: %s=%.4f , time: %.2f', epoch, name, acc, (time() - tic))
                if ( (self.save_freq > 0) and not ( (epoch + 1) % self.save_freq) ) or  ((epoch + 1) >= (self.start_epoch + nepochs)) :
                    self.checkpoint(epoch+1)
        except KeyboardInterrupt:
            getLogger('ecGAN').info('Training interrupted by user.')
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

        getLogger('ecGAN').info('%s test acc: %s=%.4f', self.config.nets.regressor.type, name, acc)

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

        getLogger('ecGAN').info('Learned signal estimator %s for net %s', estimator, self.config.nets.regressor.type)

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

            getLogger('ecGAN').info('pattern training epoch %04d , time: %.2f', epoch, (time() - tic))
            if ( (save_freq > 0) and not ( (epoch + 1) % save_freq) ) or  ((epoch + 1) >= (start_epoch + nepochs)) :
                self.save_pattern_params(epoch=epoch+1)

        getLogger('ecGAN').info('Learned pattern for net %s',
                             self.config.nets.regressor.type)

    def explain_pattern(self, data):
        if not isinstance(self.netR, PatternNet):
            raise NotImplementedError('\'%s\' is not yet Explainable!'%self.config.nets.regressor.type)

        R = self.netR.explain_pattern(data)

        return R

@register_model
class GAN(Model):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.netG = self.nets['generator']
        self.netD = self.nets[self.config.nets.generator.get('top', 'discriminator')]

    def train(self, data, batch_size, nepochs):

        config = self.config

        netG = self.netG
        netD = self.netD

        ctx = self.ctx

        one_hot = fuzzy_one_hot if config.fuzzy_labels else nd.one_hot

        data_iter = gluon.data.DataLoader(data, batch_size, shuffle=True)

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

        getLogger('ecGAN').info('Starting training of model %s, discriminator %s, generator %s at epoch %d',
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

                    trainerD.step(num)
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

                    trainerG.step(num)

                name, acc = metric.get()
                metric.reset()
                getLogger('ecGAN').info('netD training acc epoch %04d: %s=%.4f , time: %.2f', epoch, name, acc, (time() - tic))
                if ( (self.save_freq > 0) and not ( (epoch + 1) % self.save_freq) ) or  ((epoch + 1) >= (self.start_epoch + nepochs)):
                    self.checkpoint(epoch+1)
                if ( (self.gen_freq > 0) and not ( (epoch + 1) % self.gen_freq) ) or  ((epoch + 1) >= (self.start_epoch + nepochs)):
                    self.save_generated(epoch+1)
        except KeyboardInterrupt:
            getLogger('ecGAN').info('Training interrupted by user.')
            self.checkpoint('I%d'%epoch)
            self.save_generated('I%d'%epoch)

    def save_generated(self, epoch):
        if self.config.genout:
            gen_n = '%s<%s>'%tuple([self.config.nets.generator.get(nam, '') for nam in ['name', 'type']])
            fpath = self.config.sub('genout', net_epoch=epoch, data_desc='trainlog', iter=0, ftype='png', net=self.config.nets.generator.name)
            save_aligned_image(data=self.generate(num=30),
                               fpath=fpath,
                               bbox=self.data_bbox,
                               what='%s(…) epoch %s'%(gen_n, epoch))

    def generate(self, noise=None, num=None):
        if num is None:
            if noise is not None:
                num = len(noise)
            else:
                raise RuntimeError('Either num or noise have to be supplied!')
        if noise is None:
            noise = nd.random_normal(shape=(num, 100, 1, 1), ctx=ctx)

        return netG(noise)

    def explain(self, K, noise=None, num=None, mkwargs={}, ctx=None):
        if not all([isinstance(net, Explainable) for net in [self.netD, self.netG]]):
            raise NotImplementedError('At least one net is not an Explainable!')

        if num is None:
            if noise is not None:
                num = len(noise)
            else:
                raise RuntimeError('At least one arg has to be supplied!')
        if noise is None:
            noise = nd.random_normal(shape=(num, 100, 1, 1), ctx=ctx)

        net = Sequential()
        net.add(self.netG, self.netD)
        Rn, Rc = net.relevance(data=noise, out=None, **mkwargs)

        return Rn, Rc

    def predict(self, data):
        net = Sequential()
        with net.name_scope():
            net.add(self.netG, self.netD)

        return net(data).argmax(axis=1)

@register_model
class CGAN(GAN):
    def train(self, data, batch_size, nepochs):
        config = self.config
        netG = self.netG
        netD = self.netD
        ctx = self.ctx

        if not isinstance(netD, Intermediate):
            raise TypeError('Discriminator is not an Intermediate!')

        K = len(data.classes)

        data_iter = gluon.data.DataLoader(data, batch_size, shuffle=True)

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

        getLogger('ecGAN').info('Starting training of model %s, discriminator %s, generator %s at epoch %d',
                        config.model, config.nets.discriminator.type, config.nets.generator.type, epoch)

        # iter_g = 0
        # ncritic = self.config.get('ncritic', 5)

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

                    # === discriminator ===
                    with autograd.record():
                        real_output = netD(data)
                        errD_real = loss(real_output, real_label)

                        fake = netG([noise, cond])
                        if config.get('clip_penalty', None) is not None:
                            lo, hi = config.nets.generator.kwargs.get('clip', [-1., 1.])
                            fake_nb = fake
                            fake = nd.clip(fake, lo, hi)
                        fake_output = netD(fake.detach())
                        errD_fake = loss(fake_output, fake_label)
                        errD = errD_real + errD_fake
                    errD.backward()

                    trainerD.step(num)
                    metric.update([real_dense, ], [real_output, ])
                    metric.update([fake_dense, ], [fake_output, ])

                    # === generator ===
                    # real_intermed = netD.forward(data, cond, depth=-2)
                    # diter = 100 if ((iter_g < 25) or not (iter_g % 500)) else ncritic
                    # if not (i % diter):
                    with autograd.record():
                        # fake = netG(noise, cond)
                        if config.feature_matching:
                            # feature matching
                            real_intermed = netD.forward(data, depth=-2)
                            fake_intermed = netD.forward(fake, depth=-2)
                            errG = ((real_intermed - fake_intermed)**2).flatten().sum(axis=1)
                        else:
                            output = netD(fake)
                            errG = loss(output, real_label)
                        if config.get('clip_penalty', None) is not None:
                            cp_gamma = config.get('clip_penalty', 1.)
                            clip_penalty = ((fake_nb - fake)**2).flatten().sum(axis=1)
                            errG = errG + cp_gamma*clip_penalty
                    errG.backward()

                    trainerG.step(num)

                name, acc = metric.get()
                metric.reset()
                getLogger('ecGAN').info('netD training acc epoch %04d: %s=%.4f , time: %.2f', epoch, name, acc, (time() - tic))
                if ( (self.save_freq > 0) and not ( (epoch + 1) % self.save_freq) ) or  ((epoch + 1) >= (self.start_epoch + nepochs)):
                    self.checkpoint(epoch+1)
                if ( (self.gen_freq > 0) and not ( (epoch + 1) % self.gen_freq) ) or  ((epoch + 1) >= (self.start_epoch + nepochs)):
                    cond = one_hot(linspace(0, K, 30, ctx=ctx, dtype='int32'), K).reshape((30, K, 1, 1))
                    self.save_generated(epoch+1, cond)
        except KeyboardInterrupt:
            getLogger('ecGAN').info('Training interrupted by user.')
            self.checkpoint('I%d'%epoch)
            cond = one_hot(linspace(0, K, 30, ctx=ctx, dtype='int32'), K).reshape((30, K, 1, 1))
            self.save_generated('I%d'%epoch, cond)

    def test(self, K, num, batch_size=64):
        noise = nd.random_normal(shape=(num, 100, 1, 1), ctx=self.ctx)
        cond = nd.random.uniform(0, K, shape=num, ctx=self.ctx).floor()

        dset = ArrayDataset(noise, cond)

        data_iter = gluon.data.DataLoader(dset, batch_size)

        net = Sequential()
        with net.name_scope():
            net.add(self.netG, self.netD)

        #confusion = nd.zeroes((K,K), ctx=self.ctx)

        metric = mx.metric.Accuracy()
        for i, (data, label) in enumerate(data_iter):
            data = data.as_in_context(self.ctx)
            label = label.as_in_context(self.ctx)
            label_oh = nd.one_hot(label, K).reshape((label.shape[0], -1, 1, 1))

            output = net([data, label_oh])
            metric.update([label, ], [output, ])
            #confusion += nd.one_hot(label, K) == nd.one_hot(nd.argmax(output, axis=1), K)

        return metric.get()

    def generate(self, K=None, noise=None, cond=None, num=None):
        if num is None:
            if noise is not None:
                num = len(noise)
            elif cond is not None:
                num = len(cond)
            else:
                raise RuntimeError('Either num or one of either noise or cond have to be supplied!')
        if K is None:
            if cond is not None:
                K = cond.shape[0]
            else:
                raise RuntimeError('Either number of classes or labels have to be supplied!')
        if noise is None:
            noise = nd.random_normal(shape=(num, 100, 1, 1), ctx=self.ctx)
        if cond is None:
            cond = nd.random.uniform(0, K, shape=num, ctx=self.ctx).floor()
            cond = nd.one_hot(cond, K).reshape((num, -1, 1, 1))

        return self.netG([noise, cond])

    def save_generated(self, epoch, cond=None):
        if self.config.genout:
            gen_n = '%s<%s>'%tuple([self.config.nets.generator.get(nam, '') for nam in ['name', 'type']])
            fpath = self.config.sub('genout', net_epoch=epoch, data_desc='trainlog', iter=0, ftype='png', net=self.config.nets.generator.name)
            save_aligned_image(data=self.generate(num=30, cond=cond),
                               fpath=fpath,
                               bbox=self.data_bbox,
                               what='%s(…) epoch %s'%(gen_n, epoch))

    def fit_pattern(self, data, batch_size):
        if not all([isinstance(net, PatternNet) for net in [self.netD, self.netG]]):
            raise NotImplementedError('At least one net is not a PatternNet!')

        data_iter = gluon.data.DataLoader(data, batch_size)
        K = len(data.classes)
        one_hot = fuzzy_one_hot if self.config.fuzzy_labels else nd.one_hot

        trainerD = gluon.Trainer(self.netD.collect_pparams(),
            self.config.pattern.get('optimizer', 'SGD'),
            self.config.pattern.get('optkwargs', {'learning_rate': 0.01}))
        trainerG = gluon.Trainer(self.netG.collect_pparams(),
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

                num = data.shape[0]

                # === Data Pass ===
                noise = nd.random_normal(shape=(num, 100, 1, 1), ctx=self.ctx)
                cond = one_hot(label, K).reshape((num, -1, 1, 1))

                self.netG.fit_pattern([noise, cond])
                self.netD.fit_pattern(data)

                trainerD.step(num, ignore_stale_grad=True)
                trainerG.step(num, ignore_stale_grad=True)

                # === Fake Pass ===
                noise = nd.random_normal(shape=(num, 100, 1, 1), ctx=self.ctx)
                rand_cond = nd.random.uniform(0,K,shape=num, ctx=self.ctx).floor()
                cond = one_hot(rand_cond, K).reshape((num, -1, 1, 1))

                fake = self.netG.fit_pattern([noise, cond])
                self.netD.fit_pattern(fake)

                trainerD.step(num, ignore_stale_grad=True)
                trainerG.step(num, ignore_stale_grad=True)

            etxtD = ', '.join([" & ".join(['%.2e'%(err.mean().asscalar()) for err in toperr if isinstance(err, nd.NDArray)])
                              for toperr in self.netD._err if isinstance(toperr, list)])
            etxtG = ', '.join([" & ".join(['%.2e'%(err.mean().asscalar()) for err in toperr if isinstance(err, nd.NDArray)])
                              for toperr in self.netG._err if isinstance(toperr, list)])
            getLogger('ecGAN').info('pattern training epoch %04d , time: %.2f, errors: %s | %s',
                             epoch, (time() - tic), etxtD, etxtG)
            if ( (save_freq > 0) and not ( (epoch + 1) % save_freq) ) or ((epoch + 1) >= (start_epoch + nepochs)) :
                self.save_pattern_params(fit_epoch=epoch+1,
                                         ase_epoch=self.config.pattern.get('aepoch', 0))

        getLogger('ecGAN').info('Learned pattern')

    def fit_assess_pattern(self, data, batch_size):
        if not all([isinstance(net, PatternNet) for net in [self.netD, self.netG]]):
            raise NotImplementedError('At least one net is not a PatternNet!')

        data_iter = gluon.data.DataLoader(data, batch_size)

        trainerD = gluon.Trainer(self.netD.collect_pparams(),
            self.config.pattern.get('optimizer', 'SGD'),
            self.config.pattern.get('optkwargs', {'learning_rate': 0.01}))
        trainerG = gluon.Trainer(self.netG.collect_pparams(),
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

                self.netD.fit_assess_pattern(data)
                self.netG.fit_assess_pattern(data)
                trainerD.step(len(data), ignore_stale_grad=True)
                trainerG.step(len(data), ignore_stale_grad=True)

            etxtD = ', '.join([" & ".join(['%.2e'%(err.mean().asscalar()) for err in toperr if isinstance(err, nd.NDArray)])
                              for toperr in self.netD._err if isinstance(toperr, list)])
            etxtG = ', '.join([" & ".join(['%.2e'%(err.mean().asscalar()) for err in toperr if isinstance(err, nd.NDArray)])
                              for toperr in self.netG._err if isinstance(toperr, list)])
            getLogger('ecGAN').info('pattern training epoch %04d , time: %.2f, errors: %s | %s',
                             epoch, (time() - tic), etxtD, etxtG)
            if ( (save_freq > 0) and not ( (epoch + 1) % save_freq) ) or  ((epoch + 1) >= (start_epoch + nepochs)) :
                self.save_pattern_params(fit_epoch=epoch+1,
                                         ase_epoch=self.config.pattern.get('aepoch', 0))

        getLogger('ecGAN').info('Learned pattern')

    def stats_assess_pattern(self, data, batch_size):
        # if not isinstance(self.netD, PatternNet):
        #     raise NotImplementedError('\'%s\' is not a PatterNet!'%self.config.nets.classifier.type)

        data_iter = gluon.data.DataLoader(data, batch_size)

        net = Sequential()
        net.add(self.netG, self.netD)
        for i, (data, label) in enumerate(data_iter):
            data = data.as_in_context(self.ctx)
            label = label.as_in_context(self.ctx)

            net.forward_logged(data)
            net.stats_assess_pattern()

        self.save_pattern_params(fit_epoch=self.config.pattern.get('start_epoch', 0),
                                 ase_epoch=self.config.pattern.get('aepoch', 0))

    def explain(self, K, noise=None, cond=None, num=None, mkwargs={}, ctx=None):
        if not all([isinstance(net, Explainable) for net in [self.netD, self.netG]]):
            raise NotImplementedError('At least one net is not an Explainable!')

        if num is None:
            if noise is not None:
                num = len(noise)
            elif cond is not None:
                num = len(cond)
            else:
                raise RuntimeError('At least one arg has to be supplied!')
        if noise is None:
            noise = nd.random_normal(shape=(num, 100, 1, 1), ctx=ctx)
        if cond is None:
            cond = nd.random.uniform(0, K, shape=num, ctx=self.ctx).floor()
            cond = nd.one_hot(cond, K).reshape((num, -1, 1, 1))

        net = Sequential()
        with net.name_scope():
            net.add(self.netG, self.netD)
        # self.netG.merge_batchnorm(ctx=self.ctx)
        # self.netD.merge_batchnorm(ctx=self.ctx)
        # Rn, Rc = self.netG.relevance(data=noise, cond=cond, out=None, **mkwargs)
        Rn, Rc = net.relevance(data=[noise, cond], out=None, **mkwargs)
        self._out = net._out

        return Rn, Rc

    def explain_pattern(self, K, noise=None, cond=None, num=None, attribution=False, ctx=None):
        if num is None:
            if noise is not None:
                num = len(noise)
            elif cond is not None:
                num = len(cond)
            else:
                raise RuntimeError('At least one arg has to be supplied!')
        if noise is None:
            noise = nd.random_normal(shape=(num, 100, 1, 1), ctx=ctx)
        if cond is None:
            cond = nd.random.uniform(0,K,shape=num, ctx=self.ctx).floor()
            cond = nd.one_hot(cond, K).reshape((num, -1, 1, 1))

        net = Sequential()
        with net.name_scope():
            net.add(self.netG, self.netD)
        data = [noise, cond]
        R = net.explain_pattern(data, attribution=attribution)
        self._out = net._out
        return R

    def assess_pattern(self):
        net = Sequential()
        net.add(self.netG, self.netD)
        R = net.assess_pattern()

        return R

    def backward_pattern(self, data):
        net = Sequential()
        net.add(self.netG, self.netD)
        y = net.forward_logged(data)
        R = net.backward_pattern(y)

        return R

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

        getLogger('ecGAN').info('Starting training of model %s, discriminator %s, generator %s at epoch %d',
                                config.model, config.nets.discriminator.type, config.nets.generator.type, epoch)

        iter_g = 0

        gpcoef = 0.

        try:
            for epoch in range(self.start_epoch, self.start_epoch + nepochs):
                tic = time()
                for i, data in enumerate(data_iter):
                    # === Discriminator ===
                    if type(data) in [tuple, list]:
                        data = data[0]
                    data = data.as_in_context(ctx)
                    noise = nd.random_normal(shape=(data.shape[0], 100), ctx=ctx)
                    # gpeps = nd.random.uniform()

                    fake = netG(noise)
                    # comb = gpeps * data + (1. - gpeps) * fake
                    # comb.attach_grad()
                    # with autograd.record():
                    #     out_comb = netD(comb)
                    # out_comb.backward()

                    with autograd.record():
                        out_real = netD(data)
                        out_fake = netD(fake.detach())

                        # gradpen = gpcoef * ()
                        errD = (out_real - out_fake).mean(axis=0)
                        errD.backward()

                    trainerD.step(batch_size)
                    metric.update(None, [-errD, ])

                    # === Generator ===
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
                getLogger('ecGAN').info('netD training est epoch %04d: %s=%.4f , time: %.2f', epoch, name, est, (time() - tic))
                if ( (self.save_freq > 0) and not ( (epoch + 1) % self.save_freq) ) or  ((epoch + 1) >= (self.start_epoch + nepochs)):
                    self.checkpoint(epoch+1)
                if ( (self.gen_freq > 0) and not ( (epoch + 1) % self.gen_freq) ) or  ((epoch + 1) >= (self.start_epoch + nepochs)):
                    self.save_generated(epoch+1)
        except KeyboardInterrupt:
            getLogger('ecGAN').info('Training interrupted by user.')
            self.checkpoint('I%d'%epoch)
            self.save_generated('I%d'%epoch)

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

        getLogger('ecGAN').info('Starting training of model %s, discriminator %s, generator %s at epoch %d',
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
                getLogger('ecGAN').info('netD training est epoch %04d: %s=%.4f , time: %.2f', epoch, name, est, (time() - tic))
                if ( (self.save_freq > 0) and not ( (epoch + 1) % self.save_freq) ) or  ((epoch + 1) >= (self.start_epoch + nepochs)):
                    self.checkpoint(epoch+1)
                if ( (self.gen_freq > 0) and not ( (epoch + 1) % self.gen_freq) ) or  ((epoch + 1) >= (self.start_epoch + nepochs)):
                    cond = one_hot(linspace(0, K, 30, ctx=ctx, dtype='int32'), K)
                    self.save_generated(epoch+1, cond)
        except KeyboardInterrupt:
            getLogger('ecGAN').info('Training interrupted by user.')
            self.checkpoint('I%d'%epoch)
            cond = one_hot(linspace(0, K, 30, ctx=ctx, dtype='int32'), K)
            self.save_generated('I%d'%epoch, cond)


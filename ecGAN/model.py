import mxnet as mx
import numpy as np
from time import time
from mxnet import gluon, autograd, nd
from imageio import imwrite
from logging import getLogger

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
                getLogger('ecGAN').debug('Not saving %s \'%s\' epoch %s.', key, self.config.nets[key].type, epoch, fpath)

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

                    with autograd.record():
                        output = netC(data)
                        err = loss(output, label)

                        err.backward()

                    trainer.step(batch_size)
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
        data_iter = gluon.data.DataLoader(data, batch_size, last_batch='discard')

        metric = mx.metric.Accuracy()
        for i, (data, label) in enumerate(data_iter):
            data = data.as_in_context(self.ctx).reshape((-1, 784))
            label = label.as_in_context(self.ctx)

            output = self.netC(data)
            metric.update([label, ], [output, ])

        name, acc = metric.get()

        getLogger('ecGAN').info('%s test acc: %s=%.4f', self.config.nets.classifier.type, name, acc)

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
            getLogger('ecGAN').debug('Explanation sums: %s', ', '.join([str(fl) for fl in Rsums]))

        return R[-1]

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

        getLogger('ecGAN').info('Learned signal estimator %s for net %s', estimator, self.config.nets.classifier.type)

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

    def explain_pattern(self, data):
        if not isinstance(self.netC, PatternNet):
            raise NotImplementedError('\'%s\' is not yet Interpretable!'%self.config.nets.classifier.type)

        R = self.netC.explain_pattern(data)

        return R

    def explain_attribution_pattern(self, data):
        if not isinstance(self.netC, PatternNet):
            raise NotImplementedError('\'%s\' is not yet Interpretable!'%self.config.nets.classifier.type)

        R = self.netC.explain_attribution_pattern(data)

        return R

    def assess_pattern(self):
        if not isinstance(self.netC, PatternNet):
            raise NotImplementedError('\'%s\' is not yet Interpretable!'%self.config.nets.classifier.type)

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
                self.save_pattern_params(epoch+1)

        getLogger('ecGAN').info('Learned pattern for net %s',
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
                getLogger('ecGAN').info('netD training acc epoch %04d: %s=%.4f , time: %.2f', epoch, name, acc, (time() - tic))
                if ( (self.save_freq > 0) and not ( (epoch + 1) % self.save_freq) ) or  ((epoch + 1) >= (self.start_epoch + nepochs)):
                    self.checkpoint(epoch+1)
                if ( (self.gen_freq > 0) and not ( (epoch + 1) % self.gen_freq) ) or  ((epoch + 1) >= (self.start_epoch + nepochs)):
                    self.generate_sample(epoch+1)
        except KeyboardInterrupt:
            getLogger('ecGAN').info('Training interrupted by user.')
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
            getLogger('ecGAN').info('Saved data generated by \'%s\' epoch %s in \'%s\'.',
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
    #         getLogger('ecGAN').info('Saved generated sensitivity by \'%s\' epoch %s in \'%s\'.',
    #                          self.config.nets.generator.type, epoch, fpath)

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
            getLogger('ecGAN').debug('Explanation sums: %s', ', '.join([str(fl) for fl in Rsums]))

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
class CGAN(GAN):
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

        getLogger('ecGAN').info('Starting training of model %s, discriminator %s, generator %s at epoch %d',
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
                getLogger('ecGAN').info('netD training acc epoch %04d: %s=%.4f , time: %.2f', epoch, name, acc, (time() - tic))
                if ( (self.save_freq > 0) and not ( (epoch + 1) % self.save_freq) ) or  ((epoch + 1) >= (self.start_epoch + nepochs)):
                    self.checkpoint(epoch+1)
                if ( (self.gen_freq > 0) and not ( (epoch + 1) % self.gen_freq) ) or  ((epoch + 1) >= (self.start_epoch + nepochs)):
                    cond = one_hot(linspace(0, K, 30, ctx=ctx, dtype='int32'), K).reshape((30, K, 1, 1))
                    self.generate_sample(epoch+1, cond)
        except KeyboardInterrupt:
            getLogger('ecGAN').info('Training interrupted by user.')
            self.checkpoint('I%d'%epoch)
            cond = one_hot(linspace(0, K, 30, ctx=ctx, dtype='int32'), K).reshape((30, K, 1, 1))
            self.generate_sample('I%d'%epoch, cond)

    def test(self, data, batch_size):
        data_iter = gluon.data.DataLoader(data, batch_size, last_batch='discard')

        metric = mx.metric.Accuracy()
        for i, (data, label) in enumerate(data_iter):
            data = data.as_in_context(self.ctx).reshape((-1, 784))
            label = label.as_in_context(self.ctx)

            output = self.netC(data)
            metric.update([label, ], [output, ])

        name, acc = metric.get()

        getLogger('ecGAN').info('%s test acc: %s=%.4f', self.config.nets.classifier.type, name, acc)

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

                self.netG.fit_pattern(noise, cond)
                self.netD.fit_pattern(data)

                trainerD.step(num, ignore_stale_grad=True)
                trainerG.step(num, ignore_stale_grad=True)

                # === Fake Pass ===
                noise = nd.random_normal(shape=(num, 100, 1, 1), ctx=self.ctx)
                rand_cond = nd.random.uniform(0,K,shape=num, ctx=self.ctx).floor()
                cond = one_hot(rand_cond, K).reshape((num, -1, 1, 1))

                fake = self.netG.fit_pattern(noise, cond)
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

    def explain_pattern(self, K, noise=None, cond=None, num=None):
        if not all([isinstance(net, PatternNet) for net in [self.netD, self.netG]]):
            raise NotImplementedError('At least one net is not a PatternNet!')


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
            one_hot = fuzzy_one_hot if config.fuzzy_labels else nd.one_hot
            cond = nd.random.uniform(0,K,shape=num, ctx=self.ctx).floor()
            cond = one_hot(cond, K).reshape((num, -1, 1, 1))

        noise.attach_grad()
        cond.attach_grad()
        with autograd.record():
            y_G = self.netG.forward_pattern(noise, cond)
            y_D = self.netD.forward_pattern(*y_G)
        y_D[1].backward(out_grad=y_D[0])

        return noise.grad, cond.grad

    def explain_attribution_pattern(self, data):
        if not all([isinstance(net, PatternNet) for net in [self.netD, self.netG]]):
            raise NotImplementedError('At least one net is not a PatternNet!')

        R = self.netC.explain_attribution_pattern(data)

        return R

    def assess_pattern(self):
        if not isinstance(self.netC, PatternNet):
            raise NotImplementedError('\'%s\' is not yet Interpretable!'%self.config.nets.classifier.type)

        R = self.netC.assess_pattern()

        return R

    def backward_pattern(self, data):
        if not isinstance(self.netC, PatternNet):
            raise NotImplementedError('\'%s\' is no PatternNet!'%self.config.nets.classifier.type)

        y = self.netC.forward_logged(data)
        R = self.netC.backward_pattern(y)

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
                getLogger('ecGAN').info('netD training est epoch %04d: %s=%.4f , time: %.2f', epoch, name, est, (time() - tic))
                if ( (self.save_freq > 0) and not ( (epoch + 1) % self.save_freq) ) or  ((epoch + 1) >= (self.start_epoch + nepochs)):
                    self.checkpoint(epoch+1)
                if ( (self.gen_freq > 0) and not ( (epoch + 1) % self.gen_freq) ) or  ((epoch + 1) >= (self.start_epoch + nepochs)):
                    self.generate_sample(epoch+1)
        except KeyboardInterrupt:
            getLogger('ecGAN').info('Training interrupted by user.')
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
                    self.generate_sample(epoch+1, cond)
        except KeyboardInterrupt:
            getLogger('ecGAN').info('Training interrupted by user.')
            self.checkpoint('I%d'%epoch)
            cond = one_hot(linspace(0, K, 30, ctx=ctx, dtype='int32'), K)
            self.generate_sample('I%d'%epoch, cond)


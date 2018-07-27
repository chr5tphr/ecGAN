from ecGAN.net import register_net
from ecGAN.layer import Sequential, Dense, Conv2D, Conv2DTranspose, Identity, BatchNorm, LeakyReLU, ReLU, Concat, BatchNorm, Clip, Tanh, Flatten
from ecGAN.explain.pattern.regimes import LinearPatternRegime, PositivePatternRegime, NegativePatternRegime
from ecGAN.explain.pattern.estimator import estimators

@register_net
class MYTCN28(Sequential):
    def __init__(self, **kwargs):
        outnum = kwargs.pop('outnum', 1)
        outact = kwargs.pop('outact', None)
        numhid = kwargs.pop('numhid', 64)
        clip = kwargs.pop('clip', [-1., 1.])
        use_bias = kwargs.pop('use_bias', False)

        # patest = dict(relu='relu', clip='clip', pixel='relu', gauss='relu')
        patest = dict(relu='linear', clip='linear', pixel='linear', gauss='linear')
        patest.update(kwargs.pop('patest', {}))
        explain = dict(relu='zplus', clip='zclip', pixel='zb', gauss='wsquare')
        explain.update(kwargs.pop('explain', {}))
        super().__init__(**kwargs)
        with self.name_scope():

            self.add(Concat())

            self.add(Conv2DTranspose(numhid * 8, 4, strides=1, padding=0, use_bias=use_bias,
                                     explain=explain['gauss'], regimes=estimators[patest['gauss']]()))
            self.add(BatchNorm())
            self.add(ReLU())
            # _numhid x 4 x 4

            self.add(Conv2DTranspose(numhid * 4, 4, strides=1, padding=0, use_bias=use_bias,
                                     explain=explain['relu'], regimes=estimators[patest['relu']]()))
            self.add(BatchNorm())
            self.add(ReLU())
            # _numhid x 7 x 7

            self.add(Conv2DTranspose(numhid * 2, 4, strides=2, padding=1, use_bias=use_bias,
                                     explain=explain['relu'], regimes=estimators[patest['relu']]()))
            self.add(BatchNorm())
            self.add(ReLU())
            # _numhid x 14 x 14

            self.add(Conv2DTranspose(outnum, 4, strides=2, padding=1, use_bias=use_bias,
                                     explain=explain['clip'], regimes=estimators[patest['clip']](),
                                     noregconst=-1.))
            if outact == 'relu':
                self.add(ReLU())
            elif outact == 'clip':
                self.add(Clip(low=clip[0], high=clip[1]))
            elif outact == 'tanh':
                self.add(Tanh())
            elif outact == 'batchnorm':
                self.add(BatchNorm(scale=False, center=False))
                self.add(Identity())
            else:
                self.add(Identity())
            # _numhid x 28 x 28

@register_net
class MSCN28(Sequential):
    def __init__(self, **kwargs):
        outnum = kwargs.pop('outnum', 1)
        outact = kwargs.pop('outact', None)
        numhid = kwargs.pop('numhid', 64)
        leakage = kwargs.pop('leakage', 0.1)
        use_bias = kwargs.pop('use_bias', False)

        # patest = dict(relu='relu', clip='clip', pixel='relu', gauss='relu')
        patest = dict(relu='linear', clip='linear', pixel='linear', gauss='linear')
        patest.update(kwargs.pop('patest', {}))
        explain = dict(relu='zplus', clip='zclip', pixel='zb', gauss='wsquare')
        explain.update(kwargs.pop('explain', {}))
        super().__init__(**kwargs)
        with self.name_scope():
            # _numhid x 28 x 28
            self.add(Conv2D(numhid, 4, strides=2, padding=1, use_bias=use_bias,
                            explain=explain['pixel'], regimes=estimators[patest['pixel']]()))
            self.add(LeakyReLU(leakage))
            # _numhid x 14 x 14

            self.add(Conv2D(numhid * 2, 4, strides=2, padding=1, use_bias=use_bias,
                            explain=explain['relu'], regimes=estimators[patest['relu']]()))
            self.add(BatchNorm())
            self.add(LeakyReLU(leakage))
            # _numhid x 7 x 7

            self.add(Conv2D(numhid * 4, 4, strides=1, padding=0, use_bias=use_bias,
                            explain=explain['relu'], regimes=estimators[patest['relu']]()))
            self.add(BatchNorm())
            self.add(LeakyReLU(leakage))
            # _numhid x 4 x 4

            self.add(Conv2D(outnum, 4, strides=1, padding=0, use_bias=use_bias,
                            explain=explain['relu'], regimes=estimators[patest['relu']]()))
            self.add(Flatten())
            # self.add(Conv2D(numhid * 8, 4, strides=1, padding=0, use_bias=use_bias,
            #                 explain=explain['relu'], regimes=estimators[patest['relu']]()))
            # self.add(BatchNorm())
            # self.add(LeakyReLU(leakage))
            # # filters x 1 x 1

            # self.add(Dense(outnum,
            #                explain=explain['relu'], regimes=estimators[patest['relu']]()))
            if outact == 'relu':
                self.add(ReLU())
            else:
                self.add(Identity())

@register_net
class MYTCN32(Sequential):
    def __init__(self, **kwargs):
        outnum = kwargs.pop('outnum', 1)
        outact = kwargs.pop('outact', None)
        numhid = kwargs.pop('numhid', 64)
        clip = kwargs.pop('clip', [-1., 1.])
        use_bias = kwargs.pop('use_bias', False)

        # patest = dict(relu='relu', clip='clip', pixel='relu', gauss='relu')
        patest = dict(relu='linear', clip='linear', pixel='linear', gauss='linear')
        patest.update(kwargs.pop('patest', {}))
        explain = dict(relu='zplus', clip='zclip', pixel='zb', gauss='wsquare')
        explain.update(kwargs.pop('explain', {}))
        super().__init__(**kwargs)
        with self.name_scope():

            self.add(Concat())

            self.add(Conv2DTranspose(numhid * 8, 4, strides=1, padding=0, use_bias=use_bias,
                                     explain=explain['gauss'], regimes=estimators[patest['gauss']]()))
            self.add(BatchNorm())
            self.add(ReLU())
            # _numhid x 4 x 4

            self.add(Conv2DTranspose(numhid * 4, 4, strides=2, padding=1, use_bias=use_bias,
                                     explain=explain['relu'], regimes=estimators[patest['relu']]()))
            self.add(BatchNorm())
            self.add(ReLU())
            # _numhid x 8 x 8

            self.add(Conv2DTranspose(numhid * 2, 4, strides=2, padding=1, use_bias=use_bias,
                                     explain=explain['relu'], regimes=estimators[patest['relu']]()))
            self.add(BatchNorm())
            self.add(ReLU())
            # _numhid x 16 x 16

            self.add(Conv2DTranspose(outnum, 4, strides=2, padding=1, use_bias=use_bias,
                                     explain=explain['clip'], regimes=estimators[patest['clip']](),
                                     noregconst=-1.))
            if outact == 'relu':
                self.add(ReLU())
            elif outact == 'clip':
                self.add(Clip(low=clip[0], high=clip[1]))
            elif outact == 'tanh':
                self.add(Tanh())
            elif outact == 'batchnorm':
                self.add(BatchNorm(scale=False, center=False))
                self.add(Identity())
            else:
                self.add(Identity())
            # _numhid x 32 x 32

@register_net
class MSCN32(Sequential):
    def __init__(self, **kwargs):
        outnum = kwargs.pop('outnum', 1)
        outact = kwargs.pop('outact', None)
        numhid = kwargs.pop('numhid', 64)
        leakage = kwargs.pop('leakage', 0.1)
        use_bias = kwargs.pop('use_bias', False)

        # patest = dict(relu='relu', clip='clip', pixel='relu', gauss='relu')
        patest = dict(relu='linear', clip='linear', pixel='linear', gauss='linear')
        patest.update(kwargs.pop('patest', {}))
        explain = dict(relu='zplus', clip='zclip', pixel='zb', gauss='wsquare')
        explain.update(kwargs.pop('explain', {}))
        super().__init__(**kwargs)
        with self.name_scope():
            # _numhid x 32 x 32
            self.add(Conv2D(numhid, 4, strides=2, padding=1, use_bias=use_bias,
                            explain=explain['pixel'], regimes=estimators[patest['pixel']]()))
            self.add(LeakyReLU(leakage))
            # _numhid x 16 x 14

            self.add(Conv2D(numhid * 2, 4, strides=2, padding=1, use_bias=use_bias,
                            explain=explain['relu'], regimes=estimators[patest['relu']]()))
            self.add(BatchNorm())
            self.add(LeakyReLU(leakage))
            # _numhid x 8 x 8

            self.add(Conv2D(numhid * 4, 4, strides=2, padding=1, use_bias=use_bias,
                            explain=explain['relu'], regimes=estimators[patest['relu']]()))
            self.add(BatchNorm())
            self.add(LeakyReLU(leakage))
            # _numhid x 4 x 4

            self.add(Conv2D(outnum, 4, strides=1, padding=0, use_bias=use_bias,
                            explain=explain['relu'], regimes=estimators[patest['relu']]()))
            self.add(Flatten())
            # self.add(BatchNorm())
            # self.add(LeakyReLU(leakage))
            # # filters x 1 x 1

            # self.add(Dense(outnum,
            #                explain=explain['relu'], regimes=estimators[patest['relu']]()))
            if outact == 'relu':
                self.add(ReLU())
            else:
                self.add(Identity())

@register_net
class MSFC(Sequential):
    def __init__(self, **kwargs):
        outnum = kwargs.pop('outnum', 1)
        numhid = kwargs.pop('numhid', 64)
        outact = kwargs.pop('outact', None)
        patest = kwargs.pop('patest', 'linear')
        outest = kwargs.pop('outest', patest)
        super().__init__(**kwargs)
        with self.name_scope():
            self.add(Dense(numhid, regimes=estimators[patest]()))
            self.add(ReLU(regimes=estimators[patest]()))
            self.add(Dense(numhid, regimes=estimators[patest]()))
            self.add(ReLU(regimes=estimators[patest]()))
            self.add(Dense(numhid, regimes=estimators[patest]()))
            self.add(ReLU(regimes=estimators[patest]()))
            self.add(Dense(outnum, regimes=estimators[outest]()))
            self.add(Identity(regimes=estimators[outest]()))


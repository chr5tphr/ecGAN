from ecGAN.net import register_net
from ecGAN.layer import Sequential, YSequential, Dense, Conv2D, Conv2DTranspose, Identity, BatchNorm, LeakyReLU, Activation, ReLU
from ecGAN.pattern.regimes import LinearPatternRegime, PositivePatternRegime, NegativePatternRegime
from ecGAN.pattern.estimator import estimators

@register_net
class MYTCN28(YSequential):
    def __init__(self, **kwargs):
        self._outnum = kwargs.pop('outnum', 1)
        self._numhid = kwargs.pop('numhid', 64)
        self._patest = kwargs.pop('patest', 'linear')
        self._outest = kwargs.pop('outest', self._patest)
        super().__init__(**kwargs)
        with self.name_scope():

            self.addData(Identity())
            self.addCond(Identity())

            self.add(Conv2DTranspose(self._numhid * 8, 4, strides=1, padding=0, use_bias=False, isinput=True, regimes=estimators[self._patest]()))
            self.add(ReLU(regimes=estimators[self._patest]()))
            self.add(Activation('relu'))
            # _numhid x 4 x 4

            self.add(Conv2DTranspose(self._numhid * 4, 4, strides=1, padding=0, use_bias=False, regimes=estimators[self._patest]()))
            self.add(ReLU(regimes=estimators[self._patest]()))
            # self.add(Activation('relu'))
            # _numhid x 7 x 7

            self.add(Conv2DTranspose(self._numhid * 2, 4, strides=2, padding=1, use_bias=False, regimes=estimators[self._patest]()))
            self.add(ReLU(regimes=estimators[self._patest]()))
            # self.add(Activation('relu'))
            # _numhid x 14 x 14

            self.add(Conv2DTranspose(self._outnum, 4, strides=2, padding=1, use_bias=False, regimes=estimators[self._outest]()))
            self.add(Identity(regimes=estimators[self._outest]()))
            # self.add(Activation(self._outact))
            # _numhid x 28 x 28

@register_net
class MSCN28(Sequential):
    def __init__(self, **kwargs):
        self._outnum = kwargs.pop('outnum', 1)
        self._numhid = kwargs.pop('numhid', 64)
        self._outact = kwargs.pop('outact', None)
        self._patest = kwargs.pop('patest', 'linear')
        self._outest = kwargs.pop('outest', self._patest)
        super().__init__(**kwargs)
        with self.name_scope():
            # _numhid x 28 x 28
            self.add(Conv2D(self._numhid, 4, strides=2, padding=1, use_bias=False, regimes=estimators[self._patest]()))
            self.add(ReLU(regimes=estimators[self._patest]()))
            # _numhid x 14 x 14

            self.add(Conv2D(self._numhid * 2, 4, strides=2, padding=1, use_bias=False, regimes=estimators[self._patest]()))
            self.add(ReLU(regimes=estimators[self._patest]()))
            # _numhid x 7 x 7

            self.add(Conv2D(self._numhid * 4, 4, strides=1, padding=0, use_bias=False, regimes=estimators[self._patest]()))
            self.add(ReLU(regimes=estimators[self._patest]()))
            # _numhid x 4 x 4

            self.add(Conv2D(self._numhid * 8, 4, strides=1, padding=0, use_bias=False, regimes=estimators[self._patest]()))
            self.add(ReLU(regimes=estimators[self._patest]()))
            # filters x 1 x 1

            self.add(Dense(self._outnum, regimes=estimators[self._outest]()))
            self.add(Identity(regimes=estimators[self._outest]()))

@register_net
class MSFC(Sequential):
    def __init__(self, **kwargs):
        self._outnum = kwargs.pop('outnum', 1)
        self._numhid = kwargs.pop('numhid', 64)
        self._outact = kwargs.pop('outact', None)
        self._patest = kwargs.pop('patest', 'linear')
        self._outest = kwargs.pop('outest', self._patest)
        super().__init__(**kwargs)
        with self.name_scope():
            self.add(Dense(self._numhid, regimes=estimators[self._patest]()))
            self.add(ReLU(regimes=estimators[self._patest]()))
            self.add(Dense(self._numhid, regimes=estimators[self._patest]()))
            self.add(ReLU(regimes=estimators[self._patest]()))
            self.add(Dense(self._numhid, regimes=estimators[self._patest]()))
            self.add(ReLU(regimes=estimators[self._patest]()))
            self.add(Dense(self._outnum, regimes=estimators[self._outest]()))
            self.add(Identity(regimes=estimators[self._outest]()))


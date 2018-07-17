from ecGAN.net import register_net
from ecGAN.layer import Sequential, YSequential, Dense, Conv2D, Conv2DTranspose, Identity, BatchNorm, LeakyReLU, Activation, ReLU, Concat, BatchNorm, Clip, Tanh
from ecGAN.pattern.regimes import LinearPatternRegime, PositivePatternRegime, NegativePatternRegime
from ecGAN.pattern.estimator import estimators

@register_net
class MYTCN28(Sequential):
    def __init__(self, **kwargs):
        self._outnum = kwargs.pop('outnum', 1)
        self._outact = kwargs.pop('outact', None)
        self._numhid = kwargs.pop('numhid', 64)
        self._patest = kwargs.pop('patest', 'linear')
        self._outest = kwargs.pop('outest', self._patest)
        self._clip = kwargs.pop('clip', [-1., 1.])
        self._use_bias = kwargs.pop('use_bias', False)
        super().__init__(**kwargs)
        with self.name_scope():

            self.add(Concat())

            self.add(Conv2DTranspose(self._numhid * 8, 4, strides=1, padding=0, use_bias=self._use_bias, isinput=True, regimes=estimators[self._patest]()))
            self.add(BatchNorm())
            self.add(ReLU(regimes=estimators[self._patest]()))
            # _numhid x 4 x 4

            self.add(Conv2DTranspose(self._numhid * 4, 4, strides=1, padding=0, use_bias=self._use_bias, regimes=estimators[self._patest]()))
            self.add(BatchNorm())
            self.add(ReLU(regimes=estimators[self._patest]()))
            # _numhid x 7 x 7

            self.add(Conv2DTranspose(self._numhid * 2, 4, strides=2, padding=1, use_bias=self._use_bias, regimes=estimators[self._patest]()))
            self.add(BatchNorm())
            self.add(ReLU(regimes=estimators[self._patest]()))
            # _numhid x 14 x 14

            self.add(Conv2DTranspose(self._outnum, 4, strides=2, padding=1, use_bias=self._use_bias, regimes=estimators[self._outest]()))
            if self._outact == 'relu':
                self.add(ReLU(regimes=estimators[self._outest]()))
            elif self._outact == 'clip':
                self.add(Clip(low=self._clip[0], high=self._clip[1], regimes=estimators[self._outest]()))
            elif self._outact == 'tanh':
                self.add(Tanh(regimes=estimators[self._outest]()))
            elif self._outact == 'batchnorm':
                self.add(BatchNorm(scale=False, center=False))
                self.add(Identity(regimes=estimators[self._outest]()))
            else:
                self.add(Identity(regimes=estimators[self._outest]()))
            # _numhid x 28 x 28

@register_net
class MSCN28(Sequential):
    def __init__(self, **kwargs):
        self._outnum = kwargs.pop('outnum', 1)
        self._outact = kwargs.pop('outact', None)
        self._numhid = kwargs.pop('numhid', 64)
        self._patest = kwargs.pop('patest', 'linear')
        self._outest = kwargs.pop('outest', self._patest)
        self._isinput= kwargs.pop('isinput', False)
        self._leakage = kwargs.pop('leakage', 0.1)
        self._use_bias = kwargs.pop('use_bias', False)
        super().__init__(**kwargs)
        with self.name_scope():
            # _numhid x 28 x 28
            self.add(Conv2D(self._numhid, 4, strides=2, padding=1, use_bias=self._use_bias, regimes=estimators[self._patest](), isinput=True))
            self.add(LeakyReLU(self._leakage, regimes=estimators[self._patest]()))
            # _numhid x 14 x 14

            self.add(Conv2D(self._numhid * 2, 4, strides=2, padding=1, use_bias=self._use_bias, regimes=estimators[self._patest]()))
            self.add(BatchNorm())
            self.add(LeakyReLU(self._leakage, regimes=estimators[self._patest]()))
            # _numhid x 7 x 7

            self.add(Conv2D(self._numhid * 4, 4, strides=1, padding=0, use_bias=self._use_bias, regimes=estimators[self._patest]()))
            self.add(BatchNorm())
            self.add(LeakyReLU(self._leakage, regimes=estimators[self._patest]()))
            # _numhid x 4 x 4

            self.add(Conv2D(self._numhid * 8, 4, strides=1, padding=0, use_bias=self._use_bias, regimes=estimators[self._patest]()))
            self.add(BatchNorm())
            self.add(LeakyReLU(self._leakage, regimes=estimators[self._patest]()))
            # filters x 1 x 1

            self.add(Dense(self._outnum, regimes=estimators[self._outest]()))
            if self._outact == 'relu':
                self.add(ReLU(regimes=estimators[self._outest]()))
            else:
                self.add(Identity(regimes=estimators[self._outest]()))

@register_net
class MYTCN32(Sequential):
    def __init__(self, **kwargs):
        self._outnum = kwargs.pop('outnum', 1)
        self._outact = kwargs.pop('outact', None)
        self._numhid = kwargs.pop('numhid', 64)
        self._patest = kwargs.pop('patest', 'linear')
        self._outest = kwargs.pop('outest', self._patest)
        super().__init__(**kwargs)
        with self.name_scope():

            self.add(Concat())

            self.add(Conv2DTranspose(self._numhid * 16, 3, strides=1, padding=0, use_bias=False, isinput=True, regimes=estimators[self._patest]()))
            self.add(ReLU(regimes=estimators[self._patest]()))
            # _numhid x 3 x 3

            self.add(Conv2DTranspose(self._numhid * 8, 3, strides=1, padding=0, use_bias=False, regimes=estimators[self._patest]()))
            self.add(ReLU(regimes=estimators[self._patest]()))
            # _numhid x 5 x 5

            self.add(Conv2DTranspose(self._numhid * 4, 4, strides=1, padding=0, use_bias=False, regimes=estimators[self._patest]()))
            self.add(ReLU(regimes=estimators[self._patest]()))
            # _numhid x 8 x 8

            self.add(Conv2DTranspose(self._numhid * 2, 4, strides=2, padding=1, use_bias=False, regimes=estimators[self._patest]()))
            self.add(ReLU(regimes=estimators[self._patest]()))
            # _numhid x 16 x 16

            self.add(Conv2DTranspose(self._outnum, 4, strides=2, padding=1, use_bias=False, regimes=estimators[self._outest]()))
            if self._outact == 'relu':
                self.add(ReLU(regimes=estimators[self._outest]()))
            else:
                self.add(Identity(regimes=estimators[self._outest]()))
            # _numhid x 32 x 32

@register_net
class MSCN32(Sequential):
    def __init__(self, **kwargs):
        self._outnum = kwargs.pop('outnum', 1)
        self._outact = kwargs.pop('outact', None)
        self._numhid = kwargs.pop('numhid', 64)
        self._patest = kwargs.pop('patest', 'linear')
        self._outest = kwargs.pop('outest', self._patest)
        super().__init__(**kwargs)
        with self.name_scope():
            # _numhid x 32 x 32
            self.add(Conv2D(self._numhid, 4, strides=2, padding=1, use_bias=False, regimes=estimators[self._patest]()))
            self.add(ReLU(regimes=estimators[self._patest]()))
            # _numhid x 16 x 16

            self.add(Conv2D(self._numhid * 2, 4, strides=2, padding=1, use_bias=False, regimes=estimators[self._patest]()))
            self.add(ReLU(regimes=estimators[self._patest]()))
            # _numhid x 8 x 8

            self.add(Conv2D(self._numhid * 4, 4, strides=1, padding=0, use_bias=False, regimes=estimators[self._patest]()))
            self.add(ReLU(regimes=estimators[self._patest]()))
            # _numhid x 5 x 5

            self.add(Conv2D(self._numhid * 8, 3, strides=1, padding=0, use_bias=False, regimes=estimators[self._patest]()))
            self.add(ReLU(regimes=estimators[self._patest]()))
            # _numhid x 3 x 3

            self.add(Conv2D(self._numhid * 16, 3, strides=1, padding=0, use_bias=False, regimes=estimators[self._patest]()))
            self.add(ReLU(regimes=estimators[self._patest]()))
            # filters x 1 x 1

            self.add(Dense(self._outnum, regimes=estimators[self._outest]()))
            if self._outact == 'relu':
                self.add(ReLU(regimes=estimators[self._outest]()))
            else:
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


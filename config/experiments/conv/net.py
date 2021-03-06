from mxnet import nd
from mxnet.gluon import nn

from ecGAN.net import register_net
from ecGAN.layer import Sequential, YSequential, Dense, Conv2D, Conv2DTranspose, Identity, BatchNorm, LeakyReLU, Activation, ReLU, MaxPool2D
from ecGAN.pattern.regimes import LinearPatternRegime, PositivePatternRegime, NegativePatternRegime
from ecGAN.pattern.estimator import estimators

@register_net
class MSCN28(Sequential):
    def __init__(self, **kwargs):
        self._numhid = kwargs.pop('numhid', 64)
        self._outnum = kwargs.pop('outnum', 10)
        self._outact = kwargs.pop('outact', None)
        self._patest = kwargs.pop('patest', 'linear')
        self._outest = kwargs.pop('outest', self._patest)
        super().__init__(**kwargs)
        with self.name_scope():
            # _numhid x 28 x 28
            self.add(Conv2D(self._numhid, 4, strides=2, padding=1, use_bias=False, regimes=estimators[self._patest](), isinput=True))
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
            if self._outact == 'relu':
                self.add(ReLU(regimes=estimators[self._outest]()))
            else:
                self.add(Identity(regimes=estimators[self._outest]()))


@register_net
class M1SCN28(Sequential):
    def __init__(self, **kwargs):
        self._numhid = kwargs.pop('numhid', 64)
        self._outnum = kwargs.pop('outnum', 10)
        self._patest = kwargs.pop('patest', 'linear')
        self._outest = kwargs.pop('outest', self._patest)
        super().__init__(**kwargs)
        with self.name_scope():
            # 3 x 28 x 28
            self.add(Conv2D(self._numhid, 5, strides=1, padding=0, use_bias=True, regimes=estimators[self._patest]()))
            self.add(ReLU(regimes=estimators[self._patest]()))
            # filt x 22 x 22
            self.add(MaxPool2D(pool_size=2, strides=2))
            # filt x 11 x 11

            self.add(Conv2D(self._numhid * 2, 4, strides=1, padding=0, use_bias=True, regimes=estimators[self._patest]()))
            # filt x  8 x  8
            self.add(ReLU(regimes=estimators[self._patest]()))
            self.add(MaxPool2D(pool_size=2, strides=2))
            # filt x  4 x  4

            self.add(Dense(self._numhid * 2, regimes=estimators[self._patest]()))
            self.add(Dense(self._outnum, regimes=estimators[self._outest]()))
            self.add(Identity(regimes=estimators[self._outest]()))


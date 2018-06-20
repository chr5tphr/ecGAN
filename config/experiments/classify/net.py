from ecGAN.net import register_net
from ecGAN.layer import Sequential, YSequential, Dense, Conv2D, Conv2DTranspose, Identity, BatchNorm, LeakyReLU, Activation, ReLU
from ecGAN.pattern.regimes import LinearPatternRegime, PositivePatternRegime, NegativePatternRegime
from ecGAN.pattern.estimator import estimators


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


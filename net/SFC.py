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
        pat_est = estimators[self._patest]
        out_est = estimators[self._outest]
        super().__init__(**kwargs)
        with self.name_scope():
            self.add(Dense(self._numhid, regimes=pat_est()))
            self.add(ReLU(regimes=pat_est()))
            self.add(Dense(self._numhid, regimes=pat_est()))
            self.add(ReLU(regimes=pat_est()))
            self.add(Dense(self._numhid, regimes=pat_est()))
            self.add(ReLU(regimes=pat_est()))
            self.add(Dense(self._outnum, regimes=out_est()))
            self.add(Identity(regimes=out_est()))


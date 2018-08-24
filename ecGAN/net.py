from mxnet import nd
from mxnet.gluon import nn
from .layer import Sequential, Dense, Conv2D, Conv2DTranspose, Identity, BatchNorm, LeakyReLU, ReLU, Dropout, Flatten
from .explain.pattern.regimes import LinearPatternRegime, PositivePatternRegime, NegativePatternRegime
from .explain.pattern.estimator import estimators

nets = {}
def register_net(obj):
    nets[obj.__name__] = obj
    return obj

@register_net
class SFC(Sequential):
    def __init__(self, **kwargs):
        self._outnum = kwargs.pop('outnum', 1)
        self._numhid = kwargs.pop('numhid', 64)
        self._outact = kwargs.pop('outact', None)
        super().__init__(**kwargs)
        with self.name_scope():
            self.add(Dense(self._numhid, activation='relu'))
            self.add(Dense(self._numhid, activation='relu'))
            self.add(Dense(self._numhid, activation='relu'))
            self.add(Dense(self._outnum, activation=self._outact))

@register_net
class SOFC(Sequential):
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

#@register_net
#class mlp_3dense(Sequential):
#    def __init__(self, **kwargs):
#        outnum = kwargs.pop('outnum', 2)
#        outact = kwargs.pop('outact', None)
#        numhid = kwargs.pop('numhid', 512)
#        droprate = kwargs.pop('droprate', 0.25)
#        use_bias = kwargs.pop('use_bias', False)
#
#        # patest = dict(relu='relu', out='clip', pixel='relu', gauss='relu')
#        patest = dict(relu='linear', out='linear', pixel='linear', gauss='linear')
#        patest.update(kwargs.pop('patest', {}))
#        explain = dict(relu='zplus', out='zplus', pixel='zb', gauss='wsquare')
#        explain.update(kwargs.pop('explain', {}))
#        super().__init__(**kwargs)
#        with self.name_scope():
#            self += Flatten()
#
#            self += Dense(numhid,
#                          explain=explain['relu'], regimes=estimators[patest['relu']]())
#            self += ReLU()
#            self += Dropout(droprate)
#
#            self += Dense(numhid,
#                          explain=explain['relu'], regimes=estimators[patest['relu']]())
#            self += ReLU()
#            self += Dropout(droprate)
#
#            self += Dense(outnum,
#                          explain=explain['relu'], regimes=estimators[patest['out']]())
#
#            if outact == 'relu':
#                self += ReLU()
#            else:
#                self += Identity()
#
# @register_net
# class YSFC(YSequential):
#     def __init__(self, **kwargs):
#         self._outnum = kwargs.pop('outnum', 1)
#         self._numhid = kwargs.pop('numhid', 64)
#         self._outact = kwargs.pop('outact', None)
#         super().__init__(**kwargs)
#         with self.name_scope():
#             self.addData(Dense(self._numhid, activation='relu', isinput=True))
#
#             self.addCond(Dense(self._numhid, activation='relu'))
#
#             self.add(Dense(self._numhid, activation='relu'))
#             self.add(Dense(self._numhid, activation='relu'))
#             self.add(Dense(self._outnum, activation=self._outact))
#
# @register_net
# class YTCN28(YSequential):
#     def __init__(self, **kwargs):
#         self._outnum = kwargs.pop('outnum', 1)
#         self._numhid = kwargs.pop('numhid', 64)
#         self._outact = kwargs.pop('outact', None)
#         super().__init__(**kwargs)
#         with self.name_scope():
#
#             self.addData(Identity())
#             self.addCond(Identity())
#
#             self.add(Conv2DTranspose(self._numhid * 8, 4, strides=1, padding=0, use_bias=False, activation='relu', isinput=True))
#             # self.add(Activation('relu'))
#             # _numhid x 4 x 4
#
#             self.add(Conv2DTranspose(self._numhid * 4, 4, strides=1, padding=0, use_bias=False, activation='relu'))
#             # self.add(Activation('relu'))
#             # _numhid x 7 x 7
#
#             self.add(Conv2DTranspose(self._numhid * 2, 4, strides=2, padding=1, use_bias=False, activation='relu'))
#             # self.add(Activation('relu'))
#             # _numhid x 14 x 14
#
#             self.add(Conv2DTranspose(self._outnum, 4, strides=2, padding=1, use_bias=False, activation=self._outact))
#             # self.add(Activation(self._outact))
#             # _numhid x 28 x 28

@register_net
class STCN28(Sequential):
    def __init__(self, **kwargs):
        self._outnum = kwargs.pop('outnum', 1)
        self._numhid = kwargs.pop('numhid', 64)
        self._outact = kwargs.pop('outact', None)
        super().__init__(**kwargs)
        with self.name_scope():
            self.add(Conv2DTranspose(self._numhid * 8, 4, strides=1, padding=0, use_bias=False, activation='relu', isinput=True))
            # self.add(Activation('relu'))
            # _numhid x 4 x 4

            self.add(Conv2DTranspose(self._numhid * 4, 4, strides=1, padding=0, use_bias=False, activation='relu'))
            # self.add(Activation('relu'))
            # _numhid x 7 x 7

            self.add(Conv2DTranspose(self._numhid * 2, 4, strides=2, padding=1, use_bias=False, activation='relu'))
            # self.add(Activation('relu'))
            # _numhid x 14 x 14

            self.add(Conv2DTranspose(self._outnum, 4, strides=2, padding=1, use_bias=False, activation=self._outact))
            # self.add(Activation(self._outact))
            # _numhid x 28 x 28

@register_net
class SCN28(Sequential):
    def __init__(self, **kwargs):
        self._outnum = kwargs.pop('outnum', 1)
        self._numhid = kwargs.pop('numhid', 64)
        self._outact = kwargs.pop('outact', None)
        super().__init__(**kwargs)
        with self.name_scope():
            # _numhid x 28 x 28
            self.add(Conv2D(self._numhid, 4, strides=2, padding=1, use_bias=False, activation='relu'))
            # self.add(Activation('relu'))
            # _numhid x 14 x 14

            self.add(Conv2D(self._numhid * 2, 4, strides=2, padding=1, use_bias=False, activation='relu'))
            # self.add(Activation('relu'))
            # _numhid x 7 x 7

            self.add(Conv2D(self._numhid * 4, 4, strides=1, padding=0, use_bias=False, activation='relu'))
            # self.add(Activation('relu'))
            # _numhid x 4 x 4

            self.add(Conv2D(self._numhid * 8, 4, strides=1, padding=0, use_bias=False, activation='relu'))
            # self.add(Activation('relu'))
            # filters x 1 x 1

            self.add(Dense(self._outnum, activation=self._outact))

# Sequential Transposed Convolutional-only 28x28
@register_net
class STCO28(Sequential):
    def __init__(self, **kwargs):
        self._outnum = kwargs.pop('outnum', 1)
        self._numhid = kwargs.pop('numhid', 64)
        self._outact = kwargs.pop('outact', None)
        super().__init__(**kwargs)
        with self.name_scope():
            self.add(Conv2DTranspose(self._numhid * 4, 4, strides=1, padding=0, use_bias=False, isinput=True))
            self.add(BatchNorm())
            self.add(Activation('relu'))
            # _numhid x 4 x 4

            self.add(Conv2DTranspose(self._numhid * 2, 4, strides=1, padding=0, use_bias=False))
            self.add(BatchNorm())
            self.add(Activation('relu'))
            # _numhid x 7 x 7

            self.add(Conv2DTranspose(self._numhid, 4, strides=2, padding=1, use_bias=False))
            self.add(BatchNorm())
            self.add(Activation('relu'))
            # _numhid x 14 x 14

            self.add(Conv2DTranspose(self._outnum, 4, strides=2, padding=1, use_bias=False, activation=self._outact))
            # self.add(Activation(self._outact))
            # _numhid x 28 x 28

# # Y-shaped Sequential Transposed Convolutional-only 28x28
# @register_net
# class YTCO28(YSequential):
#     def __init__(self, **kwargs):
#         self._outnum = kwargs.pop('outnum', 1)
#         self._numhid = kwargs.pop('numhid', 64)
#         self._outact = kwargs.pop('outact', None)
#         super().__init__(**kwargs)
#         with self.name_scope():
#
#             self.addData(Identity())
#             self.addCond(Identity())
#
#             self.add(Conv2DTranspose(self._numhid * 4, 4, strides=1, padding=0, use_bias=False, isinput=True))
#             self.add(BatchNorm())
#             self.add(Activation('relu'))
#             # _numhid x 4 x 4
#
#             self.add(Conv2DTranspose(self._numhid * 2, 4, strides=1, padding=0, use_bias=False))
#             self.add(BatchNorm())
#             self.add(Activation('relu'))
#             # _numhid x 7 x 7
#
#             self.add(Conv2DTranspose(self._numhid, 4, strides=2, padding=1, use_bias=False))
#             self.add(BatchNorm())
#             self.add(Activation('relu'))
#             # _numhid x 14 x 14
#
#             self.add(Conv2DTranspose(self._outnum, 4, strides=2, padding=1, use_bias=False, activation=self._outact))
#             # self.add(Activation(self._outact))
#             # _numhid x 28 x 28
#
# # Y-shaped Sequential Convolutional/Fully-Connected 28x28
# @register_net
# class YCNFC28(YSequential):
#     def __init__(self, **kwargs):
#         self._outnum = kwargs.pop('outnum', 1)
#         self._numhid = kwargs.pop('numhid', 64)
#         self._outact = kwargs.pop('outact', None)
#         super().__init__(**kwargs)
#         with self.name_scope():
#             # _numhid x 28 x 28
#             self.addData(Conv2D(self._numhid, 4, strides=2, padding=1, use_bias=False))
#             self.addData(BatchNorm())
#             self.addData(LeakyReLU(0.2))
#             # self.add(Activation('relu'))
#             # _numhid x 14 x 14
#
#             self.addData(Conv2D(self._numhid * 2, 4, strides=2, padding=1, use_bias=False))
#             self.addData(BatchNorm())
#             self.addData(LeakyReLU(0.2))
#             # self.add(Activation('relu'))
#             # _numhid x 7 x 7
#
#             self.addData(Conv2D(self._numhid * 4, 4, strides=1, padding=0, use_bias=False))
#             self.addData(BatchNorm())
#             self.addData(LeakyReLU(0.2))
#             # self.add(Activation('relu'))
#             # _numhid x 4 x 4
#
#             self.addData(Conv2D(self._numhid * 8, 4, strides=1, padding=0, use_bias=False))
#             # _outnum x 1 x 1
#
#             self.addCond(Identity())
#
#             self.add(Dense(self._numhid, activation='relu'))
#             self.add(Dense(self._outnum, activation=self._outact))

# Sequential Convolutional/Fully-Connected 28x28
@register_net
class SCNFC28(Sequential):
    def __init__(self, **kwargs):
        self._outnum = kwargs.pop('outnum', 1)
        self._numhid = kwargs.pop('numhid', 64)
        self._outact = kwargs.pop('outact', None)
        super().__init__(**kwargs)
        with self.name_scope():
            # _numhid x 28 x 28
            self.add(Conv2D(self._numhid, 4, strides=2, padding=1, use_bias=False))
            self.add(BatchNorm())
            self.add(LeakyReLU(0.2))
            # self.add(Activation('relu'))
            # _numhid x 14 x 14

            self.add(Conv2D(self._numhid * 2, 4, strides=2, padding=1, use_bias=False))
            self.add(BatchNorm())
            self.add(LeakyReLU(0.2))
            # self.add(Activation('relu'))
            # _numhid x 7 x 7

            self.add(Conv2D(self._numhid * 4, 4, strides=1, padding=0, use_bias=False))
            self.add(BatchNorm())
            self.add(LeakyReLU(0.2))
            # self.add(Activation('relu'))
            # _numhid x 4 x 4

            self.add(Conv2D(self._numhid * 8, 4, strides=1, padding=0, use_bias=False))
            # _outnum x 1 x 1

            self.add(Dense(self._numhid, activation='relu'))
            self.add(Dense(self._outnum, activation=self._outact))

# Sequential Convolutional-only 28x28
@register_net
class SCO28(Sequential):
    def __init__(self, **kwargs):
        self._outnum = kwargs.pop('outnum', 1)
        self._numhid = kwargs.pop('numhid', 64)
        self._outact = kwargs.pop('outact', None)
        super().__init__(**kwargs)
        with self.name_scope():
            # _numhid x 28 x 28
            self.add(Conv2D(self._numhid, 4, strides=2, padding=1, use_bias=False))
            self.add(BatchNorm())
            self.add(LeakyReLU(0.2))
            # self.add(Activation('relu'))
            # _numhid x 14 x 14

            self.add(Conv2D(self._numhid * 2, 4, strides=2, padding=1, use_bias=False))
            self.add(BatchNorm())
            self.add(LeakyReLU(0.2))
            # self.add(Activation('relu'))
            # _numhid x 7 x 7

            self.add(Conv2D(self._numhid * 4, 4, strides=1, padding=0, use_bias=False))
            self.add(BatchNorm())
            self.add(LeakyReLU(0.2))
            # self.add(Activation('relu'))
            # _numhid x 4 x 4

            self.add(Conv2D(self._outnum, 4, strides=1, padding=0, use_bias=False))
            # _outnum x 1 x 1


# @register_net
# class STCN(Sequential):
#     def __init__(self, **kwargs):
#         self._outnum = kwargs.pop('outnum', 1)
#         self._numhid = kwargs.pop('numhid', 64)
#         self._outact = kwargs.pop('outact', None)
#         super().__init__(**kwargs)
#         with self.name_scope():
#             self.addData(Conv2DTranspose(self._numhid * 8, 4, strides=1, padding=0, use_bias=False, isinput=True))
#             self.addData(Activation('relu'))
#             # _numhid x 4 x 4
#
#             self.addData(Conv2DTranspose(self._numhid * 4, 4, strides=2, padding=1, use_bias=False))
#             self.addData(Activation('relu'))
#             # _numhid x 8 x 8
#
#             self.addData(Conv2DTranspose(self._numhid * 2, 4, strides=2, padding=1, use_bias=False))
#             self.addData(Activation('relu'))
#             # _numhid x 16 x 16
#
#             self.addData(Conv2DTranspose(self._numhid, 4, strides=2, padding=1, use_bias=False))
#             self.addData(Activation('relu'))
#             # _numhid x 32 x 32
#
#             self.addData(Conv2DTranspose(self._outnum, 4, strides=2, padding=1, use_bias=False))
#             self.addData(Activation(self._outact))
#             # _outnum x 64 x 64


# @register_net
# class GenFC(nn.Sequential):
#     def __init__(self, **kwargs):
#         self._outnum = kwargs.pop('outnum', 784)
#         self._outact = kwargs.pop('outact', 'tanh')
#         super().__init__(**kwargs)
#         with self.name_scope():
#             self.add(nn.Dense(256))
#             self.add(BatchNorm(axis=1, center=True, scale=True))
#             self.add(LeakyReLU(0.01))
#             self.add(nn.Dropout(.5))
#             self.add(nn.Dense(256))
#             self.add(BatchNorm(axis=1, center=True, scale=True))
#             self.add(LeakyReLU(0.01))
#             self.add(nn.Dropout(.5))
#             self.add(nn.Dense(self._outnum, activation=self._outact))
#
#
# @register_net
# class DiscrFC(nn.Sequential):
#     def __init__(self, **kwargs):
#         self._outnum = kwargs.pop('outnum', 1)
#         self._outact = kwargs.pop('outact', None)
#         super().__init__(**kwargs)
#         with self.name_scope():
#             self.add(nn.Dense(64, activation='relu'))
#             self.add(BatchNorm(axis=1, center=True, scale=True))
#             self.add(LeakyReLU(0.01))
#             self.add(nn.Dropout(.5))
#             self.add(nn.Dense(64))
#             self.add(BatchNorm(axis=1, center=True, scale=True))
#             self.add(LeakyReLU(0.01))
#             self.add(nn.Dropout(.5))
#             self.add(nn.Dense(self._outnum, activation=self._outact))
#
# @register_net
# class GSFC(nn.Sequential):
#     def __init__(self, **kwargs):
#         self._outnum = kwargs.pop('outnum', 784)
#         self._outact = kwargs.pop('outact', 'tanh')
#         super().__init__(**kwargs)
#         with self.name_scope():
#             self.add(nn.Dense(64))
#             self.add(LeakyReLU(0.01))
#             self.add(nn.Dense(64))
#             self.add(LeakyReLU(0.01))
#             self.add(nn.Dense(self._outnum, activation=self._outact))
#
# @register_net
# class DSFC(nn.Sequential):
#     def __init__(self, **kwargs):
#         self._outnum = kwargs.pop('outnum', 1)
#         self._outact = kwargs.pop('outact', None)
#         super().__init__(**kwargs)
#         with self.name_scope():
#             self.add(nn.Dense(64))
#             self.add(LeakyReLU(0.01))
#             self.add(nn.Dense(64))
#             self.add(LeakyReLU(0.01))
#             self.add(nn.Dense(64))
#             self.add(LeakyReLU(0.01))
#             self.add(nn.Dense(self._outnum, activation=self._outact))
#
# @register_net
# class GPFC(Sequential):
#     def __init__(self, **kwargs):
#         self._outnum = kwargs.pop('outnum', 784)
#         self._outact = kwargs.pop('outact', 'relu')
#         super().__init__(**kwargs)
#         with self.name_scope():
#             self.add(Dense(64, activation='relu', isinput=True))
#             self.add(Dense(64, activation='relu'))
#             self.add(Dense(64, activation='relu'))
#             self.add(Dense(self._outnum, activation=self._outact))
#
# @register_net
# class DPFC(Sequential):
#     def __init__(self, **kwargs):
#         self._outnum = kwargs.pop('outnum', 1)
#         self._outact = kwargs.pop('outact', None)
#         super().__init__(**kwargs)
#         with self.name_scope():
#             self.add(Dense(64, activation='relu'))
#             self.add(Dense(64, activation='relu'))
#             self.add(Dense(64, activation='relu'))
#             self.add(Dense(self._outnum, activation=self._outact))
#
# @register_net
# class ClassFC(nn.Sequential):
#     def __init__(self, **kwargs):
#         self._outnum = kwargs.pop('outnum', 10)
#         self._outact = kwargs.pop('outact', None)
#         super().__init__(**kwargs)
#         with self.name_scope():
#             self.add(nn.Dense(64))
#             self.add(BatchNorm(axis=1, center=True, scale=True))
#             self.add(LeakyReLU(0.01))
#             # self.add(nn.Dropout(.5))
#             self.add(nn.Dense(64))
#             self.add(BatchNorm(axis=1, center=True, scale=True))
#             self.add(LeakyReLU(0.01))
#             self.add(nn.Dropout(.5))
#             self.add(nn.Dense(self._outnum, activation=self._outact))
#
# @register_net
# class CPFC(Sequential):
#     def __init__(self, **kwargs):
#         self._outnum = kwargs.pop('outnum', 10)
#         self._outact = kwargs.pop('outact', None)
#         super().__init__(**kwargs)
#         with self.name_scope():
#             self.add(Dense(64, activation='relu'))
#             self.add(Dense(64, activation='relu'))
#             self.add(Dense(64, activation='relu'))
#             self.add(Dense(self._outnum, activation=self._outact))
#
# @register_net
# class CGPFC(YSequential):
#     def __init__(self, **kwargs):
#         self._outnum = kwargs.pop('outnum', 784)
#         self._numhid = kwargs.pop('numhid', 64)
#         self._outact = kwargs.pop('outact', None)
#         super().__init__(**kwargs)
#         with self.name_scope():
#             self.addData(Dense(self._numhid, activation='relu', isinput=True))
#
#             self.addCond(Dense(self._numhid, activation='relu'))
#
#             self.add(Dense(self._numhid, activation='relu'))
#             self.add(Dense(self._numhid, activation='relu'))
#             self.add(Dense(self._outnum, activation=self._outact))
#
#
# @register_net
# class CDPFC(YSequential):
#     def __init__(self, **kwargs):
#         self._outnum = kwargs.pop('outnum', 1)
#         self._numhid = kwargs.pop('numhid', 64)
#         self._outact = kwargs.pop('outact', None)
#         super().__init__(**kwargs)
#         with self.name_scope():
#             self.addData(Dense(self._numhid, activation='relu', isinput=True))
#
#             self.addCond(Dense(self._numhid, activation='relu'))
#
#             self.add(Dense(self._numhid, activation='relu'))
#             self.add(Dense(self._numhid, activation='relu'))
#             self.add(Dense(self._outnum, activation=self._outact))

# @register_net
# class CGenFC(YSequential):
#     def __init__(self, **kwargs):
#         super().__init__(**kwargs)
#         with self.name_scope():
#             self.addData(nn.Dense(200))
#
#             self.addCond(nn.Dense(1000))
#
#             self.add(BatchNorm(axis=1, center=True, scale=True))
#             self.add(LeakyReLU(0.01))
#             self.add(nn.Dropout(.5))
#             self.add(nn.Dense(784, activation='tanh'))
#
# @register_net
# class CDiscrFC(YSequential):
#     def __init__(self, **kwargs):
#         super().__init__(**kwargs)
#         with self.name_scope():
#             self.addData(nn.Dense(256))
#
#             self.addCond(nn.Dense(64))
#
#             # self.add(BatchNorm(axis=1, center=True, scale=True))
#             self.add(LeakyReLU(0.01))
#             self.add(nn.Dropout(.5))
#             self.add(nn.Dense(256))
#             # self.add(BatchNorm(axis=1, center=True, scale=True))
#             self.add(LeakyReLU(0.01))
#             self.add(nn.Dropout(.5))
#             self.add(nn.Dense(2))


# class DiscrCNN(nn.Sequential):
#     def __init__(self, **kwargs):
#         super().__init__(**kwargs)
#         with self.name_scope():
#             self.add(nn.Conv2D(64, (5, 5)), padding=2, stride=2)
#             # self.add(BatchNorm(axis=1, center=True, scale=True))
#             self.add(LeakyReLU(0.01))
#             self.add(nn.Conv2D(128, (5, 5)), stride=(2, 2))
#             self.add(nn.Dropout(.5))
#             self.add(nn.Dense(256))
#             # self.add(BatchNorm(axis=1, center=True, scale=True))
#             self.add(LeakyReLU(0.01))
#             self.add(nn.Dropout(.5))
#             self.add(nn.Dense(2))

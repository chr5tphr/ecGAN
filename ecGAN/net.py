from mxnet import nd
from mxnet.gluon import nn
from .func import Dense,Conv2D,Conv2DTranspose,Sequential,YSequential,Identity

nets = {}
def register_net(obj):
    nets[obj.__name__] = obj
    return obj

@register_net
class SFC(Sequential):
    def __init__(self,**kwargs):
        self._outnum = kwargs.pop('outnum',1)
        self._numhid = kwargs.pop('numhid',64)
        self._outact = kwargs.pop('outact',None)
        super().__init__(**kwargs)
        with self.name_scope():
            self.add(Dense(self._numhid, activation='relu'))
            self.add(Dense(self._numhid, activation='relu'))
            self.add(Dense(self._numhid, activation='relu'))
            self.add(Dense(self._outnum, activation=self._outact))

@register_net
class YSFC(YSequential):
    def __init__(self,**kwargs):
        self._outnum = kwargs.pop('outnum',1)
        self._numhid = kwargs.pop('numhid',64)
        self._outact = kwargs.pop('outact',None)
        super().__init__(**kwargs)
        with self.name_scope():
            self.addData(Dense(self._numhid, activation='relu', isinput=True))

            self.addCond(Dense(self._numhid, activation='relu'))

            self.add(Dense(self._numhid, activation='relu'))
            self.add(Dense(self._numhid, activation='relu'))
            self.add(Dense(self._outnum, activation=self._outact))

@register_net
class YTCN28(YSequential):
    def __init__(self,**kwargs):
        self._outnum = kwargs.pop('outnum',1)
        self._numhid = kwargs.pop('numhid',64)
        self._outact = kwargs.pop('outact',None)
        super().__init__(**kwargs)
        with self.name_scope():

            self.addData(Identity())
            self.addCond(Identity())

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
class STCN28(Sequential):
    def __init__(self,**kwargs):
        self._outnum = kwargs.pop('outnum',1)
        self._numhid = kwargs.pop('numhid',64)
        self._outact = kwargs.pop('outact',None)
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
    def __init__(self,**kwargs):
        self._outnum = kwargs.pop('outnum',1)
        self._numhid = kwargs.pop('numhid',64)
        self._outact = kwargs.pop('outact',None)
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


# @register_net
# class STCN(Sequential):
#     def __init__(self,**kwargs):
#         self._outnum = kwargs.pop('outnum',1)
#         self._numhid = kwargs.pop('numhid',64)
#         self._outact = kwargs.pop('outact',None)
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
#     def __init__(self,**kwargs):
#         self._outnum = kwargs.pop('outnum',784)
#         self._outact = kwargs.pop('outact','tanh')
#         super().__init__(**kwargs)
#         with self.name_scope():
#             self.add(nn.Dense(256))
#             self.add(nn.BatchNorm(axis=1,center=True,scale=True))
#             self.add(nn.LeakyReLU(0.01))
#             self.add(nn.Dropout(.5))
#             self.add(nn.Dense(256))
#             self.add(nn.BatchNorm(axis=1,center=True,scale=True))
#             self.add(nn.LeakyReLU(0.01))
#             self.add(nn.Dropout(.5))
#             self.add(nn.Dense(self._outnum, activation=self._outact))
#
#
# @register_net
# class DiscrFC(nn.Sequential):
#     def __init__(self,**kwargs):
#         self._outnum = kwargs.pop('outnum',1)
#         self._outact = kwargs.pop('outact',None)
#         super().__init__(**kwargs)
#         with self.name_scope():
#             self.add(nn.Dense(64, activation='relu'))
#             self.add(nn.BatchNorm(axis=1,center=True,scale=True))
#             self.add(nn.LeakyReLU(0.01))
#             self.add(nn.Dropout(.5))
#             self.add(nn.Dense(64))
#             self.add(nn.BatchNorm(axis=1,center=True,scale=True))
#             self.add(nn.LeakyReLU(0.01))
#             self.add(nn.Dropout(.5))
#             self.add(nn.Dense(self._outnum, activation=self._outact))
#
# @register_net
# class GSFC(nn.Sequential):
#     def __init__(self,**kwargs):
#         self._outnum = kwargs.pop('outnum',784)
#         self._outact = kwargs.pop('outact','tanh')
#         super().__init__(**kwargs)
#         with self.name_scope():
#             self.add(nn.Dense(64))
#             self.add(nn.LeakyReLU(0.01))
#             self.add(nn.Dense(64))
#             self.add(nn.LeakyReLU(0.01))
#             self.add(nn.Dense(self._outnum, activation=self._outact))
#
# @register_net
# class DSFC(nn.Sequential):
#     def __init__(self,**kwargs):
#         self._outnum = kwargs.pop('outnum',1)
#         self._outact = kwargs.pop('outact',None)
#         super().__init__(**kwargs)
#         with self.name_scope():
#             self.add(nn.Dense(64))
#             self.add(nn.LeakyReLU(0.01))
#             self.add(nn.Dense(64))
#             self.add(nn.LeakyReLU(0.01))
#             self.add(nn.Dense(64))
#             self.add(nn.LeakyReLU(0.01))
#             self.add(nn.Dense(self._outnum, activation=self._outact))
#
# @register_net
# class GPFC(Sequential):
#     def __init__(self,**kwargs):
#         self._outnum = kwargs.pop('outnum',784)
#         self._outact = kwargs.pop('outact','relu')
#         super().__init__(**kwargs)
#         with self.name_scope():
#             self.add(Dense(64, activation='relu', isinput=True))
#             self.add(Dense(64, activation='relu'))
#             self.add(Dense(64, activation='relu'))
#             self.add(Dense(self._outnum, activation=self._outact))
#
# @register_net
# class DPFC(Sequential):
#     def __init__(self,**kwargs):
#         self._outnum = kwargs.pop('outnum',1)
#         self._outact = kwargs.pop('outact',None)
#         super().__init__(**kwargs)
#         with self.name_scope():
#             self.add(Dense(64, activation='relu'))
#             self.add(Dense(64, activation='relu'))
#             self.add(Dense(64, activation='relu'))
#             self.add(Dense(self._outnum, activation=self._outact))
#
# @register_net
# class ClassFC(nn.Sequential):
#     def __init__(self,**kwargs):
#         self._outnum = kwargs.pop('outnum',10)
#         self._outact = kwargs.pop('outact',None)
#         super().__init__(**kwargs)
#         with self.name_scope():
#             self.add(nn.Dense(64))
#             self.add(nn.BatchNorm(axis=1,center=True,scale=True))
#             self.add(nn.LeakyReLU(0.01))
#             # self.add(nn.Dropout(.5))
#             self.add(nn.Dense(64))
#             self.add(nn.BatchNorm(axis=1,center=True,scale=True))
#             self.add(nn.LeakyReLU(0.01))
#             self.add(nn.Dropout(.5))
#             self.add(nn.Dense(self._outnum, activation=self._outact))
#
# @register_net
# class CPFC(Sequential):
#     def __init__(self,**kwargs):
#         self._outnum = kwargs.pop('outnum',10)
#         self._outact = kwargs.pop('outact',None)
#         super().__init__(**kwargs)
#         with self.name_scope():
#             self.add(Dense(64, activation='relu'))
#             self.add(Dense(64, activation='relu'))
#             self.add(Dense(64, activation='relu'))
#             self.add(Dense(self._outnum, activation=self._outact))
#
@register_net
class CGPFC(YSequential):
    def __init__(self,**kwargs):
        self._outnum = kwargs.pop('outnum',784)
        self._numhid = kwargs.pop('numhid',64)
        self._outact = kwargs.pop('outact',None)
        super().__init__(**kwargs)
        with self.name_scope():
            self.addData(Dense(self._numhid, activation='relu', isinput=True))

            self.addCond(Dense(self._numhid, activation='relu'))

            self.add(Dense(self._numhid, activation='relu'))
            self.add(Dense(self._numhid, activation='relu'))
            self.add(Dense(self._outnum, activation=self._outact))


@register_net
class CDPFC(YSequential):
    def __init__(self,**kwargs):
        self._outnum = kwargs.pop('outnum',1)
        self._numhid = kwargs.pop('numhid',64)
        self._outact = kwargs.pop('outact',None)
        super().__init__(**kwargs)
        with self.name_scope():
            self.addData(Dense(self._numhid, activation='relu', isinput=True))

            self.addCond(Dense(self._numhid, activation='relu'))

            self.add(Dense(self._numhid, activation='relu'))
            self.add(Dense(self._numhid, activation='relu'))
            self.add(Dense(self._outnum, activation=self._outact))

# @register_net
# class CGenFC(YSequential):
#     def __init__(self,**kwargs):
#         super().__init__(**kwargs)
#         with self.name_scope():
#             self.addData(nn.Dense(200))
#
#             self.addCond(nn.Dense(1000))
#
#             self.add(nn.BatchNorm(axis=1,center=True,scale=True))
#             self.add(nn.LeakyReLU(0.01))
#             self.add(nn.Dropout(.5))
#             self.add(nn.Dense(784,activation='tanh'))
#
# @register_net
# class CDiscrFC(YSequential):
#     def __init__(self,**kwargs):
#         super().__init__(**kwargs)
#         with self.name_scope():
#             self.addData(nn.Dense(256))
#
#             self.addCond(nn.Dense(64))
#
#             # self.add(nn.BatchNorm(axis=1,center=True,scale=True))
#             self.add(nn.LeakyReLU(0.01))
#             self.add(nn.Dropout(.5))
#             self.add(nn.Dense(256))
#             # self.add(nn.BatchNorm(axis=1,center=True,scale=True))
#             self.add(nn.LeakyReLU(0.01))
#             self.add(nn.Dropout(.5))
#             self.add(nn.Dense(2))


# class DiscrCNN(nn.Sequential):
#     def __init__(self,**kwargs):
#         super().__init__(**kwargs)
#         with self.name_scope():
#             self.add(nn.Conv2D(64,(5,5)),padding=2,stride=2)
#             # self.add(nn.BatchNorm(axis=1,center=True,scale=True))
#             self.add(nn.LeakyReLU(0.01))
#             self.add(nn.Conv2D(128,(5,5)),stride=(2,2))
#             self.add(nn.Dropout(.5))
#             self.add(nn.Dense(256))
#             # self.add(nn.BatchNorm(axis=1,center=True,scale=True))
#             self.add(nn.LeakyReLU(0.01))
#             self.add(nn.Dropout(.5))
#             self.add(nn.Dense(2))

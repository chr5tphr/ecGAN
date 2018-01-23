from mxnet import nd
from mxnet.gluon import nn
from .func import Dense,Sequential,YSequential

nets = {}
def register_net(obj):
    nets[obj.__name__] = obj
    return obj

@register_net
class GenFC(nn.Sequential):
    def __init__(self,**kwargs):
        self._outnum = kwargs.pop('outnum',784)
        self._outact = kwargs.pop('outact','tanh')
        super().__init__(**kwargs)
        with self.name_scope():
            self.add(nn.Dense(256))
            self.add(nn.BatchNorm(axis=1,center=True,scale=True))
            self.add(nn.LeakyReLU(0.01))
            self.add(nn.Dropout(.5))
            self.add(nn.Dense(256))
            self.add(nn.BatchNorm(axis=1,center=True,scale=True))
            self.add(nn.LeakyReLU(0.01))
            self.add(nn.Dropout(.5))
            self.add(nn.Dense(self._outnum, activation=self._outact))


@register_net
class DiscrFC(nn.Sequential):
    def __init__(self,**kwargs):
        self._outnum = kwargs.pop('outnum',1)
        self._outact = kwargs.pop('outact',None)
        super().__init__(**kwargs)
        with self.name_scope():
            self.add(nn.Dense(64, activation='relu'))
            self.add(nn.BatchNorm(axis=1,center=True,scale=True))
            self.add(nn.LeakyReLU(0.01))
            self.add(nn.Dropout(.5))
            self.add(nn.Dense(64))
            self.add(nn.BatchNorm(axis=1,center=True,scale=True))
            self.add(nn.LeakyReLU(0.01))
            self.add(nn.Dropout(.5))
            self.add(nn.Dense(self._outnum, activation=self._outact))

@register_net
class GSFC(nn.Sequential):
    def __init__(self,**kwargs):
        self._outnum = kwargs.pop('outnum',784)
        self._outact = kwargs.pop('outact','tanh')
        super().__init__(**kwargs)
        with self.name_scope():
            self.add(nn.Dense(64))
            self.add(nn.LeakyReLU(0.01))
            self.add(nn.Dense(64))
            self.add(nn.LeakyReLU(0.01))
            self.add(nn.Dense(self._outnum, activation=self._outact))

@register_net
class DSFC(nn.Sequential):
    def __init__(self,**kwargs):
        self._outnum = kwargs.pop('outnum',1)
        self._outact = kwargs.pop('outact',None)
        super().__init__(**kwargs)
        with self.name_scope():
            self.add(nn.Dense(64))
            self.add(nn.LeakyReLU(0.01))
            self.add(nn.Dense(64))
            self.add(nn.LeakyReLU(0.01))
            self.add(nn.Dense(64))
            self.add(nn.LeakyReLU(0.01))
            self.add(nn.Dense(self._outnum, activation=self._outact))

@register_net
class GPFC(Sequential):
    def __init__(self,**kwargs):
        self._outnum = kwargs.pop('outnum',784)
        self._outact = kwargs.pop('outact','relu')
        super().__init__(**kwargs)
        with self.name_scope():
            self.add(Dense(64, activation='relu', isinput=True))
            self.add(Dense(64, activation='relu'))
            self.add(Dense(64, activation='relu'))
            self.add(Dense(self._outnum, activation=self._outact))

@register_net
class DPFC(Sequential):
    def __init__(self,**kwargs):
        self._outnum = kwargs.pop('outnum',1)
        self._outact = kwargs.pop('outact',None)
        super().__init__(**kwargs)
        with self.name_scope():
            self.add(Dense(64, activation='relu'))
            self.add(Dense(64, activation='relu'))
            self.add(Dense(64, activation='relu'))
            self.add(Dense(self._outnum, activation=self._outact))

@register_net
class ClassFC(nn.Sequential):
    def __init__(self,**kwargs):
        self._outnum = kwargs.pop('outnum',10)
        self._outact = kwargs.pop('outact',None)
        super().__init__(**kwargs)
        with self.name_scope():
            self.add(nn.Dense(64))
            self.add(nn.BatchNorm(axis=1,center=True,scale=True))
            self.add(nn.LeakyReLU(0.01))
            # self.add(nn.Dropout(.5))
            self.add(nn.Dense(64))
            self.add(nn.BatchNorm(axis=1,center=True,scale=True))
            self.add(nn.LeakyReLU(0.01))
            self.add(nn.Dropout(.5))
            self.add(nn.Dense(self._outnum, activation=self._outact))

@register_net
class CPFC(Sequential):
    def __init__(self,**kwargs):
        self._outnum = kwargs.pop('outnum',10)
        self._outact = kwargs.pop('outact',None)
        super().__init__(**kwargs)
        with self.name_scope():
            self.add(Dense(64, activation='relu'))
            self.add(Dense(64, activation='relu'))
            self.add(Dense(64, activation='relu'))
            self.add(Dense(self._outnum, activation=self._outact))

@register_net
class CGPFC(YSequential):
    def __init__(self,**kwargs):
        self._outnum = kwargs.pop('outnum',784)
        self._outact = kwargs.pop('outact','relu')
        super().__init__(**kwargs)
        with self.name_scope():
            self.addData(Dense(64, activation='relu', isinput=True))

            self.addCond(Dense(64, activation='relu'))

            self.add(Dense(64, activation='relu'))
            self.add(Dense(64, activation='relu'))
            self.add(Dense(self._outnum, activation=self._outact))

@register_net
class CDPFC(YSequential):
    def __init__(self,**kwargs):
        self._outnum = kwargs.pop('outnum',1)
        self._outact = kwargs.pop('outact',None)
        super().__init__(**kwargs)
        with self.name_scope():
            self.addData(Dense(64, activation='relu', isinput=True))

            self.addCond(Dense(64, activation='relu'))

            self.add(Dense(64, activation='relu'))
            self.add(Dense(64, activation='relu'))
            self.add(Dense(self._outnum, activation=self._outact))

@register_net
class CGenFC(YSequential):
    def __init__(self,**kwargs):
        super().__init__(**kwargs)
        with self.name_scope():
            self.addData(nn.Dense(200))

            self.addCond(nn.Dense(1000))

            self.add(nn.BatchNorm(axis=1,center=True,scale=True))
            self.add(nn.LeakyReLU(0.01))
            self.add(nn.Dropout(.5))
            self.add(nn.Dense(784,activation='tanh'))

@register_net
class CDiscrFC(YSequential):
    def __init__(self,**kwargs):
        super().__init__(**kwargs)
        with self.name_scope():
            self.addData(nn.Dense(256))

            self.addCond(nn.Dense(64))

            # self.add(nn.BatchNorm(axis=1,center=True,scale=True))
            self.add(nn.LeakyReLU(0.01))
            self.add(nn.Dropout(.5))
            self.add(nn.Dense(256))
            # self.add(nn.BatchNorm(axis=1,center=True,scale=True))
            self.add(nn.LeakyReLU(0.01))
            self.add(nn.Dropout(.5))
            self.add(nn.Dense(2))


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

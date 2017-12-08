from mxnet import nd
from mxnet.gluon import nn

nets = {}
def register_net(obj):
    nets[obj.__name__] = obj
    return obj

@register_net
class GenFC(nn.Sequential):
    def __init__(self,**kwargs):
        super().__init__(**kwargs)
        with self.name_scope():
            # self.add(nn.Dense(1024))
            # self.add(nn.BatchNorm(axis=1,center=True,scale=True))
            # self.add(nn.LeakyReLU(0.01))
            # self.add(nn.Dropout(.5))
            self.add(nn.Dense(64))
            self.add(nn.BatchNorm(axis=1,center=True,scale=True))
            self.add(nn.LeakyReLU(0.01))
            self.add(nn.Dropout(.5))
            self.add(nn.Dense(784,activation='tanh'))

@register_net
class DiscrFC(nn.Sequential):
    def __init__(self,**kwargs):
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
            self.add(nn.Dense(2))

@register_net
class ClassFC(nn.Sequential):
    def __init__(self,**kwargs):
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
            self.add(nn.Dense(10))

# @register_net
# class ClassMO(nn.Sequential):
#     def __init__(self,**kwargs):
#         super().__init__(**kwargs)
#         with self.name_scope():
#             self.add(nn.Dense(256))
#             # self.add(nn.BatchNorm(axis=1,center=True,scale=True))
#             self.add(nn.LeakyReLU(0.01))
#             self.add(nn.Dropout(.5))
#             self.add(nn.Dense(256))
#             # self.add(nn.BatchNorm(axis=1,center=True,scale=True))
#             self.add(nn.LeakyReLU(0.01))
#             self.add(nn.Dropout(.5))
#             self.add(nn.Dense(10))

class YSequential(nn.Block):
    def __init__(self,**kwargs):
        super().__init__(**kwargs)
        with self.name_scope():
            self._data = nn.Sequential()
            self.register_child(self._data)

            self._cond = nn.Sequential()
            self.register_child(self._cond)

            self._main = nn.Sequential()
            self.register_child(self._main)

    def addData(self,*args,**kwargs):
        with self._data.name_scope():
            self._data.add(*args,**kwargs)

    def addCond(self,*args,**kwargs):
        with self._cond.name_scope():
            self._cond.add(*args,**kwargs)

    def add(self,*args,**kwargs):
        with self._main.name_scope():
            self._main.add(*args,**kwargs)

    def forward(self,x,y,dim=1):
        data = self._data(x)
        cond = self._cond(y)
        combo = nd.concat(data,cond,dim=dim)
        return self._main(combo)

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

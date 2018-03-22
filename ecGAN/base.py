from mxnet.gluon import nn
from mxnet import nd
class Block(nn.Block):
    def forward_logged(self, *args, **kwargs):
        self._in = args
        self._out = self.forward(*args)
        return self._out

class Intermediate(Block):
    def forward(self, *args, depth=-1):
        return self.forward(self, *args)

class YSequentialBase(Block):
    def __init__(self, **kwargs):
        self._concat_dim = kwargs.pop('concat_dim', 1)
        super().__init__(**kwargs)
        with self.name_scope():
            self._data_net = Sequential()
            self.register_child(self._data_net)

            self._cond_net = Sequential()
            self.register_child(self._cond_net)

            self._main_net = Sequential()
            self.register_child(self._main_net)

    def addData(self, *args, **kwargs):
        with self._data_net.name_scope():
            self._data_net.add(*args, **kwargs)

    def addCond(self, *args, **kwargs):
        with self._cond_net.name_scope():
            self._cond_net.add(*args, **kwargs)

    def add(self, *args, **kwargs):
        with self._main_net.name_scope():
            self._main_net.add(*args, **kwargs)

    def forward(self, x, y):
        data = self._data_net(x)
        cond = self._cond_net(y)
        combo = nd.concat(data, cond, dim=self._concat_dim)
        return self._main_net.forward(combo)

class ReLUBase(Block):
    def forward(self, x):
        return nd.maximum(0., x)

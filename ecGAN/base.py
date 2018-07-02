from mxnet.gluon import nn
from mxnet import nd

class Block(nn.Block):
    def forward_logged(self, *args, **kwargs):
        self._in = args
        self._out = self.forward(*args, **kwargs)
        return self._out

class Intermediate(Block):
    def forward(self, *args, depth=-1):
        return self.forward(self, *args)

class SequentialBase(Block, nn.Sequential):
    def forward_logged(self, x):
        self._in = x
        for child in self._children.values():
            x = child.forward_logged(x)
        self._out = x
        return x

class ParallelBase(Block):
    def forward(self, X):
        Y = []
        for child, x in zip(self._children.values(), X):
            Y.append(child.forward(x))
        return Y

    def forward_logged(self, X):
        self._in = X
        Y = []
        for child, x in zip(self._children.values(), X):
            Y.append(child.forward_logged(x))
        self._out = Y
        return Y

class ConcatBase(Block):
    def __init__(self, **kwargs):
        self._concat_dim = kwargs.pop('concat_dim', 1)
        super().__init__(**kwargs)

    def forward(self, X):
        return nd.concat(*X, dim=self._concat_dim)


class YSequentialBase(Block):
    _Subnet = None
    def __init__(self, **kwargs):
        self._concat_dim = kwargs.pop('concat_dim', 1)
        super().__init__(**kwargs)
        with self.name_scope():
            self._data_net = self._Subnet()
            self.register_child(self._data_net)

            self._cond_net = self._Subnet()
            self.register_child(self._cond_net)

            self._main_net = self._Subnet()
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

    def forward(self, *args):
        # WARNING This is hacky and nowhere standardized
        if len(args) == 1:
            x,y = args[0]
        else:
            x,y = args

        data = self._data_net(x)
        cond = self._cond_net(y)
        combo = nd.concat(data, cond, dim=self._concat_dim)
        return self._main_net.forward(combo)

    def forward_logged(self, x, y):
        self._in = [x, y]

        data = self._data_net.forward_logged(x)
        cond = self._cond_net.forward_logged(y)
        combo = nd.concat(data, cond, dim=self._concat_dim)
        self._out = self._main_net.forward_logged(combo)
        return self._out

class ReLUBase(Block):
    def forward(self, x):
        return nd.maximum(0., x)

class IdentityBase(Block):
    def forward(self, *args, **kwargs):
        return args[0]

class TanhBase(Block):
    def forward(self, x):
        return nd.tanh(x)


from mxnet.gluon import nn

class Block(nn.Block):
    def forward_logged(self, *args, **kwargs):
        self._in = args
        self._out = self.forward(*args)
        return self._out

class Intermediate(Block):
    def forward(self, *args, depth=-1):
        return self.forward(self, *args)


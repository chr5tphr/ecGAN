from mxnet import nd, autograd

from ...func import Mlist
from ...base import Block

class GradBasedExplainable(Block)
    def relevance_sensitivity(self, data, out=None, **kwargs):
        data = Mlist(data)
        data.attach_grad()
        with autograd.record():
            y = self.forward(data)
        y.backward(out_grad=out)
        # WARNING: is hacky and sucks
        self._out = y
        return data.grad

    def relevance_intgrads(self, data, out=None, num=50, base=None, **kwargs):
        if isinstance(data, nd.NDArray):
            data = [data]
        if base is None:
            base = [x.zeros_like() for x in data]

        alpha = linspace(0., 1., num, ctx=data[0].context)
        diff = [x - y for x, y in zip(data, base)]

        res = [x.zeros_like() for x in data]
        for a in alpha:
            ddat = [ba + a * di for ba, di in zip(base, diff)]
            ret = self.relevance_sensitivity(data=ddat, out=out)
            for tar, val in zip(res, ret):
                tar += val

        return res


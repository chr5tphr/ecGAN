from mxnet import nd, gluon, autograd
from mxnet.gluon import nn

from .func import Interpretable


class Dense(Interpretable,nn.Dense):
    def relevance_sensitivity(self,R):
        if self._in is None:
            raise RuntimeError('Block has not yet executed forward_logged!')
        a = self._in[0]
        a.attach_grad()
        with autograd.record():
            z = self(a)
        return autograd.grad(z,a,head_grads=R)

    def relevance_dtd(self,R,lo=-1,hi=1):
        if self._in is None:
            raise RuntimeError('Block has not yet executed forward_logged!')
        a = self._in[0]
        if self._isinput: #zb
            wplus = nd.maximum(0.,self.weight.data())
            wminus = nd.minimum(0.,self.weight.data())
            upper = nd.ones_like(a)*hi
            lower = nd.ones_like(a)*lo
            a.attach_grad()
            upper.attach_grad()
            lower.attach_grad()
            with autograd.record():
                zlh = (  self(a)
                       - nd.FullyConnected(lower,wplus,None,no_bias=True,num_hidden=self._units,flatten=self._flatten)
                       - nd.FullyConnected(upper,wminus,None,no_bias=True,num_hidden=self._units,flatten=self._flatten) )
            zlh.backward(out_grad=R/zlh)
            return a*a.grad + upper*upper.grad + lower*lower.grad
        else: #z+
            wplus = nd.maximum(0.,self.weight.data())
            a.attach_grad()
            with autograd.record():
                z = nd.FullyConnected(a,wplus,None,no_bias=True,num_hidden=self._units,flatten=self._flatten)
                if self.act is not None:
                    z = self.act(z)
            c = autograd.grad(z,a,head_grads=R/z)
            return a*c

class Identity(Interpretable, nn.Block):
    def forward(self,*args,**kwargs):
        return args[0]
    def relevance(self,*args,**kwargs):
        return args[0]

class Clip(Interpretable,nn.Block):
    def forward(self,x):
        return nd.clip(x,0.,1.)
    def relevance(self,R):
        return R

class Conv2D(Interpretable, nn.Conv2D):
    pass

class Conv2DTranspose(Interpretable, nn.Conv2DTranspose):
    pass

class BatchNorm(Interpretable, nn.BatchNorm):
    pass

class LeakyReLU(Interpretable, nn.BatchNorm):
    pass

class Activation(Interpretable, nn.Activation):
    pass

class MaxOut(nn.Block):
    pass

import mxnet as mx

from mxnet import nd, gluon, autograd
from mxnet.gluon import nn


def fuzzy_one_hot(x,size):
    return nd.where(nd.one_hot(x,size),
                    nd.uniform(low=0.7,high=1.2,shape=(x.shape[0],size),ctx=x.context),
                    nd.uniform(low=0.0,high=0.3,shape=(x.shape[0],size),ctx=x.context))


class Interpretable(object):
    def __init__(self,*args,**kwargs):
        self._isinput = kwargs.pop('isinput',False)
        super().__init__(*args,**kwargs)

    def relevance(self,*args,method='dtd',**kwargs):
        func = getattr(self,'relevance_'+method)
        return func(*args,**kwargs)

    def relevance_sensitivity(self):
        # result = []
        # for param in self.collect_params().items():
        #     result.append()
        # return self.collect_params()
        raise NotImplementedError

    def relevance_dtd(self,a,R):
        raise NotImplementedError

class Dense(Interpretable,nn.Dense):
    def relevance_sensitivity(self,a,R):
        a.attach_grad()
        with autograd.record():
            z = self(a)
        return autograd.grad(z,a,head_grads=R)

    def relevance_dtd(self,a,R,lo=-1,hi=1):
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

class Sequential(Interpretable,nn.Sequential):
    def relevance(self,x,y=None,method='dtd',ret_all=False):
        A = [x]
        for child in self._children:
            A.append(child.forward(A[-1]))
        z = A.pop()
        R = [z if y is None else y]
        for child,a in zip(self._children[::-1],A[::-1]):
            R.append(child.relevance(a,R[-1],method=method))
        return R if ret_all else R[-1]

class MaxOut(nn.Block):
    pass

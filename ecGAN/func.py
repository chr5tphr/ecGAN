import mxnet as mx

from mxnet import nd, gluon, autograd
from mxnet.gluon import nn


def fuzzy_one_hot(arr,size):
    x = arr.reshape((-1,))
    return nd.where(nd.one_hot(x,size),
                    nd.uniform(low=0.7,high=1.2,shape=(x.shape[0],size),ctx=x.context),
                    nd.uniform(low=0.0,high=0.3,shape=(x.shape[0],size),ctx=x.context))

def linspace(start=0.,stop=1.,num=1,ctx=None,dtype=None):
    return nd.arange(start,stop,(stop-start)/num,ctx=ctx,dtype='float32').astype(dtype)

def randint(low=0.,high=1.,shape=(1,),ctx=None,dtype='int32'):
    return nd.uniform(low=low,high=high,shape=shape,ctx=ctx).astype(dtype)

class Intermediate(object):
    def forward(self,*args,depth=-1):
        return self.forward(self,*args)

class Interpretable(object):
    def __init__(self,*args,**kwargs):
        self._isinput = kwargs.pop('isinput',False)
        super().__init__(*args,**kwargs)
        self._in = None
        self._out = None

    def forward_logged(self,*args,**kwargs):
        self._in = args
        self._out = self.forward(*args)
        return self._out

    def relevance(self,*args,**kwargs):
        method = kwargs.pop('method',dtd)
        func = getattr(self,'relevance_'+method)
        return func(*args,**kwargs)

    def relevance_sensitivity(self):
        raise NotImplementedError

    def relevance_dtd(self,a,R):
        raise NotImplementedError

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

class Sequential(Interpretable, Intermediate, nn.Sequential):
    # def relevance(self,x,y=None,method='dtd',ret_all=False):
    #     if self._in is None:
    #         raise RuntimeError('Block has not yet executed forward!')
    #     A = [x]
    #     for child in self._children:
    #         A.append(child.forward(A[-1]))
    #     z = A.pop()
    #     R = [z if y is None else y]
    #     for child,a in zip(self._children[::-1],A[::-1]):
    #         R.append(child.relevance(a,R[-1],method=method))
    #     return R if ret_all else R[-1]
    def relevance(self,y=None,method='dtd',ret_all=False):
        if self._in is None:
            raise RuntimeError('Block has not yet executed forward_logged!')
        R = [self._out if y is None else y]
        for child in self._children[::-1]:
            R.append(child.relevance(a,R[-1],method=method))
        return R if ret_all else R[-1]

    def forward(self, x, depth=-1):
        rdep = depth if depth > 0 else (len(self._children) + depth)
        for i,block in enumerate(self._children):
            x = block(x)
            if i == depth:
                break
        return x

class YSequential(Interpretable, Intermediate, nn.Block):
    def __init__(self,**kwargs):
        self._concat_dim = kwargs.pop('concat_dim',1)
        super().__init__(**kwargs)
        with self.name_scope():
            self._data_net = Sequential()
            self.register_child(self._data_net)

            self._cond_net = Sequential()
            self.register_child(self._cond_net)

            self._main_net = nn.Sequential()
            self.register_child(self._main_net)

    def addData(self,*args,**kwargs):
        with self._data_net.name_scope():
            self._data_net.add(*args,**kwargs)

    def addCond(self,*args,**kwargs):
        with self._cond_net.name_scope():
            self._cond_net.add(*args,**kwargs)

    def add(self,*args,**kwargs):
        with self._main_net.name_scope():
            self._main_net.add(*args,**kwargs)

    def forward(self,x,y,depth=-1):
        data = self._data_net(x)
        cond = self._cond_net(y)
        combo = nd.concat(data,cond,dim=self._concat_dim)
        return self._main_net(combo,depth=depth)

    def relevance(self,y=None,method='dtd',ret_all=False):
        if self._in is None:
            raise RuntimeError('Block has not yet executed forward_logged!')
        R = self._main_net.relevance(R,y=y,method=method,ret_all=True)

        dim_d = self._data_net._out.shape[self._concat_dim]

        Rtd = R.slice_axis(axis=self._concat_dim, begin=0, end=dim_d)
        Rtc = R.slice_axis(axis=self._concat_dim, begin=dim_d, end=-1)

        Rd = self._data_net.relevance(Rtd,method=method,ret_all=True)
        Rc = self._cond_net.relevance(Rtc,method=method,ret_all=True)

        return Rd,Rc,R if ret_all else Rd[-1],Rc[-1]

class Clip(Interpretable,nn.Block):
    def forward(self,x):
        return nd.clip(x,0.,1.)
    def relevance(self,R):
        return R

class MaxOut(nn.Block):
    pass

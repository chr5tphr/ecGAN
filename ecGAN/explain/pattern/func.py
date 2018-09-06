from mxnet import nd, autograd

class tanh(autograd.Function):
    def forward(self, x):
        y = nd.tanh(x)
        self.save_for_backward(x,y)
        return y

    def backward(self, dy):
        x, y = self.saved_tensors
        gamma = x / (y + (y==0.))
        return dy * gamma

class leaky_relu(autograd.Function):
    def __init__(self, *args, slope=0., **kwargs):
        super().__init__(*args, **kwargs)
        self._alpha = slope

    def forward(self, x):
        y = nd.LeakyReLU(x, act_type='leaky', slope=self._alpha, name='fwd')
        self.save_for_backward(x<0.)
        return y

    def backward(self, dy):
        xneg, = self.saved_tensors
        xpos = xneg == 0.
        gamma = xneg * self._alpha
        gamma = xneg * (gamma != 0.) / (gamma + (gamma == 0.)) + xpos
        return dy * gamma


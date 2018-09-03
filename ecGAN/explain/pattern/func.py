from mxnet import nd, autograd

class tanh(autograd.Function):
    def forward(self, x):
        return nd.tanh(x)

    def backward(self, dy):
        return nd.arctanh(dy)

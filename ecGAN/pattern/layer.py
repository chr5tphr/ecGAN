import mxnet as mx
from mxnet import nd, gluon, autograd
from mxnet.gluon import nn
import numpy as np

from .base import PatternNet
from ..func import im2col_indices

class DensePatternNet(PatternNet, nn.Dense):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    ##############
    ### Linear ###
    ##############

    def init_pattern_linear(self):
        self._init_pattern_linear(self.weight.shape)

    def learn_pattern_linear(self):
        if self._in is None:
            raise RuntimeError('Block has not yet executed forward_logged!')
        x = self._in[0].flatten()
        y = self._out.flatten()
        self._learn_pattern_linear(x, y)

    def forward_pattern_linear(self, *args):
        x = args[0]
        if len(args) < 2:
            x_neut = x
        else:
            x_neut = args[1]
        # omit number of samples, since present in both cov and var
        a = self.cov.data() / (self.var_y.data().T + 1e-12)
        z_neut = self.forward(x_neut)
        z = nd.FullyConnected(x, a, None, no_bias=True, num_hidden=self._units, flatten=self._flatten)

        # this depends on ReLU!
        z = nd.where(z_neut>0, z, nd.zeros_like(z))
        return z, z_neut

    def learn_assess_pattern_linear(self):
        pass
        #if self._in is None:
        #    raise RuntimeError('Block has not yet executed forward_logged!')
        #x = self._in[0].flatten()
        #y = self._out.flatten()
        #a = self.cov.data() / (self.var_y.data().T + 1e-12)
        #dtd = self.qual_dtd.data()
        #dty = self.qual_dty.data()
        #signal = nd.dot(a, y, transpose_a=True)
        #d = x - signal

        #dtd += nd.dot(d, d, transpose_a=True)
        #dty += nd.dot(d, y, transpose_a=True)

        #self.qual_dtd.set_data(dtd)
        #self.qual_dty.set_data(dty)

    def assess_pattern(self):
        return

    #####################
    ### Two-Component ###
    #####################

    def init_pattern_twocomponent(self):
        self._init_pattern_twocomponent(self.weight.shape)

    def learn_pattern_twocomponent(self):
        if self._in is None:
            raise RuntimeError('Block has not yet executed forward_logged!')
        x = self._in[0].flatten()
        y = self._out.flatten()
        self._learn_pattern_twocomponent(x, y)

    def forward_pattern_twocomponent(self, *args):
        x = args[0]
        if len(args) < 2:
            x_neut = x
        else:
            x_neut = args[1]
        cov_pos = self.mean_xy_pos.data() - nd.dot(self.mean_y.data(), self.mean_x_pos.data(), transpose_a=True)
        cov_neg = self.mean_xy_neg.data() - nd.dot(self.mean_y.data(), self.mean_x_neg.data(), transpose_a=True)
        weight = self.weight.data()
        a_pos = cov_pos / ((weight * cov_pos).sum(axis=1, keepdims=True) + 1e-12)
        a_neg = cov_neg / ((weight * cov_neg).sum(axis=1, keepdims=True) + 1e-12)

        z_neut = self.forward(x_neut)
        z_pos = nd.FullyConnected(x, a_pos, None, no_bias=True, num_hidden=self._units, flatten=self._flatten)
        z_neg = nd.FullyConnected(x, a_neg, None, no_bias=True, num_hidden=self._units, flatten=self._flatten)

        # this depends on ReLU!
        z = nd.where(z_neut>0., z_pos, z_neg)

        return z, z_neut

    def assess_pattern_twocomponent(self):
        pass

class Conv2DPatternNet(PatternNet, nn.Conv2D):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def assess_pattern(self):
        return

    ##############
    ### Linear ###
    ##############

    def init_pattern_linear(self):
        chan = self.weight.shape[0]
        ksize = np.prod(self.weight.shape[1:])
        self._init_pattern_linear((chan, ksize))

    def learn_pattern_linear(self):
        if self._in is None:
            raise RuntimeError('Block has not yet executed forward_logged!')
        for x,y in zip(self._in[0], self._out):
            # -> patch_size x number_of_patches -> transposed
            x = im2col_indices(nd.expand_dims(x, 0), self._kwargs['kernel'][0], self._kwargs['kernel'][1], self._kwargs['pad'][0], self._kwargs['stride'][0]).T
            # -> outsize x number_of_patches -> transposed
            y = y.flatten().T
            self._learn_pattern_linear(x, y)

    def forward_pattern_linear(self, *args):
        x = args[0]
        if len(args) < 2:
            x_neut = x
        else:
            x_neut = args[1]
        # omit number of samples, since present in both cov and var
        a = (self.cov.data() / (self.var_y.data().T + 1e-12)).reshape(self.weight.shape)
        z_neut = self.forward(x_neut)
        kwargs = self._kwargs
        kwargs['no_bias'] = True
        z = nd.Convolution(x, a, None, name='fwd', **kwargs)

        # this depends on ReLU!
        z = nd.where(z_neut>0, z, nd.zeros_like(z))
        return z, z_neut

#    def learn_assess_pattern_linear(self):
#        if self._in is None:
#            raise RuntimeError('Block has not yet executed forward_logged!')
#        x = self._in[0].flatten()
#        y = self._out.flatten()
#        a = self.cov.data() / (self.var_y.data().T + 1e-12)
#        dtd = self.qual_dtd.data()
#        dty = self.qual_dty.data()
#        signal = nd.dot(a, y, transpose_a=True)
#        d = x - signal
#
#        dtd += nd.dot(d, d, transpose_a=True)
#        dty += nd.dot(d, y, transpose_a=True)
#
#        self.qual_dtd.set_data(dtd)
#        self.qual_dty.set_data(dty)

    #####################
    ### Two-Component ###
    #####################

    def init_pattern_twocomponent(self):
        chan = self.weight.shape[0]
        ksize = np.prod(self.weight.shape[1:])
        self._init_pattern_twocomponent((chan, ksize))

    def learn_pattern_twocomponent(self):
        if self._in is None:
            raise RuntimeError('Block has not yet executed forward_logged!')
        for x,y in zip(self._in[0], self._out):
            # -> patch_size x number_of_patches -> transposed
            x = im2col_indices(nd.expand_dims(x, 0), self._kwargs['kernel'][0], self._kwargs['kernel'][1], self._kwargs['pad'][0], self._kwargs['stride'][0]).T
            # -> outsize x number_of_patches -> transposed
            y = y.flatten().T
            self._learn_pattern_twocomponent(x, y)

    def forward_pattern_twocomponent(self, *args):
        x = args[0]
        if len(args) < 2:
            x_neut = x
        else:
            x_neut = args[1]
        cov_pos = self.mean_xy_pos.data() - nd.dot(self.mean_y.data(), self.mean_x_pos.data(), transpose_a=True)
        cov_neg = self.mean_xy_neg.data() - nd.dot(self.mean_y.data(), self.mean_x_neg.data(), transpose_a=True)
        weight = self.weight.data().flatten()
        a_pos = cov_pos / ((weight * cov_pos).sum(axis=1, keepdims=True) + 1e-12)
        a_neg = cov_neg / ((weight * cov_neg).sum(axis=1, keepdims=True) + 1e-12)

        z_neut = self.forward(x_neut)
        kwargs = self._kwargs
        kwargs['no_bias'] = True
        z_pos = nd.Convolution(x, a_pos, None, name='fwd', **kwargs)
        z_neg = nd.Convolution(x, a_neg, None, name='fwd', **kwargs)

        # this depends on ReLU!
        z = nd.where(z_neut>0, z_pos, z_neg)

        return z, z_neut

class Conv2DTransposePatternNet(PatternNet, nn.Conv2DTranspose):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def assess_pattern(self):
        return

    ##############
    ### Linear ###
    ##############

    def init_pattern_linear(self):
        chan = self.weight.shape[0]
        ksize = np.prod(self.weight.shape[1:])
        with self.name_scope():
            self.mean_x = self.pparams.get('mean_x',
                                          shape=(1, chan),
                                          init=mx.initializer.Zero(),
                                          grad_req='null')
            self.mean_y = self.pparams.get('mean_y',
                                          shape=(1, ksize),
                                          init=mx.initializer.Zero(),
                                          grad_req='null')
            self.var_y = self.pparams.get('var_y',
                                         shape=(1, ksize),
                                         init=mx.initializer.Zero(),
                                         grad_req='null')
            self.cov = self.pparams.get('cov',
                                       shape=(ksize, chan),
                                       init=mx.initializer.Zero(),
                                       grad_req='null')
            self.num_samples = self.pparams.get('num_samples',
                                               shape=(1,),
                                               init=mx.initializer.Zero(),
                                               grad_req='null')

    def learn_pattern_linear(self):
        if self._in is None:
            raise RuntimeError('Block has not yet executed forward_logged!')
        for x,y in zip(self._in[0], self._out):
            # -> patch_size x number_of_patches -> transposed
            x = x.flatten().T
            #x = im2col_indices(nd.expand_dims(x, 0), self._kwargs['kernel'][0], self._kwargs['kernel'][1], self._kwargs['pad'][0], self._kwargs['stride'][0]).T
            # -> outsize x number_of_patches -> transposed
            y = im2col_indices(nd.expand_dims(y, 0), self._kwargs['kernel'][0], self._kwargs['kernel'][1], self._kwargs['pad'][0], self._kwargs['stride'][0]).T
            #y = y.flatten().T
            mean_x = self.mean_x.data()
            mean_y = self.mean_y.data()
            n = self.num_samples.data()
            var_y = self.var_y.data()
            cov = self.cov.data()
            m = x.shape[0]

            mean_x_cur = x.mean(axis=0, keepdims=True)
            mean_y_cur = y.mean(axis=0, keepdims=True)
            dx = x - mean_x_cur
            dy = y - mean_y_cur

            #import ipdb; ipdb.set_trace()

            cov_cur = nd.dot(dy, dx, transpose_a=True)
            cov += cov_cur + nd.dot((mean_y - mean_y_cur), (mean_x - mean_x_cur), transpose_a=True) * n * m / (n+m)

            var_y_cur = nd.sum(dy**2, axis=0)
            var_y += var_y_cur + ((mean_y - mean_y_cur) * (mean_y - mean_y_cur)) * n * m / (n+m)

            mean_x = (n * mean_x + m * mean_x_cur) / (n+m)
            mean_y = (n * mean_y + m * mean_y_cur) / (n+m)
            n += m

            self.mean_x.set_data(mean_x)
            self.mean_y.set_data(mean_y)
            self.num_samples.set_data(n)
            self.var_y.set_data(var_y)
            self.cov.set_data(cov)

    def forward_pattern_linear(self, *args):
        x = args[0]
        if len(args) < 2:
            x_neut = x
        else:
            x_neut = args[1]
        # omit number of samples, since present in both cov and var
        a = (self.cov.data() / (self.var_y.data().T + 1e-12)).reshape(self.weight.shape)
        z_neut = self.forward(x_neut)
        kwargs = self._kwargs
        kwargs['no_bias'] = True
        z = nd.Deconvolution(x, a, None, name='fwd', **kwargs)

        # this depends on ReLU!
        z = nd.where(z_neut>0, z, nd.zeros_like(z))
        return z, z_neut

#    def learn_assess_pattern_linear(self):
#        if self._in is None:
#            raise RuntimeError('Block has not yet executed forward_logged!')
#        x = self._in[0].flatten()
#        y = self._out.flatten()
#        a = self.cov.data() / (self.var_y.data().T + 1e-12)
#        dtd = self.qual_dtd.data()
#        dty = self.qual_dty.data()
#        signal = nd.dot(a, y, transpose_a=True)
#        d = x - signal
#
#        dtd += nd.dot(d, d, transpose_a=True)
#        dty += nd.dot(d, y, transpose_a=True)
#
#        self.qual_dtd.set_data(dtd)
#        self.qual_dty.set_data(dty)

    #####################
    ### Two-Component ###
    #####################

    def init_pattern_twocomponent(self):
        chan = self.weight.shape[0]
        ksize = np.prod(self.weight.shape[1:])
        with self.name_scope():
            self.mean_x_pos = self.pparams.get('mean_x_pos',
                                          shape=(1, ksize),
                                          init=mx.initializer.Zero(),
                                          grad_req='null')
            self.mean_x_neg = self.pparams.get('mean_x_neg',
                                          shape=(1, ksize),
                                          init=mx.initializer.Zero(),
                                          grad_req='null')
            self.mean_y = self.pparams.get('mean_y',
                                          shape=(1, chan),
                                          init=mx.initializer.Zero(),
                                          grad_req='null')
            self.cov_pos = self.pparams.get('cov_pos',
                                       shape=(chan, ksize),
                                       init=mx.initializer.Zero(),
                                       grad_req='null')
            self.cov_neg = self.pparams.get('cov_neg',
                                       shape=(chan, ksize),
                                       init=mx.initializer.Zero(),
                                       grad_req='null')
            self.num_samples = self.pparams.get('num_samples',
                                               shape=(1,),
                                               init=mx.initializer.Zero(),
                                               grad_req='null')

    def learn_pattern_twocomponent(self):
        if self._in is None:
            raise RuntimeError('Block has not yet executed forward_logged!')
        for x,y in zip(self._in[0], self._out):
            # -> patch_size x number_of_patches -> transposed
            x = im2col_indices(nd.expand_dims(x, 0), self._kwargs['kernel'][0], self._kwargs['kernel'][1], self._kwargs['pad'][0], self._kwargs['stride'][0]).T
            # -> outsize x number_of_patches -> transposed
            y = y.flatten().T
            mean_x_pos = self.mean_x_pos.data()
            mean_x_neg = self.mean_x_neg.data()
            mean_y = self.mean_y.data()
            n = self.num_samples.data()
            cov_pos = self.cov_pos.data()
            cov_neg = self.cov_neg.data()
            m = x.shape[0]

            x_pos = nd.maximum(0., x)
            x_neg = nd.minimum(0., x)

            mean_x_pos_cur = x_pos.mean(axis=0, keepdims=True)
            mean_x_neg_cur = x_neg.mean(axis=0, keepdims=True)
            mean_y_cur = y.mean(axis=0, keepdims=True)

            dx_pos = x_pos - mean_x_pos_cur
            dx_neg = x_neg - mean_x_neg_cur
            dy = y - mean_y_cur

            cov_pos_cur = nd.dot(dy, dx_pos, transpose_a=True)
            cov_neg_cur = nd.dot(dy, dx_neg, transpose_a=True)
            cov_pos += cov_pos_cur + nd.dot((mean_y - mean_y_cur), (mean_x_pos - mean_x_pos_cur), transpose_a=True) * n * m / (n+m)
            cov_neg += cov_neg_cur + nd.dot((mean_y - mean_y_cur), (mean_x_neg - mean_x_neg_cur), transpose_a=True) * n * m / (n+m)

            mean_x_pos = (n * mean_x_pos + m * mean_x_pos_cur) / (n+m)
            mean_x_neg = (n * mean_x_neg + m * mean_x_neg_cur) / (n+m)
            mean_y = (n * mean_y + m * mean_y_cur) / (n+m)
            n += m

            self.mean_x_pos.set_data(mean_x_pos)
            self.mean_x_neg.set_data(mean_x_neg)
            self.mean_y.set_data(mean_y)
            self.num_samples.set_data(n)
            self.cov_pos.set_data(cov_pos)
            self.cov_neg.set_data(cov_neg)

    def forward_pattern_twocomponent(self, *args):
        x = args[0]
        if len(args) < 2:
            x_neut = x
        else:
            x_neut = args[1]
        cov_pos = self.cov_pos.data()
        cov_neg = self.cov_neg.data()
        weight = self.weight.data().flatten()
        a_pos = (cov_pos / ((weight * cov_pos).sum(axis=1, keepdims=True) + 1e-12)).reshape(self.weight.shape)
        a_neg = (cov_neg / ((weight * cov_neg).sum(axis=1, keepdims=True) + 1e-12)).reshape(self.weight.shape)

        z_neut = self.forward(x_neut)
        kwargs = self._kwargs
        kwargs['no_bias'] = True
        z_pos = nd.Deconvolution(x, a_pos, None, name='fwd', **kwargs)
        z_neg = nd.Deconvolution(x, a_neg, None, name='fwd', **kwargs)

        # this depends on ReLU!
        z = nd.where(z_neut>0, z_pos, z_neg)

        return z, z_neut

class BatchNormPatternNet(PatternNet, nn.BatchNorm):
    def init_pattern(self, *args, **kwargs):
        pass

    def learn_pattern(self, *args, **kwargs):
        pass

    def assess_pattern(self, *args, **kwargs):
        pass

    def forward_pattern(self, *args, **kwargs):
        x = args[0]
        if len(args) < 2:
            x_neut = x
        else:
            x_neut = args[1]
        return self.forward(x)

class SequentialPatternNet(PatternNet, nn.Sequential):
    def init_pattern(self):
        for block in self._children:
            block.estimator = self.estimator
            block.init_pattern()

    def forward_pattern(self, *args):
        self._in = args
        x = args
        for block in self._children:
            block.estimator = self.estimator
            x = block.forward_pattern(*x)
        self._out = x[0]
        return self._out

    def learn_pattern(self, *args, **kwargs):
        for block in self._children:
            block.estimator = self.estimator
            block.learn_pattern()

    def explain_pattern(self, *args, **kwargs):
        x = args[0]
        x.attach_grad()
        with autograd.record():
            y = self.forward_pattern(x)
        y.backward()
        return x.grad

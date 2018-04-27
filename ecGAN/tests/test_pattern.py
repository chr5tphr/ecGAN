import pytest
import matplotlib as mpl
import yaml
import mxnet as mx
import numpy as np
import sys
import os
import logging
import h5py

from mxnet import nd
from mxnet.gluon.data import ArrayDataset
from mxnet import nd, gluon, autograd, random
import numpy as np

from ecGAN.data import register_data_func, data_funcs
from ecGAN.net import register_net, nets
from ecGAN.layer import Sequential, Dense, Identity
from ecGAN.model import models
from ecGAN.util import mkfilelogger, Config, config_ctx, load_module_file
from ecGAN.plot import plot_data, save_explanation
from ecGAN.func import linspace
from ecGAN.pattern.estimator import estimators


@register_data_func
def toy(ctx, nsamples=10000, seed=0xdeadbeef):
    rs = np.random.RandomState(seed)
    label = rs.uniform(-1.,1., size=nsamples).astype(np.float32)
    signal = np.stack([label, np.zeros(nsamples).astype(np.float32)], axis=-1)
    distractor = rs.multivariate_normal([0,0],[[1,1],[1,1]],size=nsamples).astype(np.float32)
    data = signal + distractor
    return ArrayDataset(data, label)

@register_net
class TOY(Sequential):
    def __init__(self, **kwargs):
        self._patest = kwargs.pop('patest', 'linear')
        super().__init__(**kwargs)
        with self.name_scope():
            self.add(Dense(1, regimes=estimators[self._patest]()))
            self.add(Identity(regimes=estimators[self._patest]()))

@pytest.fixture(scope='session', params=['linear', 'relu', 'positive'])
def conf_patest(request):
    return request.param

@pytest.fixture(scope='session', params=[5, 10])
def conf_pepochs(request):
    return request.param

@pytest.fixture(scope='session')
def config(conf_patest, conf_pepochs):
    conf = Config()
    conf.update({
        'device': 'cpu',
        'batch_size': 64,
        'nepochs': 10,
        'data': {
            'func': 'toy',
            'args': [],
            'bbox': [-3.,3.],
        },
        'model': 'Classifier',
        'nets':{
            'classifier':{
                'type': 'TOY',
                'name': 'test',
                'kwargs': {
                    'patest': conf_patest
                }
            }
        },
        'pattern': {
            'top_net': 'classifier',
            'nepochs': conf_pepochs,
            'optimizer': 'adam',
            'optkwargs': {
                'learning_rate': 0.0001,
                'wd': 0.99,
            }
        },
    })
    return conf

@pytest.fixture(scope='session')
def data(config):
    return data_funcs[config.data.func](*(config.data.args), ctx=ctx, **(config.data.kwargs))

@pytest.fixture(scope='session')
def ctx(config):
    return config_ctx(config)

@pytest.fixture(scope='session')
def model(config, ctx):
    model = models[config.model](ctx=ctx, config=config)
    return model

@pytest.fixture(scope='session')
def train(config, data, ctx, model):
    batch_size = config.batch_size
    nepochs = config.nepochs

    model.train(data, batch_size, nepochs)
    return model

@pytest.fixture(scope='session')
def init_pattern(train):
    model = train
    model.load_pattern_params()
    return model

@pytest.fixture(scope='session')
def fit_pattern(config, data, ctx, init_pattern):
    model = init_pattern
    batch_size = config.batch_size
    model.fit_pattern(data, batch_size)
    return model

@pytest.fixture(scope='session')
def sa_pattern(config, data, ctx, fit_pattern):
    model = fit_pattern
    batch_size = config.batch_size
    model.stats_assess_pattern(data, batch_size)
    return model

@pytest.fixture(scope='session')
def fa_pattern(config, data, ctx, sa_pattern):
    model = sa_pattern
    batch_size = config.batch_size
    model.fit_assess_pattern(data, batch_size)
    return model

@pytest.fixture(scope='session')
def assess_pattern(config, data, ctx, fa_pattern):
    return fa_pattern.assess_pattern()

@pytest.fixture(scope='session')
def explain_pattern(config, data, ctx, fit_pattern):
    model = fit_pattern
    data_iter = gluon.data.DataLoader(data, 30, shuffle=False, last_batch='discard')

    for X, Y in data_iter:
        X = X.as_in_context(ctx)
        Y = Y.as_in_context(ctx)

        if config.pattern.get('manual', False):
            relevance = model.backward_pattern(X)
        else:
            relevance = model.explain_pattern(X)

        relevance = relevance.reshape(X.shape)
        break

    return relevance

def test_explain_pattern(explain_pattern):
    assert np.isfinite(explain_pattern.asnumpy()).all()


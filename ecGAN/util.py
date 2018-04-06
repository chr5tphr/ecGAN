import numpy as np
import logging
import yaml
import random
import h5py
import importlib.util

import mxnet as mx
from mxnet import nd
from string import Template as STemplate

try:
    from .gpuman import nvidia_idle
    GPU_SUPPORT = True
except ModuleNotFoundError:
    GPU_SUPPORT = False

class Template(STemplate):
    idpattern = r'[_a-z][_\.a-z0-9]*'

class ConfigNode(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for key, val in self.items():
            self[key] = self.parse(val)

        self._flat = None

    def flat(self):
        if not self._flat:
            self._flat = {}
            self._flatten(self._flat)
        return self._flat

    def _flatten(self, root, prefix=''):
        for key, val in self.items():
            cap = '%s%s%s'%(prefix, '.' if prefix else '', key)
            if isinstance(val, __class__):
                val._flatten(root, cap)
            else:
                root[cap] = val

    def update(self, new):
        for key, val in new.items():
            if isinstance(val, dict) and isinstance(self.get(key), dict):
                self[key].update(val)
            else:
                self[key] = self.parse(val)

    @staticmethod
    def parse(X):
        if type(X) is dict:
            return ConfigNode(X)
        else:
            return X

    def __getattr__(self, name):
        errA = None
        try:
            return self.__getitem__(name)
        except KeyError as err:
            # return None
            errA = err
        raise AttributeError(errA)

    def sub(self, param, **kwargs):
        return Template(self.flat()[param]).safe_substitute(self.flat(), **kwargs)

    def exsub(self, param, **kwargs):
        return Template(param).safe_substitute(self.flat(), **kwargs)

class Config(ConfigNode):

    default_config = {
        'device':           'cpu',
        'device_id':        'auto',
        'model':            'GAN',
        'init':             False,
        'fuzzy_labels':     False,
        'feature_matching': False,
        'semi_supervised':  False,
        'net_file': None,
        'nets': {
            # 'generator': {
            #     'type':     'GenFC',
            #     'kwargs':   {},
            #     'param':    None,
            #     'save':     None,
            # },
            # 'discriminator': {
            #     'type':     'DiscrFC',
            #     'kwargs':   {},
            #     'param':    None,
            #     'save':     None,
            # },
            # 'classifier': {
            #     'type':     'ClassFC',
            #     'kwargs':   {},
            #     'param':    None,
            #     'save':     None,
            # },
        },
        'data': {
            'func':         'mnist',
            'args':         [],
            'kwargs':       {},
            'bbox':         [-1, 1]
        },
        'batch_size':       32,
        'nepochs':          10,
        'start_epoch':      10,
        'save_freq':        0,
        'log':              None,
        'genout':           None,
        'gen_freq':         0,
        'explanation': {
            'method':       'sensitivity',
            'output':       None,
            'image':        None,
            'input':        None,
            'top_net':      'discriminator',
        },
    }

    def __init__(self, *args, fname=None, **kwargs):
        super().__init__(*args, **kwargs)

        defcfg = ConfigNode(__class__.default_config)
        defcfg.update(self)
        self.update(defcfg)

        if fname is not None:
            with open(fname, 'r') as fp:
                self.update(yaml.safe_load(fp))

def config_ctx(config):
    if config.device == 'cpu' or not GPU_SUPPORT:
        return mx.context.cpu()
    elif config.device == 'gpu':
        if isinstance(config.device_id, int):
            return mx.context.gpu(config.device_id)
        else:
            return mx.context.gpu(random.choice(nvidia_idle()))

def mkfilelogger(lname, fname, level=logging.INFO):
    logger = logging.getLogger(lname)
    logger.setLevel(level)
    frmt = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fhdl = logging.FileHandler(fname)
    shdl = logging.StreamHandler()
    fhdl.setLevel(level)
    shdl.setLevel(level)
    fhdl.setFormatter(frmt)
    shdl.setFormatter(frmt)
    logger.addHandler(shdl)
    logger.addHandler(fhdl)
    return logger

def load_module_file(fname, module_name):
    spec = importlib.util.spec_from_file_location(module_name, fname)
    tmodule = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tmodule)
    return tmodule

def exec_file(fname):
    if fname is not None:
        with open(fname, 'r') as fp:
            exec(fp.read(), {'__builtins__': __builtins__}, {})


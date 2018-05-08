import numpy as np
import logging
import yaml
import random
import h5py
import importlib.util
import json

from types import FunctionType as function

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
        self._parse()

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
            if isinstance(val, dict):
                if isinstance(self.get(key), dict):
                    self[key].update(val)
                else:
                    self[key] = ConfigNode(val)
            else:
                self[key] = val

    def _parse(self):
        for key, val in self.items():
            if isinstance(val, dict):
                self[key] = ConfigNode(val)
            else:
                self[key] = val

    def raw(self):
        rdic = {}
        for key, val in self.items():
            if isinstance(val, ConfigNode):
                rdic[key] = val.raw()
            else:
                rdic[key] = val
        return rdic

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

    def update_from_file(self, fname):
        with open(fname, 'r') as fp:
            self.update(yaml.safe_load(fp))

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
            self.update_from_file(fname)

class ChainNode(ConfigNode):
    def __init__(self, *args, **kwargs):
        self._parent = kwargs.pop('parent', None)
        super().__init__(*args, **kwargs)

    def __getitem__(self, key):
        try:
            val = super().__getitem__(key)
        except KeyError as e:
            if self._parent is not None:
                val = self._parent.__getitem__(key)
            else:
                raise e
        return val

    def get(self, k, d=None):
        return super().get(k, None if self._parent is None else self._parent.get(k, d))

    def update(self, new):
        print(self, new)
        for key, val in new.items():
            if isinstance(val, dict):
                if isinstance(self.get(key), dict):
                    self[key].update(val)
                else:
                    self[key] = ChainNode(val, parent=None if self._parent is None else self._parent.get(key, None))
            else:
                self[key] = val

    def _parse(self):
        for key, val in self.items():
            if isinstance(val, dict):
                self[key] = ChainNode(val, parent=None if self._parent is None else self._parent.get(key, None))
            else:
                self[key] = val

    def dict(self):
        if self._parent is not None:
            tdic = self._parent.dict()
        else:
            tdic = {}
        tdic.update(self)
        return tdic

    def items(self):
        return self.dict().items()

    def keys(self):
        return self.dict().keys()

    def values(self):
        return self.dict().values()

class ChainConfig(ChainNode):
    def __init__(self, *args, **kwargs):
        children = kwargs.pop('tail', [])
        action = kwargs.pop('action', None)
        fname = kwargs.pop('fname', None)
        priority = kwargs.pop('priority', 0)
        super().__init__(*args, **kwargs)
        self._tail = [ChainConfig(child.pop('dict', {}), parent=self, **child) for child in children]
        self._action = action
        self._priority = priority

        if fname is not None:
            self.update_from_file(fname)

    def _leaves(self):
        if len(self._tail):
            return sum([child.leaves() for child in self._tail], [])
        else:
            return [self]

    def leaves(self):
        return sorted(self._leaves(), key=lambda x: x._priority)

class RessourceManager(dict):
    def __call__(self, func, *args, **kwargs):
        try:
            return self[self.dhash(func, args, kwargs)]
        except KeyError:
            return func(*args, **kwargs)

    @staticmethod
    def dhash(*obj):
        enc = HashEncoder(sort_keys=True, separators=(',',':'))
        return hash(enc.encode(obj))

def config_ctx(config):
    if config.device == 'cpu' or not GPU_SUPPORT:
        return mx.context.cpu()
    elif config.device == 'gpu':
        if isinstance(config.device_id, int):
            return mx.context.gpu(config.device_id)
        else:
            devs = nvidia_idle()
            if not len(devs):
                raise RuntimeError("No GPUs available!")
            return mx.context.gpu(random.choice(nvidia_idle()))

def make_ctx(device, device_id):
    if device == 'cpu' or not GPU_SUPPORT:
        return mx.context.cpu()
    elif device == 'gpu':
        if isinstance(device_id, int):
            return mx.context.gpu(device_id)
        else:
            devs = nvidia_idle()
            if not len(devs):
                raise RuntimeError("No GPUs available!")
            return mx.context.gpu(random.choice(nvidia_idle()))

class HashEncoder(json.JSONEncoder):
    _convs = {
        function: lambda obj: obj.__name__,
        ChainNode: lambda obj: obj.raw(),
        ConfigNode: lambda obj: obj.raw(),
        mx.context.Context: lambda obj: {'device': obj.device_type, 'device_id': obj.device_id},
        nd.NDArray: lambda obj: obj.asnumpy(),
        np.ndarray: lambda obj: obj.tolist(),
    }
    def default(self, obj):
        for otype, conv in self._convs.items():
            if isinstance(obj, otype):
                return conv(obj)
        return super().default(obj)

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


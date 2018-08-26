import numpy as np
import logging
import yaml
import random
import h5py
import importlib.util
import json
import sys

from types import FunctionType as function
from lark import Lark, Transformer

import mxnet as mx
from mxnet import nd
from string import Template as STemplate

from .func import asnumpy

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
        self._deepen()

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

    def _deepen(self):
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

    def __getitem__(self, key):
        return super().__getitem__(key)

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
        self._root = kwargs.pop('root', None)
        self._path = kwargs.pop('path', [])
        super().__init__(*args, **kwargs)

    def __getitem__(self, key):
        try:
            return super().__getitem__(key)
        except KeyError as e:
            try:
                return self._parent()[key]
            except KeyError:
                raise e

    def _parent(self):
        if self._root is None:
            raise KeyError
        cur = self._root
        for item in self._path:
            cur = cur[item]
        return cur

    def get(self, k, d=None):
        '''
        get does not call __getitem__, so we need to define get as well
        '''
        try:
            return super().__getitem__(k)
        except KeyError:
            try:
                return self._parent()[k]
            except KeyError:
                return d

    #def _get_own(self, k, d=None):
    #    try:
    #        return super().__getitem__(k)
    #    except KeyError:
    #        return d

    def update(self, new):
        for key, val in new.items():
            if isinstance(val, dict):
                # we can use super().get here, because get does not call __getitem__
                if isinstance(super().get(key), dict):
                    self[key].update(val)
                else:
                    self[key] = ChainNode(val, root=self._root, path=self._path + [key])
            else:
                self[key] = val

    def _deepen(self):
        for key, val in super().items():
            if isinstance(val, dict):
                self[key] = ChainNode(val, root=self._root, path=self._path + [key])
            else:
                self[key] = val

    def fuse(self):
        try:
            node = self._parent().fuse()
        except KeyError:
            node = ConfigNode()
        node.update(self.copy())
        return node

    def items(self):
        return self.fuse().items()

    def keys(self):
        return self.fuse().keys()

    def values(self):
        return self.fuse().values()

class ChainConfig(ChainNode):
    def __init__(self, *args, **kwargs):
        children = kwargs.pop('tail', [])
        action = kwargs.pop('action', None)
        fname = kwargs.pop('fname', None)
        priority = kwargs.pop('priority', 0)
        tag = kwargs.pop('tag', None)
        super().__init__(*args, **kwargs)
        self._tag = tag
        self._action = action
        self._priority = priority
        self._tail = []
        for child in children:
            nkwa = child.copy()
            base = nkwa.pop('dict', {})
            nkwa['root'] = self
            self._tail.append(ChainConfig(base, **nkwa))

        if fname is not None:
            self.update_from_file(fname)

    def _leaves(self):
        if len(self._tail):
            return sum([child.leaves() for child in self._tail], [])
        else:
            return [self]

    def leaves(self):
        return sorted(self._leaves(), key=lambda x: x._priority)

    def tags(self):
        own = [] if self._tag is None else [self._tag]
        if self._root is None:
            return own
        else:
            return self._root.tags() + own

class RessourceManager(dict):
    def __call__(self, func, *args, **kwargs):
        key = self.dhash(func, args, kwargs)
        try:
            return self[key]
        except KeyError:
            val = func(*args, **kwargs)
            self[key] = val
            return val

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
        nd.NDArray: lambda obj: asnumpy(obj),
        np.ndarray: lambda obj: obj.tolist(),
    }
    def default(self, obj):
        for otype, conv in self._convs.items():
            if isinstance(obj, otype):
                return conv(obj)
        return super().default(obj)

def mkfilelogger(lname, fname=None, stream=sys.stderr, level=logging.INFO):
    logger = logging.getLogger(lname)
    for hdlr in logger.handlers.copy():
        logger.removeHandler(hdlr)
    logger.setLevel(level)
    frmt = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    if fname is not None:
        fhdl = logging.FileHandler(fname)
        fhdl.setLevel(level)
        fhdl.setFormatter(frmt)
        logger.addHandler(fhdl)
    if stream is not None:
        shdl = logging.StreamHandler()
        shdl.setLevel(level)
        shdl.setFormatter(frmt)
        logger.addHandler(shdl)
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


logic_tag_grammar = r"""
                     ?expr : or
                     ?or   : and
                           | or "|" and -> _or
                     ?and  : atom
                           | and "&" atom -> _and
                     ?atom : VALUE -> _val
                           | "!" atom -> _not
                           | "(" or ")"
                     VALUE : /[^!&|()]+/
                     """

class LogicTagTrans(Transformer):
    def __init__(self, func):
        self._func = func
    def _val(self, a):
        return self._func(a[0])
    def _or(self, a):
        return a[0] or a[1]
    def _and(self, a):
        return a[0] and a[1]
    def _not(self, a):
        return not a[0]

def get_logic_parser(func):
    trans = LogicTagTrans(func)
    parser = Lark(logic_tag_grammar, start='expr', parser='lalr', transformer=trans)
    return parser


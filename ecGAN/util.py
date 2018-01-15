import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import mxnet as mx
import logging
import yaml
import random

from mxnet import nd
from string import Template as STemplate

from .gpuman import nvidia_idle

class Template(STemplate):
    idpattern = r'[_a-z][_\.a-z0-9]*'

class ConfigNode(dict):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        for key,val in self.items():
            self[key] = self.parse(val)

        self._flat = None

    def flat(self):
        if not self._flat:
            self._flat = {}
            self._flatten(self._flat)
        return self._flat

    def _flatten(self,root,prefix=''):
        for key,val in self.items():
            cap = '%s%s%s'%(prefix, '.' if prefix else '', key)
            if isinstance(val,__class__):
                val._flatten(root,cap)
            else:
                root[cap] = val

    def update(self,new):
        for key, val in new.items():
            if isinstance(val,dict) and isinstance(self.get(key),dict):
                self[key].update(val)
            else:
                self[key] = self.parse(val)

    @classmethod
    def parse(clss,X):
        if type(X) is dict:
            return clss(X)
        else:
            return X

    def __getattr__(self,name):
        errA = None
        try:
            return self.__getitem__(name)
        except KeyError as err:
            # return None
            errA = err
        raise AttributeError(errA)

    def sub(self,param,**kwargs):
        return Template(self.flat()[param]).safe_substitute(self.flat(),**kwargs)

class Config(ConfigNode):

    default_config = {
        'device':           'cpu',
        'device_id':        'auto',
        'model':            'GAN',
        'init':             False,
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
            'func':         'get_mnist_single',
            'args':         [],
            'kwargs':       {},
        },
        'batch_size':       32,
        'nepochs':          10,
        'start_epoch':      10,
        'save_freq':        0,
        'log':              None,
        'genout':           None,
        'gen_freq':         0,
    }

    def __init__(self,fname=None,*args,**kwargs):
        super().__init__(*args,**kwargs)

        defcfg = ConfigNode(__class__.default_config)
        defcfg.update(self)
        self.update(defcfg)

        if fname:
            with open(fname,'r') as fp:
                self.update(yaml.safe_load(fp))

def config_ctx(config):
    if config.device == 'cpu':
        return mx.context.cpu()
    elif config.device == 'gpu':
        if isinstance(config.device_id,int):
            return mx.context.gpu(config.device_id)
        else:
            return mx.context.gpu(random.choice(nvidia_idle()))


def plot_data(data):
    snum = int(len(data)**.5)
    if type(data) is nd.NDArray:
        data = data.asnumpy()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.imshow(data[:snum**2].reshape(snum,snum,28,28).transpose(0,2,1,3).reshape(snum*28,snum*28),cmap='Greys')
    ax.axis('off')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.hlines(np.arange(1,snum)*28,0,snum*28)
    ax.vlines(np.arange(1,snum)*28,0,snum*28)
    fig.tight_layout()
    ax.set
    return fig

def draw_heatmap(data):
    ndat = data **2
    ndat = (ndat / ndat.max() * 255).astype(np.uint8)
    ndat = np.stack([ndat] + [np.zeros(ndat.shape)]*2,axis=-1)
    return ndat

def mkfilelogger(lname,fname,level=logging.INFO):
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

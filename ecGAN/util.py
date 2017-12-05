import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import mxnet as mx
import logging
import yaml

from mxnet import nd
from string import Template

class Config(dict):

    default_config = {
        'device':           'cpu',
        'device_id':        0,
        'netC':             'ClassFC',
        'netG':             'GenFC',
        'netD':             'DiscrFC',
        'model':            'GAN',
        'data_func':        'get_mnist_single',
        'data_args':        [],
        'data_kwargs':      {},
        'batch_size':       32,
        'nepochs':          10,
        'start_epoch':      10,
        'save_freq':        0,
        'paramC'            None,
        'paramD':           None,
        'paramG':           None,
        'saveC':            None,
        'saveD':            None,
        'saveG':            None,
        'log':              None,
        'genout':           None,
    }

    def __init__(self,fname=None,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.update(__class__.default_config)
        if fname:
            with open(fname,'r') as fp:
                self.update(yaml.safe_load(fp))

    def __getattr__(self,name):
        try:
            return self.__getitem__(name)
        except KeyError as err:
            raise AttributeError(err)

    def sub(self,param,**kwargs):
        return Template(self[param]).safe_substitute(self,**kwargs)

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

import matplotlib as mpl
mpl.use('Agg')
import yaml
import mxnet as mx
import numpy as np
import sys

from argparse import ArgumentParser
from imageio import imwrite
from mxnet import nd

from ecGAN.net import nets
from ecGAN.model import models
from ecGAN.util import data_funcs,mkfilelogger,plot_data,Config

commands = {}
def register_command(func):
    commands[func.__name__] = func
    return func

def main():
    parser = ArgumentParser()

    parser.add_argument('command',choices=commands.keys())
    parser.add_argument('-f','--config')
    parser.add_argument('-u','--update')

    args = parser.parse_args(sys.argv[1:])

    config = Config(args.config)
    if args.update:
        config.update(yaml.safe_load(args.update))

    commands[args.command](args,config)

@register_command
def train(args,config):
    ctx = mx.context.Context(config.device,config.device_id)

    batch_size = config.batch_size
    nepochs = config.nepochs
    start_epoch = config.start_epoch
    save_freq = config.save_freq

    data = data_funcs[config.data_func](*(config.data_args),**(config.data_kwargs))

    netG = nets[config.netG]()
    if config.paramG:
        netG.load_params(config.paramG,ctx=ctx)
    else:
        netG.initialize(mx.init.Xavier(magnitude=2.24),ctx=ctx)
    netD = nets[config.netD]()
    if config.paramD:
        netD.load_params(config.paramD,ctx=ctx)
    else:
        netD.initialize(mx.init.Xavier(magnitude=2.24),ctx=ctx)

    logger = None
    if config.log:
        logger = mkfilelogger('training',config.sub('log'))

    model = models[config.model](netG=netG,netD=netD,ctx=ctx,logger=logger,config=config)
    model.train(data,batch_size,nepochs)


@register_command
def generate(args,config):
    ctx = mx.context.Context(config.device, config.device_id)

    netG = nets[config.netG]()
    if config.paramG:
        netG.load_params(config.sub('paramG'),ctx=ctx)
    else:
        netG.initialize(mx.init.Xavier(magnitude=2.24),ctx=ctx)

    logger = None
    if config.log:
        logger = mkfilelogger('generation',config.sub('log'))

    noise = nd.random_normal(shape=(30, 32), ctx=ctx)
    gdat = None
    cond = None
    if config.model == 'CGAN':
        cond = nd.one_hot(nd.repeat(nd.arange(10,ctx=ctx),3),10)
        gdat = netG(noise, cond)
    else:
        gdat = netG(noise)

    if config.genout:
        fpath = config.sub('genout')
        gdat = ((gdat + 1) * 255/2).asnumpy().astype(np.uint8)
        # snum = int(len(gdat)**.5)
        # imwrite(fpath, gdat[:snum**2].reshape(snum,snum,28,28).transpose(0,2,1,3).reshape(snum*28,snum*28))
        imwrite(fpath, gdat.reshape(5,6,28,28).transpose(0,2,1,3).reshape(5*28,6*28))
        if logger:
            logger.info('Saved generated data by generator \'%s\' checkpoint \'%s\' in \'%s\'.',config.netG,config.sub('paramG'),config.sub('genout'))
    else:
        fig = plot_data(gdat)
        fig.show()

if __name__ == '__main__':
    main()

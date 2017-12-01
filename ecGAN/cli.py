import matplotlib as mpl
mpl.use('Agg')
import yaml
import mxnet as mx
import sys

from argparse import ArgumentParser

from ecGAN.net import nets,GAN
from ecGAN.util import data_funcs,mkfilelogger,plot_data,Config

commands = {}
def register_command(func):
    commands[func.__name__] = func
    return func

def call():
    parser = ArgumentParser()

    parser.add_argument('command',choices=commands.keys())
    parser.add_argument('-f','--config')

    args = parser.parse_args(sys.argv[1:])

    config = Config(args.config)

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

    model = GAN(netG=netG,netD=netD,ctx=ctx,logger=logger,config=config)
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

    fig = plot_data(netG(mx.nd.random_normal(shape=(25, 32), ctx=ctx)))
    if config.genout:
        fpath = config.sub('genout')
        fig.savefig(fpath)
        if logger:
            logger.info('Saved generated data by generator \'%s\' checkpoint \'%s\' in \'%s\'.',config.netG,config.sub('paramG'),config.sub('genout'))
    else:
        fig.show()

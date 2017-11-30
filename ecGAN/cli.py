import matplotlib as mpl
mpl.use('Agg')
import yaml
import mxnet as mx
import sys

from argparse import ArgumentParser

from ecGAN.net import nets,train_GAN
from ecGAN.util import data_funcs,mkfilelogger,plot_data,Config

commands = {}
def register_command(func):
    commands[func.__name__] = func
    return func

def call():
    parser = ArgumentParser()

    parser.add_argument('command',choices=commands.keys())

    args,rargv = parser.parse_known_args(sys.argv[1:])

    commands[args.command](rargv)

@register_command
def train(argv):
    parser = ArgumentParser()
    parser.add_argument('-f','--config')

    args = parser.parse_args(argv)

    config = Config(args.config)

    _train(args,config)

def _train(args,config):
    ctx = mx.context.Context(config.device,config.device_id)

    batch_size = config.batch_size
    nepochs = config.nepochs
    start_epoch = config.start_epoch
    chkfreq = config.chkfreq

    data = data_funcs[config.data_func](*(config.data_args),**(config.data_kwargs))

    netG = nets[config.netG]()
    if 'paramG' in config:
        netG.load_params(config.paramG,ctx=ctx)
    else:
        netG.initialize(mx.init.Xavier(magnitude=2.24),ctx=ctx)
    netD = nets[config.netD]()
    if 'paramD' in config:
        netD.load_params(config.paramD,ctx=ctx)
    else:
        netD.initialize(mx.init.Xavier(magnitude=2.24),ctx=ctx)

    logger = None
    if 'log' in config:
        logger = mkfilelogger('training',config.sub('log'))

    train_GAN(data,batch_size,netG,netD,ctx,nepochs=nepochs,logger=logger)

    if chkfreq <= 0:
        epoch = start_epoch + nepochs - 1
        if 'saveG' in config:
            netG.save_params(config.sub('saveG',epoch=epoch))
        if 'saveD' in config:
            netD.save_params(config.sub('saveD',epoch=epoch))

@register_command
def generate(argv):
    parser = ArgumentParser()
    parser.add_argument('-f','--config')

    args = parser.parse_args(argv)

    config = {}
    if args.config:
        with open(args.config,'r') as fp:
            config.update(yaml.safe_load(fp))

    _generate(args,**config)

def _generate(args,config):
    ctx = mx.context.Context(config.device, config.device_id)

    netG = nets[config.netG]()
    if 'paramG' in config:
        netG.load_params(config.sub('paramG'),ctx=ctx)
    else:
        netG.initialize(mx.init.Xavier(magnitude=2.24),ctx=ctx)

    fig = plot_data(netG(mx.nd.random_normal(shape=(25, 32), ctx=ctx)))
    if 'genout' in config:
        fig.savefig(config.sub('genout'))
    else:
        fig.show()

import matplotlib as mpl
mpl.use('Agg')
import yaml
import mxnet as mx
import sys
from argparse import ArgumentParser

from ecGAN.net import nets,train_GAN
from ecGAN.util import data_funcs,mkfilelogger,plot_data

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

    config = {}
    if args.config:
        with open(args.config,'r') as fp:
            config.update(yaml.safe_load(fp))

    _train(args,**config)

def _train(args,**config):
    ctx = mx.context.Context(config.get('device','cpu'),config.get('device_id',0))

    batch_size = config.get('batch_size',32)
    nepochs = config.get('nepochs',10)

    data = data_funcs[config.get('data_func','get_mnist_single')](*(config.get('data_args',[])),**(config.get('data_kwargs',{})))

    netG = nets[config.get('netG','GenFC')]()
    if config.get('paramG'):
        netG.load_params(config['paramG'],ctx=ctx)
    else:
        netG.initialize(mx.init.Xavier(magnitude=2.24),ctx=ctx)
    netD = nets[config.get('netD','DiscrFC')]()
    if config.get('paramD'):
        netD.load_params(config['paramD'],ctx=ctx)
    else:
        netD.initialize(mx.init.Xavier(magnitude=2.24),ctx=ctx)

    logger = None
    if config.get('log'):
        logger = mkfilelogger('training',config['log'])

    train_GAN(data,batch_size,netG,netD,ctx,nepochs=nepochs,logger=logger)

    if config.get('saveG'):
        netG.save_params(config['saveG'])
    if config.get('saveD'):
        netD.save_params(config['saveD'])

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

def _generate(args,**config):
    ctx = mx.context.Context(config.get('device','cpu'),config.get('device_id',0))

    netG = nets[config.get('netG','GenFC')]()
    if config.get('paramG'):
        netG.load_params(config['paramG'],ctx=ctx)
    else:
        netG.initialize(mx.init.Xavier(magnitude=2.24),ctx=ctx)

    fig = plot_data(netG(nd.random_normal(shape=(25, 32), ctx=ctx)))
    if config.get('genout'):
        fig.savefig(config['genout'])
    else:
        fig.show()

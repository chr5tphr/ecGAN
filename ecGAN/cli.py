import matplotlib as mpl
mpl.use('Agg')
import yaml
import mxnet as mx
import numpy as np
import sys
import logging
import h5py

from argparse import ArgumentParser
from imageio import imwrite
from mxnet import nd, gluon, autograd

from .net import nets
from .model import models
from .data import data_funcs
from .util import mkfilelogger, plot_data, Config, config_ctx
from .func import Interpretable

commands = {}
def register_command(func):
    commands[func.__name__] = func
    return func

def main():
    parser = ArgumentParser()

    parser.add_argument('command',choices=commands.keys())
    parser.add_argument('-f','--config')
    parser.add_argument('-u','--update')
    parser.add_argument('--epoch_range',nargs=3,type=int)
    parser.add_argument('--iter',type=int,default=1)

    args = parser.parse_args(sys.argv[1:])

    config = Config(args.config)
    if args.update:
        config.update(yaml.safe_load(args.update))

    commands[args.command](args,config)

@register_command
def train(args,config):
    ctx = config_ctx(config)

    batch_size = config.batch_size
    nepochs = config.nepochs

    data = data_funcs[config.data.func](*(config.data.args),**(config.data.kwargs))

    logger = None
    if config.log:
        logger = mkfilelogger('training',config.sub('log'),logging.DEBUG if config.get('debug') else logging.INFO )

    model = models[config.model](ctx=ctx,logger=logger,config=config)
    model.train(data,batch_size,nepochs)


@register_command
def generate(args,config):
    ctx = config_ctx(config)

    netG = nets[config.nets.generator.type]()


    logger = None
    if config.log:
        logger = mkfilelogger('generation',config.sub('log'))

    for epoch in (range(*args.epoch_range) if args.epoch_range else [config.start_epoch]):
        if config.nets.generator.param:
            netG.load_params(config.sub('nets.generator.param',start_epoch=epoch),ctx=ctx)
        else:
            netG.initialize(mx.init.Xavier(magnitude=2.24),ctx=ctx)
        noise = nd.random_normal(shape=(30, 100), ctx=ctx)
        gdat = None
        cond = None
        if config.model == 'CGAN':
            cond = nd.one_hot(nd.repeat(nd.arange(10,ctx=ctx),3),10)
            gdat = netG(noise, cond)
        else:
            gdat = netG(noise)

        if config.genout:
            bbox = config.data.bbox
            fpath = config.sub('genout',epoch=epoch)
            gdat = ((gdat - bbox[0]) * 255/(bbox[1]-bbox[0])).asnumpy().clip(0,255).astype(np.uint8)
            # snum = int(len(gdat)**.5)
            # imwrite(fpath, gdat[:snum**2].reshape(snum,snum,28,28).transpose(0,2,1,3).reshape(snum*28,snum*28))
            imwrite(fpath, gdat.reshape(5,6,28,28).transpose(0,2,1,3).reshape(5*28,6*28))
            if logger:
                logger.info('Saved generated data by generator \'%s\' checkpoint \'%s\' in \'%s\'.',
                            config.nets.generator.type,config.sub('nets.generator.param',start_epoch=epoch),fpath)
        else:
            fig = plot_data(gdat)
            fig.show()

@register_command
def test(args, config):
    ctx = config_ctx(config)

    logger = None
    if config.log:
        logger = mkfilelogger('testing',config.sub('log'))

    batch_size = config.batch_size
    data = data_funcs[config.data.func](*(config.data.args),**(config.data.kwargs))

    model = models[config.model](ctx=ctx,logger=logger,config=config)
    model.test(data=data,batch_size=batch_size)

@register_command
def explain(args,config):
    if config.model != 'Classifier':
        raise NotImplementedError('Only Classifier models are currently supported.')

    logger = None
    if config.log:
        logger = mkfilelogger('testing',config.sub('log'))

    data_fp = data_funcs[config.data.func](*(config.data.args),**(config.data.kwargs))

    netC = nets[config.nets.classifier.type]()
    if not isinstance(netC,Interpretable):
        raise NotImplementedError('\'%s\' is not yet Interpretable!'%config.nets.classifier.type)

    loss = gluon.loss.SoftmaxCrossEntropyLoss()
    ctx = config_ctx(config)

    if config.nets.classifier.param:
        netC.load_params(config.sub('nets.classifier.param'),ctx=ctx)
    else:
        netC.initialize(mx.init.Xavier(magnitude=2.24),ctx=ctx)

    data_iter = gluon.data.DataLoader(data_fp,30,shuffle=False,last_batch='discard')
    for i,(data,label) in enumerate(data_iter):
        if i >= args.iter:
            break
        data = data.as_in_context(ctx).reshape((-1,784))
        label = label.as_in_context(ctx)

        if config.explanation.method == 'sensitivity':
            output = netC(data)
            output.attach_grad()
            with autograd.record():
                err = loss(output, label)
            dEdy = autograd.grad(err,output)
        else:
            dEdy = None

        relevance = netC.relevance(data,dEdy,method=config.explanation.method)


        if config.explanation.output:
            with h5py.File(config.sub('explanation.output',iter=i,epoch=config.start_epoch),'w') as fp:
                fp['heatmap'] = relevance.asnumpy()
            if logger:
                logger.info('Saved explanation of \'%s\' checkpoint \'%s\' in \'%s\'.',
                            config.nets.classifier.type,config.sub('nets.classifier.param'),fpath)
        if config.explanation.image:
            rdat = relevance
            if config.explanation.method == 'sensitivity':
                rdat = relevance**2
            lo,hi = relevance.min(),relevance.max()
            fpath = config.sub('explanation.image',iter=i,epoch=config.start_epoch)
            rdat = ((relevance - lo) * 255/(hi-lo)).asnumpy().astype(np.uint8)
            # rdat = np.stack([rdat] + [np.zeros(rdat.shape)]*2,axis=-1)
            imwrite(fpath, rdat.reshape(5,6,28,28).transpose(0,2,1,3).reshape(5*28,6*28))
            if logger:
                logger.info('Saved explanation image of \'%s\' checkpoint \'%s\' in \'%s\'.',
                            config.nets.classifier.type,config.sub('nets.classifier.param'),fpath)
        if config.explanation.input:
            bbox = config.data.bbox
            fpath = config.sub('explanation.input',iter=i)
            indat = ((data - bbox[0]) * 255/(bbox[1]-bbox[0])).asnumpy().clip(0,255).astype(np.uint8)
            imwrite(fpath, indat.reshape(5,6,28,28).transpose(0,2,1,3).reshape(5*28,6*28))
            if logger:
                logger.info('Saved input data \'%s\' iter %d in \'%s\'.',
                            config.data.func,i,fpath)


if __name__ == '__main__':
    main()

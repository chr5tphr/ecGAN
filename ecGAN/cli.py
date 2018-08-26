import matplotlib as mpl
import yaml
import json
import mxnet as mx
import numpy as np
import sys
import os
import logging
import h5py
import traceback

from argparse import ArgumentParser
from imageio import imwrite
from mxnet import nd, gluon, autograd, random

from .net import nets
from .model import models
from .data import data_funcs
from .util import mkfilelogger, Config, config_ctx, make_ctx, load_module_file, ChainConfig, RessourceManager, get_logic_parser
from .plot import plot_data, save_data_h5, save_colorized_image, save_aligned_image, save_raw_image, save_cgan_visualization, save_predictions
from .func import linspace, asnumpy

getLogger = logging.getLogger

commands = {}
def register_command(func):
    commands[func.__name__] = func
    return func

ress = RessourceManager()

def main():
    parser = ArgumentParser()

    parser.add_argument(      'command'      , choices=commands.keys())
    parser.add_argument('-f', '--config'     , action='append', default=[])
    parser.add_argument('-c', '--chain'      , action='append', default=[])
    parser.add_argument('-u', '--update'     , action='append', default=[])
    parser.add_argument(      '--epoch_range', type=int, nargs=3)
    parser.add_argument(      '--iter'       , type=int, default=1)
    parser.add_argument(      '--seed'       , type=int, default=0xDEADBEEF)
    parser.add_argument(      '--classnum'   , type=int, default=10)
    parser.add_argument(      '--debug'      , action='store_true')
    parser.add_argument(      '--mkdirs'     , action='store_true')
    parser.add_argument(      '--crash_chain', action='store_true')
    parser.add_argument('-k', '--pskip'      , action='append', type=int, default=[])
    parser.add_argument('-t', '--tag'        , type=str, default='', help='-t \'(tag1|!tag2)&tag3|tag4\' := ((tag1 or not tag2) and tag3) or tag4')

    args = parser.parse_args(sys.argv[1:])

    if len(args.config):
        config = Config()

        for cpath in args.config:
            config.update_from_file(cpath)

        for ustr in args.update:
            config.update(yaml.safe_load(ustr))

        net_module = load_module_file(config.sub('net_file'), 'net_module')

        if args.mkdirs:
            mkdirs(None, lconf)
    else:
        config = None

    if args.seed:
        random.seed(args.seed)

    if args.debug:
        import ipdb; ipdb.set_trace()

    commands[args.command](args, config)

def debug():
    import ipdb
    import traceback
    try:
        ipdb.runcall(main)
    except Exception as e:
        traceback.print_exc(file=sys.stderr)
        ipdb.post_mortem(e.__traceback__)

@register_command
def setup(args, config):
    # ctx = config_ctx(config)

    # setup_engine(config.db_engine)
    # with session_scope() as session:
    #     setup = Setup(
    #         time_created = datetime.now(),
    #         time_modified = datetime.now(),
    #     )
    #     session.flush()
    #     config['setup_id'] = setup.id
    #     setup.storage_path = config.sub('setup_path')
    #     setup.config = config

    explore = [
        'log',
        'genout',
        'explanation.input',
        'explanation.outout',
        'explanation.image',
        'nets.discriminator.params',
        'nets.discriminator.save',
        'nets.generator.params',
        'nets.generator.save',
        'nets.classifier.params',
        'nets.classifier.save',
        ]
    for key in explore:
        try:
            os.makedirs(os.path.dirname(config.sub(key)), exist_ok=True)
        except KeyError:
            pass

    # with open(os.path.join([config.sub('setup_path'), 'config.yaml']), 'w') as fp:
    #     yaml.safe_dump(fp, config)

@register_command
def mkdirs(args, config):
    explore = [
        'log',
        'genout',
        'explanation.outout',
        'pattern.outout',
        'pattern.save',
        'pattern.load',
        'nets.dis.params',
        'nets.dis.save',
        'nets.gen.params',
        'nets.gen.save',
        'nets.cls.params',
        'nets.cls.save',
        ]
    for key in explore:
        try:
            os.makedirs(os.path.dirname(config.sub(key)), exist_ok=True)
        except KeyError:
            pass

@register_command
def chain(args, config):
    for fname in args.chain:
        with open(fname,'r') as fp:
            rawdict = yaml.safe_load(fp)
        content = rawdict.pop('dict', {})
        ctree = ChainConfig(content, **rawdict)

        for leaf in ctree.leaves():
            if leaf._priority in args.pskip:
                continue
            ltags = leaf.tags()
            if args.tag is not None:
                parser = get_logic_parser(lambda x: x in ltags)
                if not parser.parse(args.tag):
                    continue
            lconf = leaf.fuse()
            mkfilelogger('ecGAN', lconf.sub('log'), logging.DEBUG if lconf.get('debug') else logging.INFO)
            getLogger('ecGAN').info('Running chain leaf "%s".'%('.'.join(ltags)))
            for ustr in args.update:
                lconf.update(yaml.safe_load(ustr))
            net_module = ress(load_module_file, lconf.sub('net_file'), 'net_module')

            if args.mkdirs:
                mkdirs(None, lconf)

            if args.crash_chain:
                commands[leaf._action](args, lconf)
            else:
                try:
                    commands[leaf._action](args, lconf)
                except:
                    print(traceback.format_exc(), file=sys.stderr)


@register_command
def print_config(args, config):
    print(yaml.safe_dump(config.raw(), default_flow_style=False))

@register_command
def train(args, config):
    ctx   = ress(make_ctx, config.device, config.device_id)
    data  = ress(data_funcs[config.data.func], *(config.data.args), ctx=ctx, **(config.data.kwargs))

    if config.log:
        mkfilelogger('ecGAN', config.sub('log'), logging.DEBUG if config.get('debug') else logging.INFO)

    model = models[config.model.type](ctx=ctx, config=config, **config.model.kwargs)
    model.train(data, config.batch_size, config.nepochs)

    del model


@register_command
def generate(args, config):
    if config.log:
        mkfilelogger('ecGAN', config.sub('log'), logging.DEBUG if config.get('debug') else logging.INFO)

    ctx       = ress(make_ctx, config.device, config.device_id)
    model     = models[config.model.type](ctx=ctx, config=config, **config.model.kwargs)
    num       = 30
    K         = args.classnum
    cond      = nd.one_hot(nd.repeat(nd.arange(K, ctx=ctx), (num-1)//K+1)[:num], K).reshape((num, K, 1, 1))
    net_desc  = config.nets[config.model.kwargs.generator]
    netnam    = net_desc.name
    net_epoch = net_desc.epoch
    templ     = config.genout

    for i in range(args.iter):
        noise = nd.random_normal(shape=(num, 100, 1, 1), ctx=ctx)
        gen   = model.generate(cond=cond)

        comkw = dict(iter=i, net=netnam, net_epoch=net_epoch)

        gen_n = '%s<%s>'%tuple([config.nets[config.model.kwargs.generator].get(nam, '') for nam in ['name', 'type']])
        save_aligned_image(data=gen,
                           fpath=config.exsub(templ, data_desc='0to9', ftype='png', **comkw),
                           bbox=config.data.bbox,
                           what='%s(…|0,…,9)'%(gen_n, epoch))

@register_command
def test(args, config):
    if config.log:
        mkfilelogger('ecGAN', config.sub('log'), logging.DEBUG if config.get('debug') else logging.INFO)

    ctx        = ress(make_ctx, config.device, config.device_id)
    batch_size = config.batch_size
    data       = ress(data_funcs[config.data.func], *(config.data.args), ctx=ctx, **(config.data.kwargs))

    model      = models[config.model.type](ctx=ctx, config=config, **config.model.kwargs)
    metr, acc  = model.test(data=data, batch_size=batch_size)

    top_n      = '%s<%s>'%tuple([config.nets[config.model.kwargs.classifier].get(nam, '') for nam in ['name', 'type']])
    getLogger('ecGAN').info('%s("%s"): %s=%.4f', top_n, config.data.func, metr, acc)

@register_command
def test_gan(args, config):
    if config.log:
        mkfilelogger('ecGAN', config.sub('log'), logging.DEBUG if config.get('debug') else logging.INFO)

    ctx        = ress(make_ctx, config.device, config.device_id)
    K          = 10
    model      = models[config.model.type](ctx=ctx, config=config, **config.model.kwargs)
    metr, acc  = model.test(K=K, num=10000, batch_size=config.batch_size)

    gen_n      = '%s<%s>'%tuple([config.nets[config.model.kwargs.generator]    .get(nam, '') for nam in ['name', 'type']])
    top_n      = '%s<%s>'%tuple([config.nets[config.model.kwargs.discriminator].get(nam, '') for nam in ['name', 'type']])
    getLogger('ecGAN').info('%s(%s(…)): %s=%.4f', top_n, gen_n, metr, acc)

@register_command
def explain(args, config):
    if config.model.type == 'CGAN':
        return explain_cgan(args, config)

    if config.log:
        mkfilelogger('ecGAN', config.sub('log'), logging.DEBUG if config.get('debug') else logging.INFO)

    ctx       = ress(make_ctx, config.device, config.device_id)

    model     = models[config.model.type](ctx=ctx, config=config, **config.model.kwargs)

    dataset   = ress(data_funcs[config.data.func], *(config.data.args), ctx=ctx, **(config.data.kwargs))
    data_iter = gluon.data.DataLoader(dataset, 30, shuffle=False, last_batch='discard')
    K         = len(dataset.classes)

    net_desc  = config.nets[config.model.kwargs.classifier]
    netnam    = '%s<%s>'%(net_desc.name, net_desc.type)
    net_epoch = net_desc.epoch
    templ     = config.explanation.output

    if args.seed:
        random.seed(args.seed)

    for i, (data, label) in enumerate(data_iter):
        if i >= args.iter:
            break
        data      = data.as_in_context(ctx)
        label     = label.as_in_context(ctx)

        comkw     = dict(iter=i, net=netnam, net_epoch=net_epoch)

        relevance = model.explain(data, cond=label.squeeze(), single_out=config.explanation.get('single_out', False), mkwargs=config.explanation.get('kwargs', {}))

        save_predictions(model._out.argmax(axis=1), config.exsub(templ, data_desc='prediction', ftype='json', **comkw))
        save_predictions(label                    , config.exsub(templ, data_desc='truth'     , ftype='json', **comkw))

        save_aligned_image(data, config.exsub(templ, data_desc='input<%s>'%config.data.func, ftype='png', **comkw), config.data.bbox)
        save_colorized_image(relevance,
                               config.exsub(templ, data_desc='relevance<%s>'%config.data.func, ftype='png', **comkw),
                               center=config.get('cmap_center'),
                               cmap=config.get('cmap', 'hot'))

@register_command
def explain_cgan(args, config):
    ctx = ress(make_ctx, config.device, config.device_id)

    if config.log:
        mkfilelogger('ecGAN', config.sub('log'), logging.DEBUG if config.get('debug') else logging.INFO)

    model     = models[config.model.type](ctx=ctx, config=config, **config.model.kwargs)

    num       = 30
    K         = args.classnum
    cond_flat = nd.repeat(nd.arange(K, ctx=ctx), (num-1)//K+1)[:num]
    cond      = nd.one_hot(cond_flat, K).reshape((num, K, 1, 1))
    top_desc  = config.nets[config.model.kwargs.discriminator]
    netnam    = top_desc.name
    net_epoch = top_desc.epoch
    templ     = config.explanation.output

    if args.seed:
        random.seed(args.seed)

    for i in range(args.iter):
        noise = nd.random_normal(shape=(num, 100, 1, 1), ctx=ctx)
        args   , kwargs = [K, noise, cond], dict(single_out=config.explanation.get('single_out', False),  mkwargs=config.explanation.get('kwargs', {}))
        s_gen  , gen    = model.explain_top(*args, **kwargs)
        s_noise, s_cond = model.explain    (*args, **kwargs)

        comkw           = dict(iter=i, net=netnam, net_epoch=net_epoch)

        save_predictions(model._out.argmax(axis=1), config.exsub(templ, data_desc='prediction', ftype='json', **comkw))
        save_predictions(cond_flat                , config.exsub(templ, data_desc='truth'     , ftype='json', **comkw))

        save_cgan_visualization(asnumpy(noise.squeeze()), asnumpy(cond.squeeze()),
                                config.exsub(templ, data_desc='input<bar>', ftype='png', **comkw))

        save_data_h5(s_noise.squeeze(), config.exsub(templ, data_desc='relevance<noise>', ftype='h5', **comkw))
        save_data_h5(s_cond.squeeze() , config.exsub(templ, data_desc='relevance<cond>' , ftype='h5', **comkw))

        save_cgan_visualization(asnumpy(s_noise.squeeze()), asnumpy(s_cond.squeeze()),
                                config.exsub(templ, data_desc='relevance<bar>', ftype='png', **comkw))


        save_aligned_image(gen, config.exsub(templ, data_desc='input<gen>', ftype='png', **comkw), config.data.bbox, what='generated input data')
        save_colorized_image(s_gen, config.exsub(templ, data_desc='relevance<gen>', ftype='png', **comkw), center=0., cmap=config.cmap, what='top explanation')
        save_data_h5(s_gen, config.exsub(templ, data_desc='relevance<gen>', ftype='h5', **comkw))

@register_command
def learn_pattern(args, config):
    ctx = ress(make_ctx, config.device, config.device_id)
    batch_size = config.batch_size
    data = ress(data_funcs[config.data.func], *(config.data.args), ctx=ctx, **(config.data.kwargs))

    if config.log:
        mkfilelogger('ecGAN', config.sub('log'), logging.DEBUG if config.get('debug') else logging.INFO)

    model = models[config.model.type](ctx=ctx, config=config, **config.model.kwargs)
    model.load_pattern_params()
    model.learn_pattern(data, batch_size)

@register_command
def fit_pattern(args, config):
    ctx        = ress(make_ctx, config.device, config.device_id)
    batch_size = config.batch_size
    data       = ress(data_funcs[config.data.func], *(config.data.args), ctx=ctx, **(config.data.kwargs))

    if config.log:
        mkfilelogger('ecGAN', config.sub('log'), logging.DEBUG if config.get('debug') else logging.INFO)

    model = models[config.model.type](ctx=ctx, config=config, **config.model.kwargs)
    model.load_pattern_params()
    model.fit_pattern(data, batch_size)

    del model

@register_command
def stats_assess_pattern(args, config):
    ctx = ress(make_ctx, config.device, config.device_id)
    batch_size = config.batch_size
    data = ress(data_funcs[config.data.func], *(config.data.args), ctx=ctx, **(config.data.kwargs))

    if config.log:
        mkfilelogger('ecGAN', config.sub('log'), logging.DEBUG if config.get('debug') else logging.INFO)

    model = models[config.model.type](ctx=ctx, config=config, **config.model.kwargs)
    model.load_pattern_params()
    model.stats_assess_pattern(data, batch_size)

@register_command
def fit_assess_pattern(args, config):
    ctx = ress(make_ctx, config.device, config.device_id)
    batch_size = config.batch_size
    data = ress(data_funcs[config.data.func], *(config.data.args), ctx=ctx, **(config.data.kwargs))

    if config.log:
        mkfilelogger('ecGAN', config.sub('log'), logging.DEBUG if config.get('debug') else logging.INFO)

    model = models[config.model.type](ctx=ctx, config=config, **config.model.kwargs)
    model.load_pattern_params()
    model.fit_assess_pattern(data, batch_size)

@register_command
def assess_pattern(args, config):
    ctx = ress(make_ctx, config.device, config.device_id)

    if config.log:
        mkfilelogger('ecGAN', config.sub('log'), logging.DEBUG if config.get('debug') else logging.INFO)

    model = models[config.model.type](ctx=ctx, config=config, **config.model.kwargs)
    model.load_pattern_params()
    rho = model.assess_pattern()

    txt = ' '.join([str(elem.mean().asscalar()) for elem in rho if elem is not None])
    getLogger('ecGAN').info('Pattern Qualities rho(s) = %s'%txt)

@register_command
def explain_pattern(args, config):
    if config.model.type == 'CGAN':
        return explain_pattern_cgan(args, config)

    ctx = ress(make_ctx, config.device, config.device_id)

    if config.log:
        mkfilelogger('ecGAN', config.sub('log'), logging.DEBUG if config.get('debug') else logging.INFO)

    model = models[config.model.type](ctx=ctx, config=config, **config.model.kwargs)
    model.load_pattern_params()

    dataset = ress(data_funcs[config.data.func], *(config.data.args), ctx=ctx, **(config.data.kwargs))
    data_iter = gluon.data.DataLoader(dataset, 30, shuffle=False, last_batch='discard')

    K = len(dataset.classes)

    net_desc = config.nets[config.model.kwargs.classifier]
    netnam = '%s<%s>'%(net_desc.name, net_desc.type)
    net_epoch = net_desc.epoch
    templ = config.pattern.output

    if args.seed:
        random.seed(args.seed)

    for i, (data, label) in enumerate(data_iter):
        if i >= args.iter:
            break
        data = data.as_in_context(ctx)
        label = label.as_in_context(ctx)

        comkw = dict(iter=i, net=netnam, net_epoch=net_epoch)

        relevance = model.explain_pattern(data, cond=label.squeeze(), single_out=config.pattern.get('single_out', True), attribution=config.pattern.get('type') == 'attribution')

        relevance = relevance.reshape(data.shape)

        save_predictions(model._out.argmax(axis=1), config.exsub(templ, data_desc='prediction', ftype='json', **comkw))
        save_predictions(label                    , config.exsub(templ, data_desc='truth'     , ftype='json', **comkw))

        save_aligned_image(data, config.exsub(templ, data_desc='input<%s>'%config.data.func, ftype='png', **comkw), config.data.bbox)
        save_colorized_image(relevance,
                               config.exsub(templ, data_desc='pattern<%s>'%config.data.func, ftype='png', **comkw),
                               center=config.get('cmap_center'),
                               cmap=config.get('cmap', 'hot'))

    del model

@register_command
def explain_pattern_cgan(args, config):
    ctx = ress(make_ctx, config.device, config.device_id)

    if config.log:
        mkfilelogger('ecGAN', config.sub('log'), logging.DEBUG if config.get('debug') else logging.INFO)

    model = models[config.model.type](ctx=ctx, config=config, **config.model.kwargs)
    model.load_pattern_params()

    num       = 30
    K         = args.classnum
    cond_flat = nd.repeat(nd.arange(K, ctx=ctx), (num-1)//K+1)[:num]
    cond      = nd.one_hot(cond_flat, K).reshape((num, K, 1, 1))
    net_desc  = config.nets[config.model.kwargs.discriminator]
    netnam    = '%s<%s>'%(net_desc.name, net_desc.type)
    net_epoch = net_desc.epoch
    templ     = config.pattern.output

    if args.seed:
        random.seed(args.seed)

    for i in range(args.iter):
        noise = nd.random_normal(shape=(num, 100, 1, 1), ctx=ctx)
        args   , kwargs = [K, noise, cond], dict(single_out=config.pattern.get('single_out', True), attribution=config.pattern.get('type') == 'attribution')
        s_gen  , gen    = model.explain_pattern_top(*args, **kwargs)
        s_noise, s_cond = model.explain_pattern    (*args, **kwargs)

        comkw = dict(iter=i, net=netnam, net_epoch=net_epoch)

        save_predictions(model._out.argmax(axis=1), config.exsub(templ, data_desc='prediction', ftype='json', **comkw))
        save_predictions(cond_flat                , config.exsub(templ, data_desc='truth'     , ftype='json', **comkw))

        #save_raw_image(noise.squeeze(), config.exsub(templ, iter=i, data_desc='input.noise', ftype='png'))
        save_cgan_visualization(asnumpy(noise.squeeze()), asnumpy(cond.squeeze()), config.exsub(templ, data_desc='input<bar>', ftype='png', **comkw))

        save_data_h5(s_noise.squeeze(), config.exsub(templ, data_desc='pattern<noise>', ftype='h5', **comkw))
        save_data_h5(s_cond.squeeze(), config.exsub(templ, data_desc='pattern<cond>', ftype='h5', **comkw))

        save_cgan_visualization(asnumpy(s_noise.squeeze()), asnumpy(s_cond.squeeze()), config.exsub(templ, data_desc='pattern<bar>', ftype='png', **comkw))

        save_aligned_image(gen, config.exsub(templ, data_desc='input<gen>', ftype='png', **comkw), config.data.bbox, what='generated input data')
        save_colorized_image(s_gen, config.exsub(templ, data_desc='pattern<gen>', ftype='png', **comkw), center=0., cmap=config.cmap, what='top explanation')
        save_data_h5(s_gen, config.exsub(templ, data_desc='pattern<gen>', ftype='h5', **comkw))

@register_command
def predict(args, config):
    ctx = ress(make_ctx, config.device, config.device_id)

    if config.log:
        mkfilelogger('ecGAN', config.sub('log'), logging.DEBUG if config.get('debug') else logging.INFO)

    model = models[config.model.type](ctx=ctx, config=config, **config.model.kwargs)

    for i in range(args.iter):
        output = asnumpy(model.predict())
        txt = '\n'.join([' '.join(['%.02f'%val for val in line]) for line in output])

        getLogger('ecGAN').info('Prediction for \'%s\':\n%s', config.nets[config.explanation.top_net].param, txt)

@register_command
def check_bias(args, config):
    ctx = ress(make_ctx, config.device, config.device_id)

    if config.log:
        mkfilelogger('ecGAN', config.sub('log'), logging.DEBUG if config.get('debug') else logging.INFO)

    model = models[config.model.type](ctx=ctx, config=config, **config.model.kwargs)

    txt = 'At least one bias of nets %s is positive!'
    if model.check_bias():
        txt = 'All bias\' of nets %s are negative!'

    nets = ', '.join(['%s<%s>'%(net.name, net.type) for net in config.nets.values() if net.active])

    getLogger('ecGAN').info(txt, nets)

if __name__ == '__main__':
    main()

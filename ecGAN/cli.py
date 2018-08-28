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
from .plot import plot_data, save_data_h5, load_data_h5, save_colorized_image, save_aligned_image, save_raw_image, save_cgan_visualization, save_predictions
from .func import linspace, asnumpy
from .sampler import samplers, asim
from .visualizer import visualizers

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
def generate(args, config):
    if config.log:
        mkfilelogger('ecGAN', config.sub('log'), logging.DEBUG if config.get('debug') else logging.INFO)

    ctx       = ress(make_ctx, config.device, config.device_id)
    model     = models[config.model.type](ctx=ctx, config=config, **config.model.kwargs)
    num       = 30
    K         = args.classnum
    net_desc  = config.nets[config.model.kwargs.generator]
    netnam    = net_desc.name
    net_epoch = net_desc.epoch
    templ     = config.genout

    for i in range(config.iterations):
        noise, cond = samplers[config.sampler.type](num, K, ctx)
        gen   = model.generate(noise=noise, cond=cond)

        comkw = dict(iter=i, net=netnam, net_epoch=net_epoch)

        gen_n = '%s<%s>'%tuple([config.nets[config.model.kwargs.generator].get(nam, '') for nam in ['name', 'type']])
        save_aligned_image(data=gen,
                           fpath=config.exsub(templ, data_desc='%s'%config.sampler.type, ftype='png', **comkw),
                           bbox=config.data.bbox,
                           what='%s(%s(…))'%(gen_n, config.sampler.type))


@register_command
def explain(args, config):
    if config.model.type == 'CGAN':
        return explain_cgan(args, config)
    elif config.model.type == 'Classifier':
        return explain_clss(args, config)
    else:
        raise NotImplementedError

def explain_clss(args, config):
    ctx = ress(make_ctx, config.device, config.device_id)

    if config.log:
        mkfilelogger('ecGAN', config.sub('log'), logging.DEBUG if config.get('debug') else logging.INFO)

    model = models[config.model.type](ctx=ctx, config=config, **config.model.kwargs)
    if config.use_pattern:
        model.load_pattern_params()

    dataset = ress(data_funcs[config.data.func], *(config.data.args), ctx=ctx, **(config.data.kwargs))
    data_iter = gluon.data.DataLoader(dataset, config.explanation.batch_size, shuffle=False, last_batch='discard')

    K = len(dataset.classes)

    net_desc = config.nets[config.model.kwargs.classifier]
    netnam = '%s<%s>'%(net_desc.name, net_desc.type)
    net_epoch = net_desc.epoch
    if config.use_pattern:
        templ = config.pattern.output
    else:
        templ = config.explanation.output

    if args.seed:
        random.seed(args.seed)

    for i, (data, label) in enumerate(data_iter):
        if i >= config.explanation.iterations:
            break

        comkw = dict(iter=i, net=netnam, net_epoch=net_epoch)
        fpath = config.exsub(templ, data_desc='result<%s>'%config.data.func, ftype='h5', **comkw)
        if not config.overwrite and os.path.isfile(fpath):
            getLogger('ecGAN').info('File already exits, skipping \'%s\'...', fpath)
            continue

        data = data.as_in_context(ctx)
        label = label.as_in_context(ctx)

        if config.use_pattern:
            relevance = model.explain_pattern(data, cond=label.squeeze(), single_out=config.pattern.get('single_out', True), attribution=config.pattern.get('type') == 'attribution')
        else:
            relevance = model.explain(data, cond=label.squeeze(), single_out=config.explanation.get('single_out', False), mkwargs=config.explanation.get('kwargs', {}))

        info = {
            'input'      : data                     .asnumpy(),
            'prediction' : model._out.argmax(axis=1).asnumpy(),
            'label'      : label                    .asnumpy(),
            'relevance'  : relevance                .asnumpy(),
        }
        save_data_h5(info, fpath)

    del model

def explain_cgan(args, config):
    ctx = ress(make_ctx, config.device, config.device_id)

    if config.log:
        mkfilelogger('ecGAN', config.sub('log'), logging.DEBUG if config.get('debug') else logging.INFO)

    model = models[config.model.type](ctx=ctx, config=config, **config.model.kwargs)
    if config.use_pattern:
        model.load_pattern_params()

    num       = config.explanation.batch_size
    K         = args.classnum
    net_desc  = config.nets[config.model.kwargs.discriminator]
    netnam    = '%s<%s>'%(net_desc.name, net_desc.type)
    net_epoch = net_desc.epoch
    if config.use_pattern:
        templ     = config.pattern.output
    else:
        templ     = config.explanation.output

    if args.seed:
        random.seed(args.seed)

    for i in range(config.explanation.iterations):

        comkw = dict(iter=i, net=netnam, net_epoch=net_epoch)
        fpath = config.exsub(templ, data_desc='result<%s>'%config.sampler.type, ftype='h5', **comkw)
        if not config.overwrite and os.path.isfile(fpath):
            getLogger('ecGAN').info('File already exits, skipping \'%s\'...', fpath)
            continue

        noise, cond = samplers[config.sampler.type](num, K, ctx)

        args = [K] + asim(noise, cond)
        if config.use_pattern:
            kwargs = dict(single_out=config.pattern.get('single_out', True), attribution=config.pattern.get('type') == 'attribution')
            s_gen  , gen    = model.explain_pattern_top(*args, **kwargs)
            s_noise, s_cond = model.explain_pattern    (*args, **kwargs)
        else:
            kwargs = dict(single_out=config.explanation.get('single_out', False),  mkwargs=config.explanation.get('kwargs', {}))
            s_gen  , gen    = model.explain_top(*args, **kwargs)
            s_noise, s_cond = model.explain    (*args, **kwargs)

        info = {
            'input/noise'          : noise                    .asnumpy(),
            'input/cond'           : cond                     .asnumpy(),
            'generated'            : gen                      .asnumpy(),
            'prediction'           : model._out.argmax(axis=1).asnumpy(),
            'label'                : cond      .argmax(axis=1).asnumpy(),
            'relevance/noise'      : s_noise.squeeze()        .asnumpy(),
            'relevance/cond'       : s_cond.squeeze()         .asnumpy(),
            'relevance/generated'  : s_gen                    .asnumpy(),
        }
        save_data_h5(info, fpath)

@register_command
def visualize(args, config):
    if config.log:
        mkfilelogger('ecGAN', config.sub('log'), logging.DEBUG if config.get('debug') else logging.INFO)

    if args.seed:
        random.seed(args.seed)

    Visualizer = visualizers[config.visualizer.type]

    with Visualizer(config, *config.visualizer.get('args', []), **config.visualizer.get('kwargs', {})) as vs:
        for i in range(config.explanation.iterations):
            vs.feed()

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
def predict(args, config):
    ctx = ress(make_ctx, config.device, config.device_id)

    if config.log:
        mkfilelogger('ecGAN', config.sub('log'), logging.DEBUG if config.get('debug') else logging.INFO)

    model = models[config.model.type](ctx=ctx, config=config, **config.model.kwargs)

    for i in range(config.iterations):
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


@register_command
def analyze_explanation(args, config):
    if config.log:
        mkfilelogger('ecGAN', config.sub('log'), logging.DEBUG if config.get('debug') else logging.INFO)

    comkw = dict(iter=i, net=netnam, net_epoch=net_epoch)
    templ = config.pattern.output
    with h5py.File(config.exsub(templ, data_desc='pattern<noise>', ftype='h5', **comkw)) as fp:
        noise = fp['data']
    with h5py.File(config.exsub(templ, data_desc='pattern<cond>', ftype='h5', **comkw)) as fp:
        cond = fp['data']




if __name__ == '__main__':
    main()

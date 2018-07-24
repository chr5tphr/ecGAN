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
from .util import mkfilelogger, Config, config_ctx, make_ctx, load_module_file, ChainConfig, RessourceManager
from .plot import plot_data, save_data_h5, save_colorized_image, save_aligned_image, save_raw_image, save_cgan_visualization, save_predictions
from .func import linspace

getLogger = logging.getLogger

commands = {}
def register_command(func):
    commands[func.__name__] = func
    return func

ress = RessourceManager()

def main():
    parser = ArgumentParser()

    parser.add_argument('command', choices=commands.keys())
    parser.add_argument('-f', '--config', action='append', default=[])
    parser.add_argument('-c', '--chain', action='append', default=[])
    parser.add_argument('-u', '--update', action='append', default=[])
    parser.add_argument('--epoch_range', nargs=3, type=int)
    parser.add_argument('--iter', type=int, default=1)
    parser.add_argument('--seed', type=int)
    parser.add_argument('--classnum', type=int, default=10)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('-k', '--pskip', action='append', type=int, default=[])

    args = parser.parse_args(sys.argv[1:])

    if len(args.config):
        config = Config()

        for cpath in args.config:
            config.update_from_file(cpath)

        for ustr in args.update:
            config.update(yaml.safe_load(ustr))

        net_module = load_module_file(config.sub('net_file'), 'net_module')
    else:
        config = None

    if args.seed:
        random.seed(args.seed)

    if args.debug:
        import ipdb; ipdb.set_trace()

    commands[args.command](args, config)

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
def chain(args, config):
    for fname in args.chain:
        with open(fname,'r') as fp:
            rawdict = yaml.safe_load(fp)
        content = rawdict.pop('dict', {})
        ctree = ChainConfig(content, **rawdict)

        for leaf in ctree.leaves():
            if leaf._priority in args.pskip:
                continue
            net_module = ress(load_module_file, leaf.sub('net_file'), 'net_module')
            try:
                commands[leaf._action](args, leaf)
            except:
                print(traceback.format_exc(), file=sys.stderr)


@register_command
def print_config(args, config):
    print(yaml.safe_dump(config.raw(), default_flow_style=False))

@register_command
def train(args, config):
    # print(yaml.safe_dump(config.raw(), default_flow_style=False))
    ctx = ress(make_ctx, config.device, config.device_id)

    batch_size = config.batch_size
    nepochs = config.nepochs

    data = ress(data_funcs[config.data.func], *(config.data.args), ctx=ctx, **(config.data.kwargs))

    if config.log:
        mkfilelogger('ecGAN', config.sub('log'), logging.DEBUG if config.get('debug') else logging.INFO)

    model = models[config.model](ctx=ctx, config=config)
    model.train(data, batch_size, nepochs)

    del model


@register_command
def generate(args, config):
    ctx = ress(make_ctx, config.device, config.device_id)

    if config.log:
        mkfilelogger('ecGAN', config.sub('log'), logging.DEBUG if config.get('debug') else logging.INFO)

    model = models[config.model](ctx=ctx, config=config)

    num = 30
    K = args.classnum
    cond = nd.one_hot(linspace(0, K, num, ctx=ctx, dtype='int32'), K).reshape(num, K, 1, 1)
    netnam = config.nets.generator.name
    net_epoch = config.nets.generator.epoch
    templ = config.genout

    for i in range(args.iter):
        noise = nd.random_normal(shape=(num, 100, 1, 1), ctx=ctx)
        gen = model.generate(cond=cond)

        comkw = dict(iter=i, net=netnam, net_epoch=net_epoch)

        gen_n = '%s<%s>'%tuple([config.nets.generator.get(nam, '') for nam in ['name', 'type']])
        save_aligned_image(data=gen,
                           fpath=config.exsub(templ, data_desc='0to9', ftype='png', **comkw),
                           bbox=config.data.bbox,
                           what='%s(…|0,…,9)'%(gen_n, epoch))

@register_command
def test(args, config):
    ctx = ress(make_ctx, config.device, config.device_id)

    if config.log:
        mkfilelogger('ecGAN', config.sub('log'), logging.DEBUG if config.get('debug') else logging.INFO)

    batch_size = config.batch_size
    data = ress(data_funcs[config.data.func], *(config.data.args), ctx=ctx, **(config.data.kwargs))

    model = models[config.model](ctx=ctx, config=config)
    metr, acc = model.test(data=data, batch_size=batch_size)

    top_n = '%s<%s>'%tuple([config.nets.classifier.get(nam, '') for nam in ['name', 'type']])
    getLogger('ecGAN').info('%s("%s"): %s=%.4f', top_n, config.data.func, metr, acc)

@register_command
def test_gan(args, config):
    ctx = ress(make_ctx, config.device, config.device_id)

    if config.log:
        mkfilelogger('ecGAN', config.sub('log'), logging.DEBUG if config.get('debug') else logging.INFO)

    batch_size = config.batch_size
    #K = config.nets.get(config.nets.generator.top, 'discriminator').kwargs.get('outnum', 10)
    K = 10

    model = models[config.model](ctx=ctx, config=config)
    metr, acc = model.test(K=K, num=10000, batch_size=batch_size)

    gen_n = '%s<%s>'%tuple([config.nets.generator.get(nam, '') for nam in ['name', 'type']])
    top_n = '%s<%s>'%tuple([config.nets.get(config.nets.generator.get('top', 'discriminator')).get(nam, '') for nam in ['name', 'type']])
    getLogger('ecGAN').info('%s(%s(…)): %s=%.4f', top_n, gen_n, metr, acc)

@register_command
def debug(args, config):
    ctx = ress(make_ctx, config.device, config.device_id)

    if config.log:
        mkfilelogger('ecGAN', config.sub('log'), logging.DEBUG if config.get('debug') else logging.INFO)

    batch_size = config.batch_size
    data = ress(data_funcs[config.data.func], *(config.data.args), ctx=ctx, **(config.data.kwargs))

    model = models[config.model](ctx=ctx, config=config)

    import ipdb; ipdb.set_trace()

@register_command
def explain(args, config):
    ctx = ress(make_ctx, config.device, config.device_id)

    if config.log:
        mkfilelogger('ecGAN', config.sub('log'), logging.DEBUG if config.get('debug') else logging.INFO)

    model = models[config.model](ctx=ctx, config=config)

    data_fp = ress(data_funcs[config.data.func], *(config.data.args), ctx=ctx, **(config.data.kwargs))
    data_iter = gluon.data.DataLoader(data_fp, 30, shuffle=False, last_batch='discard')

    netnam = config.nets.classifier.name
    net_epoch = config.nets.classifier.epoch
    templ = config.explanation.output

    if args.seed:
        random.seed(args.seed)

    for i, (data, label) in enumerate(data_iter):
        if i >= args.iter:
            break
        data = data.as_in_context(ctx)
        label = label.as_in_context(ctx)

        comkw = dict(iter=i, net=netnam, net_epoch=net_epoch)

        relevance = model.explain(data, mkwargs=config.explanation.get('kwargs', {}))

        save_aligned_image(data, config.exsub(templ, data_desc='input.%s'%config.data.func, ftype='png', **comkw), config.data.bbox)
        save_colorized_image(relevance,
                               config.exsub(templ, data_desc=config.data.func, ftype='png', **comkw),
                               center=config.get('cmap_center'),
                               cmap=config.get('cmap', 'hot'))

@register_command
def explain_cgan(args, config):
    ctx = ress(make_ctx, config.device, config.device_id)

    if config.log:
        mkfilelogger('ecGAN', config.sub('log'), logging.DEBUG if config.get('debug') else logging.INFO)

    model = models[config.model](ctx=ctx, config=config)

    num = 30
    K = args.classnum
    cond = nd.one_hot(linspace(0, K, num, ctx=ctx, dtype='int32'), K).reshape(num, K, 1, 1)
    netnam = config.nets.get(config.nets.generator.top, 'discriminator').name
    net_epoch = config.nets.get(config.nets.generator.top, 'discriminator').epoch
    templ = config.explanation.output

    if args.seed:
        random.seed(args.seed)

    for i in range(args.iter):
        noise = nd.random_normal(shape=(num, 100, 1, 1), ctx=ctx)
        s_gen, gen = model.explain_top(K, noise, cond, mkwargs=config.explanation.get('kwargs', {}))
        s_noise, s_cond = model.explain(K, noise, cond, mkwargs=config.explanation.get('kwargs', {}))

        comkw = dict(iter=i, net=netnam, net_epoch=net_epoch)

        save_predictions(model._out.argmax(axis=1), config.exsub(templ, data_desc='prediction', ftype='json', **comkw))

        #save_raw_image(noise.squeeze(), config.exsub(templ, iter=i, data_desc='input.noise', ftype='png'))
        save_cgan_visualization(noise.squeeze().asnumpy(), cond.squeeze().asnumpy(),
                                config.exsub(templ, data_desc='input.bar', ftype='png', **comkw))

        save_data_h5(s_noise.squeeze(), config.exsub(templ, data_desc='noise', ftype='h5', **comkw))
        save_data_h5(s_cond.squeeze(), config.exsub(templ, data_desc='cond', ftype='h5', **comkw))

        save_cgan_visualization(s_noise.squeeze().asnumpy(), s_cond.squeeze().asnumpy(),
                                config.exsub(templ, data_desc='bar', ftype='png', **comkw))


        save_aligned_image(gen, config.exsub(templ, data_desc='input.gen', ftype='png', **comkw), config.data.bbox, what='generated input data')
        save_colorized_image(s_gen, config.exsub(templ, data_desc='gen', ftype='png', **comkw), center=0., cmap='bwr', what='top explanation')
        save_data_h5(s_gen, config.exsub(templ, data_desc='gen', ftype='h5', **comkw))

@register_command
def explain_gan(args, config):
    ctx = ress(make_ctx, config.device, config.device_id)

    if config.log:
        mkfilelogger('ecGAN', config.sub('log'), logging.DEBUG if config.get('debug') else logging.INFO)

    model = models[config.model](ctx=ctx, config=config)

    label = None
    if config.model in ['CCGAN', 'CGAN', 'WCGAN']:
        K = args.classnum
        label = nd.one_hot(linspace(0, K, 30, ctx=ctx, dtype='int32'), K)

    for i in range(args.iter):
        relTop, Rtc, relG, Rc, noise, gen = model.explain(label=label)

        # TODO FIX
        save_explanation(relTop,
                         data=gen,
                         config=config,
                         image=config.explanation.image,
                         output=config.explanation.output,
                         source=config.explanation.input,
                         data_desc='%s.%s'%(config.nets.generator.type, config.start_epoch),
                         net=config.explanation.top_net,
                         i=i)
        # TODO FIX
        save_explanation(relG,
                         data=noise,
                         config=config,
                         image=config.explanation.image,
                         output=config.explanation.output,
                         source=config.explanation.input,
                         data_desc='noise',
                         net='generator',
                         i=i)

        if Rc is not None:
            save_explanation(Rc,
                             data=label,
                             config=config,
                             image=config.explanation.image,
                             output=config.explanation.output,
                             source=config.explanation.input,
                             data_desc='condition',
                             net='generator',
                             i=i)

@register_command
def learn_pattern(args, config):
    ctx = ress(make_ctx, config.device, config.device_id)
    batch_size = config.batch_size
    data = ress(data_funcs[config.data.func], *(config.data.args), ctx=ctx, **(config.data.kwargs))

    if config.log:
        mkfilelogger('ecGAN', config.sub('log'), logging.DEBUG if config.get('debug') else logging.INFO)

    model = models[config.model](ctx=ctx, config=config)
    model.load_pattern_params()
    model.learn_pattern(data, batch_size)

@register_command
def fit_pattern(args, config):
    ctx = ress(make_ctx, config.device, config.device_id)
    batch_size = config.batch_size
    data = ress(data_funcs[config.data.func], *(config.data.args), ctx=ctx, **(config.data.kwargs))

    if config.log:
        mkfilelogger('ecGAN', config.sub('log'), logging.DEBUG if config.get('debug') else logging.INFO)

    model = models[config.model](ctx=ctx, config=config)
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

    model = models[config.model](ctx=ctx, config=config)
    model.load_pattern_params()
    model.stats_assess_pattern(data, batch_size)

@register_command
def fit_assess_pattern(args, config):
    ctx = ress(make_ctx, config.device, config.device_id)
    batch_size = config.batch_size
    data = ress(data_funcs[config.data.func], *(config.data.args), ctx=ctx, **(config.data.kwargs))

    if config.log:
        mkfilelogger('ecGAN', config.sub('log'), logging.DEBUG if config.get('debug') else logging.INFO)

    model = models[config.model](ctx=ctx, config=config)
    model.load_pattern_params()
    model.fit_assess_pattern(data, batch_size)

@register_command
def assess_pattern(args, config):
    ctx = ress(make_ctx, config.device, config.device_id)

    if config.log:
        mkfilelogger('ecGAN', config.sub('log'), logging.DEBUG if config.get('debug') else logging.INFO)

    model = models[config.model](ctx=ctx, config=config)
    model.load_pattern_params()
    rho = model.assess_pattern()

    txt = ' '.join([str(elem.mean().asscalar()) for elem in rho if elem is not None])
    getLogger('ecGAN').info('Pattern Qualities rho(s) = %s'%txt)

@register_command
def explain_pattern(args, config):
    ctx = ress(make_ctx, config.device, config.device_id)

    if config.log:
        mkfilelogger('ecGAN', config.sub('log'), logging.DEBUG if config.get('debug') else logging.INFO)

    model = models[config.model](ctx=ctx, config=config)
    model.load_pattern_params()

    data_fp = ress(data_funcs[config.data.func], *(config.data.args), ctx=ctx, **(config.data.kwargs))
    data_iter = gluon.data.DataLoader(data_fp, 30, shuffle=False, last_batch='discard')

    netnam = config.nets.classifier.name
    net_epoch = config.nets.classifier.epoch
    templ = config.pattern.output

    for i, (data, label) in enumerate(data_iter):
        if i >= args.iter:
            break
        data = data.as_in_context(ctx)
        label = label.as_in_context(ctx)

        comkw = dict(iter=i, net=netnam, net_epoch=net_epoch)

        relevance = model.explain_pattern(data, attribution=config.pattern.get('type') == 'attribution')

        relevance = relevance.reshape(data.shape)

        save_aligned_image(data, config.exsub(templ, data_desc='input.%s'%config.data.func, ftype='png', **comkw), config.data.bbox)
        save_colorized_image(relevance,
                               config.exsub(templ, data_desc=config.data.func, ftype='png', **comkw),
                               center=config.get('cmap_center'),
                               cmap=config.get('cmap', 'hot'))

    del model

@register_command
def explain_pattern_cgan(args, config):
    ctx = ress(make_ctx, config.device, config.device_id)

    if config.log:
        mkfilelogger('ecGAN', config.sub('log'), logging.DEBUG if config.get('debug') else logging.INFO)

    model = models[config.model](ctx=ctx, config=config)
    model.load_pattern_params()

    num = 30
    K = args.classnum
    cond = nd.one_hot(linspace(0, K, num, ctx=ctx, dtype='int32'), K).reshape(num, K, 1, 1)
    # save_raw_image(cond.squeeze(), config.exsub(templ, iter=0, data_desc='input.cond', ftype='png'))
    netnam = config.nets.get(config.nets.generator.top, 'discriminator').name
    net_epoch = config.nets.get(config.nets.generator.top, 'discriminator').epoch
    templ = config.pattern.output

    for i in range(args.iter):
        noise = nd.random_normal(shape=(num, 100, 1, 1), ctx=ctx)
        s_gen, gen = model.explain_pattern_top(K, noise, cond, attribution=config.pattern.get('type') == 'attribution')
        s_noise, s_cond = model.explain_pattern(K, noise, cond, attribution=config.pattern.get('type') == 'attribution')

        comkw = dict(iter=i, net=netnam, net_epoch=net_epoch)

        save_predictions(model._out.argmax(axis=1), config.exsub(templ, data_desc='prediction', ftype='json', **comkw))

        #save_raw_image(noise.squeeze(), config.exsub(templ, iter=i, data_desc='input.noise', ftype='png'))
        save_cgan_visualization(noise.squeeze().asnumpy(), cond.squeeze().asnumpy(), config.exsub(templ, data_desc='input.bar', ftype='png', **comkw))

        save_data_h5(s_noise.squeeze(), config.exsub(templ, data_desc='noise', ftype='h5', **comkw))
        save_data_h5(s_cond.squeeze(), config.exsub(templ, data_desc='cond', ftype='h5', **comkw))

        save_cgan_visualization(s_noise.squeeze().asnumpy(), s_cond.squeeze().asnumpy(), config.exsub(templ, data_desc='bar', ftype='png', **comkw))

        save_aligned_image(gen, config.exsub(templ, data_desc='input.gen', ftype='png', **comkw), config.data.bbox, what='generated input data')
        save_colorized_image(s_gen, config.exsub(templ, data_desc='gen', ftype='png', **comkw), center=0., cmap='bwr', what='top explanation')
        save_data_h5(s_gen, config.exsub(templ, data_desc='gen', ftype='h5', **comkw))

@register_command
def predict(args, config):
    ctx = ress(make_ctx, config.device, config.device_id)

    if config.log:
        mkfilelogger('ecGAN', config.sub('log'), logging.DEBUG if config.get('debug') else logging.INFO)

    model = models[config.model](ctx=ctx, config=config)

    for i in range(args.iter):
        output = model.predict().asnumpy()
        txt = '\n'.join([' '.join(['%.02f'%val for val in line]) for line in output])

        getLogger('ecGAN').info('Prediction for \'%s\':\n%s', config.nets[config.explanation.top_net].param, txt)

if __name__ == '__main__':
    main()

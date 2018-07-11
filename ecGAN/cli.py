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
from .plot import plot_data, save_explanation_data, save_explanation_image, save_source_image, save_source_raw_image, save_cgan_visualization
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

    netG = nets[config.nets.generator.type]()

    if config.log:
        mkfilelogger('ecGAN', config.sub('log'), logging.DEBUG if config.get('debug') else logging.INFO)

    for epoch in (range(*args.epoch_range) if args.epoch_range else [config.start_epoch]):
        if config.nets.generator.param:
            netG.load_params(config.sub('nets.generator.param', start_epoch=epoch), ctx=ctx)
        else:
            netG.initialize(mx.init.Xavier(magnitude=2.24), ctx=ctx)
        noise = nd.random_normal(shape=(30, 100), ctx=ctx)
        gdat = None
        cond = None
        if config.model == 'CGAN':
            cond = nd.one_hot(nd.repeat(nd.arange(10, ctx=ctx), 3), 10)
            gdat = netG(noise, cond)
        else:
            gdat = netG(noise)

        if config.genout:
            bbox = config.data.bbox
            fpath = config.sub('genout', epoch=epoch)
            gdat = ((gdat - bbox[0]) * 255/(bbox[1]-bbox[0])).asnumpy().clip(0, 255).astype(np.uint8)
            # snum = int(len(gdat)**.5)
            # imwrite(fpath, gdat[:snum**2].reshape(snum, snum, 28, 28).transpose(0, 2, 1, 3).reshape(snum*28, snum*28))
            imwrite(fpath, gdat.reshape(5, 6, 28, 28).transpose(0, 2, 1, 3).reshape(5*28, 6*28))
            getLogger('ecGAN').info('Saved generated data by generator \'%s\' checkpoint \'%s\' in \'%s\'.',
                                    config.nets.generator.type, config.sub('nets.generator.param', start_epoch=epoch), fpath)
        else:
            fig = plot_data(gdat)
            fig.show()

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

    for i, (data, label) in enumerate(data_iter):
        if i >= args.iter:
            break
        data = data.as_in_context(ctx)
        label = label.as_in_context(ctx)

        relevance = model.explain(data, mkwargs=config.explanation.get('kwargs', {}))

        save_source_image(data, config.sub('explanation.output', data_desc='input.%s'%config.data.func, iter=i, ftype='png', net=netnam), config.data.bbox)
        save_explanation_image(relevance,
                               config.sub('explanation.output', data_desc=config.data.func, iter=i, ftype='png', net=netnam),
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

    for i in range(args.iter):
        noise = nd.random_normal(shape=(num, 100, 1, 1), ctx=ctx)
        s_noise, s_cond = model.explain(K, noise, cond, mkwargs=config.explanation.get('kwargs', {}))

        out = model.netD._out.argmax(axis=1)
        with open(config.sub('explanation.output', iter=i, data_desc='prediction', ftype='json', net=netnam), 'w') as fp:
            json.dump(out.asnumpy().astype(int).tolist(), fp, indent=2)

        #save_source_raw_image(noise.squeeze(), config.sub('explanation.output', iter=i, data_desc='input.noise', ftype='png'))
        save_cgan_visualization(noise.squeeze().asnumpy(), cond.squeeze().asnumpy(),
                                config.sub('explanation.output', iter=i, data_desc='input.bar', ftype='pdf', net=netnam))

        save_explanation_data(s_noise.squeeze(), config.sub('explanation.output', iter=i, data_desc='noise', ftype='h5', net=netnam))
        save_explanation_data(s_cond.squeeze(), config.sub('explanation.output', iter=i, data_desc='cond', ftype='h5', net=netnam))

        save_cgan_visualization(s_noise.squeeze().asnumpy(), s_cond.squeeze().asnumpy(),
                                config.sub('explanation.output', iter=i, data_desc='bar', ftype='pdf', net=netnam))

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

    for i, (data, label) in enumerate(data_iter):
        if i >= args.iter:
            break
        data = data.as_in_context(ctx)
        label = label.as_in_context(ctx)

        relevance = model.explain_pattern(data, attribution=config.pattern.get('type') == 'attribution')

        relevance = relevance.reshape(data.shape)

        save_source_image(data, config.sub('pattern.output', data_desc='input.%s'%config.data.func, iter=i, ftype='png', net=netnam), config.data.bbox)
        save_explanation_image(relevance,
                               config.sub('pattern.output', data_desc=config.data.func, iter=i, ftype='png', net=netnam),
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
    # save_source_raw_image(cond.squeeze(), config.sub('pattern.output', iter=0, data_desc='input.cond', ftype='png'))
    netnam = config.nets.get(config.nets.generator.top, 'discriminator').name

    for i in range(args.iter):
        noise = nd.random_normal(shape=(num, 100, 1, 1), ctx=ctx)
        s_noise, s_cond = model.explain_pattern(K, noise, cond, attribution=config.pattern.get('type') == 'attribution')

        out = model.netD._out.argmax(axis=1)
        with open(config.sub('pattern.output', iter=i, data_desc='prediction', ftype='json', net=netnam), 'w') as fp:
            json.dump(out.asnumpy().astype(int).tolist(), fp, indent=2)

        #save_source_raw_image(noise.squeeze(), config.sub('pattern.output', iter=i, data_desc='input.noise', ftype='png'))
        save_cgan_visualization(noise.squeeze().asnumpy(), cond.squeeze().asnumpy(), config.sub('pattern.output', iter=0, data_desc='input.bar', ftype='pdf', net=netnam))

        save_explanation_data(s_noise.squeeze(), config.sub('pattern.output', iter=i, data_desc='noise', ftype='h5', net=netnam))
        save_explanation_data(s_cond.squeeze(), config.sub('pattern.output', iter=i, data_desc='cond', ftype='h5', net=netnam))

        save_cgan_visualization(s_noise.squeeze().asnumpy(), s_cond.squeeze().asnumpy(), config.sub('pattern.output', iter=i, data_desc='bar', ftype='pdf', net=netnam))

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

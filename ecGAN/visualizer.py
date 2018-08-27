import matplotlib as mpl
import yaml
import json
import mxnet as mx
import numpy as np
import sys
import os
import logging
import h5py

from .util import mkfilelogger
from .plot import plot_data, save_data_h5, load_data_h5, save_colorized_image, save_aligned_image, save_raw_image, save_cgan_visualization, save_predictions

getLogger = logging.getLogger

visualizers = {}
def register_visualizer(func):
    visualizers[func.__name__] = func
    return func

class Visualizer(object):
    def __init__(self, config):
        self.config = config
        self.comkw  = None
        self.templ  = None

    def feed(self):
        raise NotImplementedError

    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_value, traceback):
        pass

@register_visualizer
class PlainVisualizer(Visualizer):
    @staticmethod
    def _check_write(fpath, func, args={}, kwargs={}):
        if not config.overwrite and os.path.isfile(fpath):
            getLogger('ecGAN').info('\'%s\' already exits, skipping...', fpath)
        else:
            func(*args, fpath=fpath, **kwargs)

    def __enter__(self):
        if self.config.model == 'Classifier':
            top_net = 'classifier'
        elif self.config.model == 'CGAN':
            top_net = 'discriminator'
        else:
            raise NotImplementedError
        net_desc = self.config.nets[self.config.model.kwargs[top_net]]
        self.comkw = {
            'iter'      : 0,
            'net'       : '%s<%s>'%(net_desc.name, net_desc.type),
            'net_epoch' : net_desc.epoch
        }
        if self.config.use_pattern:
            self.templ = self.config.pattern.output
        else:
            self.templ = self.config.explanation.output

    def feed(self):
        if self.config.model == 'CGAN':
            return self._feed_cgan()
        elif self.config.model == 'Classifier':
            return self._feed_clss()
        else:
            raise NotImplementedError

    def _feed_cgan(self):
        info_keys = [
            'input/noise',
            'input/cond',
            'generated',
            'prediction',
            'label',
            'relevance/noise',
            'relevance/cond',
            'relevance/generated',
        ]
        noise, cond, gen, pred, label, s_noise, s_cond, s_gen = load_data_h5(info_keys, self.config.exsub(self.templ, data_desc='result', ftype='h5', **self.comkw))
        files = {
            'pred'      : {
                'fpath'  : self.config.exsub(self.templ, data_desc='prediction', ftype='json', **self.comkw),
                'func'   : save_predictions,
                'args'   : [pred],
                'kwargs' : {},
            },
            'label'     : {
                'fpath'  : self.config.exsub(self.templ, data_desc='truth'     , ftype='json', **self.comkw),
                'func'   : save_predictions,
                'args'   : [label],
                'kwargs' : {},
            },
            'input'     : {
                'fpath'  : self.config.exsub(self.templ, data_desc='input<bar>', ftype='png', **self.comkw)
                'func'   : save_cgan_visualization,
                'args'   : [noise, cond],
                'kwargs' : {},
            },
            'relevance' : {
                'fpath'  : self.config.exsub(self.templ, data_desc='relevance<bar>', ftype='png', **self.comkw),
                'func'   : save_cgan_visualization,
                'args'   : [s_noise, s_cond],
                'kwargs' : {},
            },
            'gen'     : {
                'fpath'  : self.config.exsub(self.templ, data_desc='input<gen>', ftype='png', **self.comkw),
                'func'   : save_aligned_image,
                'args'   : [gen, self.config.data.bbox],
                'kwargs' : {'what': 'generated input data'},
            },
            'genrel' : {
                'fpath'  : self.config.exsub(self.templ, data_desc='relevance<gen>', ftype='png', **self.comkw),
                'func'   : save_colorized_image,
                'args'   : [s_gen],
                'kwargs' : {'center': self.config.get('cmap_center'), 'cmap': self.config.get('cmap', 'bwr'), 'what': 'top explanation'},
            },
        }
        for key, val in files.items():
            self.check_write(val['fpath'], val['func'], val['args'], val['kwargs'])
        self.comkw['iter'] += 1

    def _feed_cls(self):
        info_keys = [
            'input',
            'prediction',
            'label',
            'relevance',
        ]
        data, pred, label, relevance = load_data_h5(info_keys, self.config.exsub(self.templ, data_desc='result', ftype='h5', **self.comkw))

        files = {
            'pred'      : {
                'fpath'  : self.config.exsub(self.templ, data_desc='prediction', ftype='json', **self.comkw),
                'func'   : save_predictions,
                'args'   : [pred],
                'kwargs' : {},
            },
            'label'     : {
                'fpath'  : self.config.exsub(self.templ, data_desc='truth'     , ftype='json', **self.comkw),
                'func'   : save_predictions,
                'args'   : [label],
                'kwargs' : {},
            },
            'input'     : {
                'fpath'  : self.config.exsub(self.templ, data_desc='input<%s>'%self.config.data.func, ftype='png', **self.comkw),
                'func'   : save_aligned_image,
                'args'   : [data, self.config.data.bbox],
                'kwargs' : {},
            },
            'relevance' : {
                'fpath'  : self.config.exsub(self.templ, data_desc='relevance<%s>'%self.config.data.func, ftype='png', **self.comkw),
                'func'   : save_colorized_image,
                'args'   : [relevance],
                'kwargs' : {'center': self.config.get('cmap_center'), 'cmap': self.config.get('cmap', 'bwr')},
            },
        }
        for key, val in files.items():
            self.check_write(val['fpath'], val['func'], val['args'], val['kwargs'])

        self.comkw['iter'] += 1

@register_visualizer
class MeanVisualizer(Visualizer):
    def __init__(self, *args, pred_cond=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.pred_cond = pred_cond

    @staticmethod
    def _check_write(fpath, func, args={}, kwargs={}):
        if not config.overwrite and os.path.isfile(fpath):
            getLogger('ecGAN').info('\'%s\' already exits, skipping...', fpath)
        else:
            func(*args, fpath=fpath, **kwargs)

    def __enter__(self):
        if self.config.model == 'Classifier':
            top_net = 'classifier'
        elif self.config.model == 'CGAN':
            top_net = 'discriminator'
        else:
            raise NotImplementedError
        net_desc = self.config.nets[self.config.model.kwargs[top_net]]
        self.comkw = {
            'iter'      : 0,
            'net'       : '%s<%s>'%(net_desc.name, net_desc.type),
            'net_epoch' : net_desc.epoch
        }
        if self.config.use_pattern:
            self.templ = self.config.pattern.output
        else:
            self.templ = self.config.explanation.output
        self._acc = {}

    def feed(self):
        if self.config.model == 'CGAN':
            return self._feed_cgan()
        elif self.config.model == 'Classifier':
            return self._feed_clss()
        else:
            raise NotImplementedError

    def _feed_cgan(self):
        info_keys = [
            'prediction',
            'label',
            'relevance/noise',
            'relevance/cond',
        ]
        vals = load_data_h5(info_keys, self.config.exsub(self.templ, data_desc='result', ftype='h5', **self.comkw))
        for pred, label, s_noise, s_cond in zip(*vals):
            noise_all = self._acc.setdefault('noise_a', np.zeros_like(s_noise))
            cond_all  = self._acc.setdefault('cond_a', np.zeros_like(s_cond))
            count_all = self._acc.setdefault('count_a', np.zeros([1]))

            noise_all += s_noise
            cond_all  += s_cond
            count_all += 1

            alpha = pred if self.pred_cond else label
            noise_one = self._acc.setdefault('noise', {}).setdefault('%02d'%alpha, np.zeros_like(s_noise))
            cond_one  = self._acc.setdefault('cond',  {}).setdefault('%02d'%alpha, np.zeros_like(s_cond))
            count_one = self._acc.setdefault('count', {}).setdefault('%02d'%alpha, np.zeros([1], dtype=int))

            noise_one += s_noise
            cond_one  += s_cond
            count_one += 1
        self.comkw['iter'] += 1


    def _feed_cls(self):
        raise NotImplementedError

    def _save_data(self, fpath):
        info = {
            'noise/all': self._acc('noise_a'),
            'cond/all': self._acc('cond_a'),
            'count/all': self._acc('count_a'),
        }
        for (key, noise), (_, cond), (_, count) in zip(*[self._acc[mk].items() for mk in ['noise', 'cond', 'count']):
            info['noise/%s'%key] = noise
            info['cond/%s'%key]  = cond
            info['count/%s'%key] = count
        save_data_h5(info, fpath)

    def _save_figure(self, fpath):
        fig = plt.figure(figsize=(16, 9))
        num = len(self._acc['noise']) + 1

        anoi = self._acc['noise_a']
        acon = self._acc['cond_a']
        acnt = self._acc['count_a']

        nlen = len(anoi)
        clen = len(acon)
        ax = fig.add_subplot(num//2+1, 1, 1)
        ax.title('Global Mean')
        ax.bar(np.arange(nlen), anoi/acnt, color='b')
        ax.bar(np.arange(nlen, nlen + clen), acon/acnt, color='r')
        ax.set_xlim(-1, nlen + clen)
        #ax.set_xticks([])
        #ax.set_yticks([])

        for i, [noise, cond, count] in list(zip(self._acc['noise'], self._acc['cond'], self._acc['count'])) + []:
            nlen = len(noise)
            clen = len(cond)
            ax = fig.add_subplot(num, 1, i+1)
            ax.bar(np.arange(nlen), noise/count, color='b')
            ax.bar(np.arange(nlen, nlen + clen), cond/count, color='r')
            ax.set_xlim(-1, nlen + clen)
            #ax.set_xticks([])
            #ax.set_yticks([])
        fig.tight_layout()
        fig.savefig(fpath)
        plt.close(fig)

    def _save_all(self):
        files = {
            'data'   : {
                'fpath'  : self.config.exsub(self.templ, data_desc='mean_cond_result', ftype='h5', **self.comkw),
                'func'   : self._save_data,
                'args'   : [],
                'kwargs' : {},
            },
            'figure' : {
                'fpath'  : self.config.exsub(self.templ, data_desc='mean_cond_relevance', ftype='png', **self.comkw),
                'func'   : self._save_figure,
                'args'   : [],
                'kwargs' : {},
            },
        }
        for key, val in files.items():
            self.check_write(val['fpath'], val['func'], val['args'], val['kwargs'])
            getLogger('ecGAN').info('Saved %s in \'%s\'.', 'mean cond relevance', fpath)

    def __exit__(self, exc_type, exc_value, traceback):
        if exc_type is not None:
            return
        self._save_all()
